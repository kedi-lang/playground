import {
  BrowserModelRuntime,
  DEFAULT_CONTEXT_SOFT_LIMIT_PERCENT,
  completionPrompt,
  estimateMessagesTokens,
  nextContextSize,
} from "./runtime.js";

const VERSION = "3.5.1";
const MODULE_URL = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${VERSION}/esm/index.js`;
const WASM_URL = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${VERSION}/esm/wasm/wllama.wasm`;
const DEFAULT_CONTEXT_SIZE = 1024;

export class WllamaRuntime extends BrowserModelRuntime {
  constructor() {
    super();
    this.instance = null;
    this.modelKey = null;
    this.sourceKey = null;
    this.loadController = null;
    this.loadedConfig = null;
    this.onProgress = null;
    this.contextSize = 0;
    this.maximumContextSize = 0;
    this.contextSoftLimitPercent = DEFAULT_CONTEXT_SOFT_LIMIT_PERCENT;
  }

  async load(config, onProgress, signal) {
    const file = config.fileObject;
    const sourceKey = modelSourceKey(config);
    const contextSize = positiveInteger(config.contextWindow, DEFAULT_CONTEXT_SIZE);
    const modelKey = `${sourceKey}:context:${contextSize}`;
    if (this.instance && this.modelKey === modelKey) {
      return;
    }
    await this.unload();
    onProgress({ phase: "runtime", message: "Loading wllama" });
    const controller = new AbortController();
    this.loadController = controller;
    const abort = () => controller.abort(signal?.reason);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) {
      abort();
    }
    const instance = await createWllama();
    const params = {
      n_ctx: contextSize,
      n_threads: 1,
      n_gpu_layers: 999,
      ctx_shift: true,
      useCache: true,
      signal: controller.signal,
      progressCallback: ({ loaded, total }) => {
        onProgress({ phase: "download", loaded, total });
      },
    };
    try {
      if (config.source === "file") {
        if (!file) {
          throw new Error("Choose a GGUF file");
        }
        await instance.loadModel([file], params);
      } else {
        await removeInvalidCachedModel(instance, config);
        await loadCachedModel(instance, config, params);
        await markCachedModel(instance, config);
      }
      this.instance = instance;
      this.modelKey = modelKey;
      this.sourceKey = sourceKey;
      const context = instance.getLoadedContextInfo();
      this.contextSize = positiveInteger(context.n_ctx, contextSize);
      this.maximumContextSize = positiveInteger(context.n_ctx_train, this.contextSize);
      this.contextSoftLimitPercent = softLimitPercent(config.contextSoftLimitPercent);
      this.loadedConfig = { ...config, contextWindow: this.contextSize };
      this.onProgress = onProgress;
    } catch (error) {
      if (config.source !== "file") {
        clearCacheMarker(config);
        await removeInvalidCachedModel(instance, config);
      }
      await instance.exit?.();
      if (controller.signal.aborted) {
        throw new DOMException("Model loading cancelled", "AbortError");
      }
      throw error;
    } finally {
      signal?.removeEventListener("abort", abort);
      if (this.loadController === controller) {
        this.loadController = null;
      }
    }
  }

  async isCached(config) {
    if (this.instance && this.sourceKey === modelSourceKey(config)) {
      return true;
    }
    const inspector = await createWllama();
    try {
      const entries = await inspector.cacheManager.list();
      const markedSize = Number(localStorage.getItem(cacheMarkerKey(config)));
      return (
        Number.isFinite(markedSize) &&
        markedSize > 0 &&
        entries.some(
          (entry) =>
            matchesCachedModel(entry, config) && Number(entry.size) === markedSize,
        )
      );
    } finally {
      await inspector.exit?.();
    }
  }

  async generate({ messages, settings = {}, responseFormat = "text" }) {
    if (!this.instance) {
      throw new Error("Bonsai is not loaded");
    }
    const maxTokens = positiveInteger(settings.max_tokens, 48);
    await this.expandContextFor(
      estimateMessagesTokens(messages) + maxTokens,
    );
    try {
      return await this.generateOnce({
        messages,
        settings,
        responseFormat,
        maxTokens,
      });
    } catch (error) {
      const actualRequestTokens = contextRequestTokens(error);
      if (actualRequestTokens === null) {
        throw error;
      }
      const expanded = await this.expandContextFor(
        actualRequestTokens + maxTokens,
      );
      if (!expanded) {
        throw contextCapacityError(
          actualRequestTokens,
          maxTokens,
          this.contextSize,
          this.maximumContextSize,
        );
      }
      return this.generateOnce({
        messages,
        settings,
        responseFormat,
        maxTokens,
      });
    }
  }

  async generateOnce({ messages, settings, responseFormat, maxTokens }) {
    const generation = {
      max_tokens: maxTokens,
      temperature: Number(settings.temperature ?? 0.7),
      top_p: Number(settings.top_p ?? 0.95),
      stop: settings.stop_sequences,
    };
    if (settings.seed !== undefined && settings.seed !== null) {
      generation.seed = Number(settings.seed);
    }
    let result;
    if (typeof this.instance.createChatCompletion === "function") {
      try {
        const request = {
          messages,
          ...generation,
        };
        if (responseFormat === "json_object") {
          request.response_format = { type: "json_object" };
        }
        result = await this.instance.createChatCompletion(request);
      } catch (error) {
        if (contextRequestTokens(error) !== null) {
          throw error;
        }
        result = null;
      }
    }
    if (!result) {
      result = await this.instance.createCompletion({
        prompt: completionPrompt(messages),
        ...generation,
      });
    }
    return {
      text: completionText(result),
      usage: completionUsage(result),
      finishReason: result?.choices?.[0]?.finish_reason ?? "stop",
      model: result?.model ?? null,
      responseId: result?.id ?? null,
    };
  }

  async expandContextFor(requiredTokens) {
    const target = nextContextSize(
      this.contextSize,
      requiredTokens,
      this.contextSoftLimitPercent,
      this.maximumContextSize,
    );
    if (target <= this.contextSize || !this.loadedConfig) {
      return false;
    }
    const config = { ...this.loadedConfig, contextWindow: target };
    const onProgress = this.onProgress ?? (() => {});
    await this.load(config, onProgress);
    return true;
  }

  async unload() {
    this.cancelLoad();
    if (this.instance) {
      await this.instance.exit?.();
    }
    this.instance = null;
    this.modelKey = null;
    this.sourceKey = null;
    this.loadedConfig = null;
    this.onProgress = null;
    this.contextSize = 0;
    this.maximumContextSize = 0;
  }

  cancelLoad() {
    this.loadController?.abort();
  }
}

function modelSourceKey(config) {
  const file = config.fileObject;
  return config.source === "file"
    ? `file:${file?.name}:${file?.size}:${file?.lastModified}`
    : `model:${config.id}`;
}

function modelUrl(config) {
  return `https://huggingface.co/${config.repo}/resolve/main/${config.file}`;
}

async function createWllama() {
  const { Wllama } = await import(MODULE_URL);
  return new Wllama(
    { default: WASM_URL },
    {
      logger: {
        debug: () => {},
        log: () => {},
        warn: () => {},
        error: () => {},
      },
    },
  );
}

function matchesCachedModel(entry, config) {
  const originalUrl = entry.metadata?.originalURL ?? "";
  return (
    (originalUrl.includes(`/${config.repo}/`) &&
      originalUrl.includes(`/${config.file}`)) ||
    entry.name.includes(config.file)
  );
}

async function loadCachedModel(instance, config, params) {
  try {
    await instance.loadModelFromUrl(modelUrl(config), params);
  } catch (error) {
    if (!String(error?.message).startsWith("Model file not found:")) {
      throw error;
    }
    await removeCachedModel(instance, config);
    await instance.loadModelFromUrl(modelUrl(config), params);
  }
}

async function removeInvalidCachedModel(instance, config) {
  const entries = await instance.cacheManager.list();
  const markedSize = Number(localStorage.getItem(cacheMarkerKey(config)));
  for (const entry of entries) {
    if (
      matchesCachedModel(entry, config) &&
      (!Number.isFinite(markedSize) ||
        markedSize <= 0 ||
        Number(entry.size) !== markedSize ||
        entry.size !== entry.metadata?.originalSize)
    ) {
      clearCacheMarker(config);
      await instance.cacheManager.delete(entry.name);
    }
  }
}

async function removeCachedModel(instance, config) {
  clearCacheMarker(config);
  await instance.cacheManager.deleteMany((entry) => matchesCachedModel(entry, config));
}

async function markCachedModel(instance, config) {
  const entries = await instance.cacheManager.list();
  const entry = entries.find((candidate) => matchesCachedModel(candidate, config));
  if (!entry || !Number.isFinite(Number(entry.size)) || Number(entry.size) <= 0) {
    throw new Error("Loaded GGUF is missing from the browser cache");
  }
  localStorage.setItem(cacheMarkerKey(config), String(entry.size));
}

function clearCacheMarker(config) {
  localStorage.removeItem(cacheMarkerKey(config));
}

function cacheMarkerKey(config) {
  return `kedi:wllama-cache-complete:${config.repo}:${config.file}`;
}

function completionText(result) {
  if (typeof result === "string" && result.trim()) {
    return result.trim();
  }
  const text =
    result?.choices?.[0]?.message?.content ??
    result?.choices?.[0]?.text ??
    result?.text ??
    result?.content ??
    result?.completion;
  if (typeof text !== "string" || !text.trim()) {
    throw new Error("wllama returned no text");
  }
  return text.trim();
}

function completionUsage(result) {
  const usage = result?.usage;
  if (!usage) {
    return null;
  }
  const inputTokens = Number(usage.prompt_tokens ?? usage.input_tokens ?? 0);
  const outputTokens = Number(usage.completion_tokens ?? usage.output_tokens ?? 0);
  return {
    inputTokens: Number.isFinite(inputTokens) ? inputTokens : 0,
    outputTokens: Number.isFinite(outputTokens) ? outputTokens : 0,
  };
}

function contextRequestTokens(error) {
  const match = String(error?.message ?? error).match(
    /request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)/i,
  );
  return match ? Number(match[1]) : null;
}

function contextCapacityError(requestTokens, maxTokens, contextSize, maximumSize) {
  return new Error(
    `WebGPU request needs ${requestTokens + maxTokens} tokens including the output reserve, ` +
      `but this model can provide at most ${maximumSize || contextSize} context tokens`,
  );
}

function softLimitPercent(value) {
  const percent = Number(value);
  return Number.isFinite(percent) && percent > 0 && percent < 100
    ? percent
    : DEFAULT_CONTEXT_SOFT_LIMIT_PERCENT;
}

function positiveInteger(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : fallback;
}
