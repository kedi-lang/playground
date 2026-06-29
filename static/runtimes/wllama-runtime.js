import { BrowserModelRuntime, completionPrompt } from "./runtime.js";

const VERSION = "3.5.1";
const MODULE_URL = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${VERSION}/esm/index.js`;
const WASM_URL = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${VERSION}/esm/wasm/wllama.wasm`;

export class WllamaRuntime extends BrowserModelRuntime {
  constructor() {
    super();
    this.instance = null;
    this.modelKey = null;
    this.loadController = null;
  }

  async load(config, onProgress, signal) {
    const file = config.fileObject;
    const modelKey =
      config.source === "file"
        ? `file:${file?.name}:${file?.size}:${file?.lastModified}`
        : `model:${config.id}`;
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
    const { Wllama } = await import(MODULE_URL);
    const instance = new Wllama(
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
    const params = {
      n_ctx: 512,
      n_threads: 1,
      n_gpu_layers: 999,
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
      }
      this.instance = instance;
      this.modelKey = modelKey;
    } catch (error) {
      if (config.source !== "file") {
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

  async generate({ messages, settings = {} }) {
    if (!this.instance) {
      throw new Error("Bonsai is not loaded");
    }
    const generation = {
      max_tokens: Number(settings.max_tokens ?? 48),
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
        result = await this.instance.createChatCompletion({
          messages,
          ...generation,
          response_format: { type: "json_object" },
        });
      } catch {
        result = null;
      }
    }
    if (!result) {
      result = await this.instance.createCompletion({
        prompt: completionPrompt(messages),
        ...generation,
      });
    }
    return completionText(result);
  }

  async unload() {
    this.cancelLoad();
    if (this.instance) {
      await this.instance.exit?.();
    }
    this.instance = null;
    this.modelKey = null;
  }

  cancelLoad() {
    this.loadController?.abort();
  }
}

function modelUrl(config) {
  return `https://huggingface.co/${config.repo}/resolve/main/${config.file}`;
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
  for (const entry of entries) {
    const originalUrl = entry.metadata?.originalURL ?? "";
    const matchesModel =
      originalUrl.includes(`/${config.repo}/`) &&
      originalUrl.includes(`/${config.file}`);
    if (
      matchesModel &&
      entry.size !== entry.metadata?.originalSize
    ) {
      await instance.cacheManager.delete(entry.name);
    }
  }
}

async function removeCachedModel(instance, config) {
  await instance.cacheManager.deleteMany((entry) => {
    const originalUrl = entry.metadata?.originalURL ?? "";
    return (
      (originalUrl.includes(`/${config.repo}/`) &&
        originalUrl.includes(`/${config.file}`)) ||
      entry.name.includes(config.file)
    );
  });
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
