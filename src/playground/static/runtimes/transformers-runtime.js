import { BrowserModelRuntime } from "./runtime.js";

const VERSION = "4.2.0";
const MODULE_URL = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${VERSION}`;

export class TransformersRuntime extends BrowserModelRuntime {
  constructor() {
    super();
    this.model = null;
    this.tokenizer = null;
    this.modelKey = null;
    this.TextStreamer = null;
    this.random = null;
    this.loadController = null;
  }

  async load(config, onProgress, signal) {
    const fileOptions = modelFileOptions(config);
    const modelKey = `${config.model}:${config.dtype}:${config.device}:${config.file ?? ""}`;
    if (this.model && this.modelKey === modelKey) {
      return;
    }
    await this.unload();
    onProgress({ phase: "runtime", message: "Loading Transformers.js" });
    const {
      AutoModelForCausalLM,
      AutoTokenizer,
      ModelRegistry,
      TextStreamer,
      env,
      random,
    } = await import(MODULE_URL);
    env.useBrowserCache = true;
    const cached = await ModelRegistry.is_cached(config.model, {
      device: config.device,
      dtype: config.dtype,
      ...fileOptions,
    });
    onProgress({
      phase: "runtime",
      message: cached ? "Loading model from browser cache" : "Caching model in browser",
    });

    const controller = new AbortController();
    this.loadController = controller;
    const abort = () => controller.abort(signal?.reason);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) {
      abort();
    }
    const previousFetch = env.fetch;
    env.fetch = (input, init = {}) =>
      fetch(input, { ...init, signal: controller.signal });
    const progress = (event) => renderTransformersProgress(event, onProgress);
    const options = {
      local_files_only: false,
      progress_callback: progress,
    };
    let model = null;
    try {
      const tokenizer = await AutoTokenizer.from_pretrained(config.model, options);
      model = await AutoModelForCausalLM.from_pretrained(config.model, {
        ...options,
        device: config.device,
        dtype: config.dtype,
        ...fileOptions,
      });
      this.tokenizer = tokenizer;
      this.model = model;
      this.TextStreamer = TextStreamer;
      this.random = random;
      this.modelKey = modelKey;
    } catch (error) {
      await model?.dispose?.();
      if (controller.signal.aborted) {
        throw new DOMException("Model loading cancelled", "AbortError");
      }
      throw error;
    } finally {
      env.fetch = previousFetch;
      signal?.removeEventListener("abort", abort);
      if (this.loadController === controller) {
        this.loadController = null;
      }
    }
  }

  async isCached(config) {
    const fileOptions = modelFileOptions(config);
    const modelKey = `${config.model}:${config.dtype}:${config.device}:${config.file ?? ""}`;
    if (this.model && this.modelKey === modelKey) {
      return true;
    }
    const { ModelRegistry, env } = await import(MODULE_URL);
    env.useBrowserCache = true;
    return ModelRegistry.is_cached(config.model, {
      device: config.device,
      dtype: config.dtype,
      ...fileOptions,
    });
  }

  async generate({ messages, settings = {} }) {
    if (!this.model || !this.tokenizer || !this.TextStreamer || !this.random) {
      throw new Error("Ternary Bonsai is not loaded");
    }
    if (settings.seed !== undefined && settings.seed !== null) {
      this.random.seed(Number(settings.seed));
    }
    const inputs = this.tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: true,
    });
    let text = "";
    const streamer = new this.TextStreamer(this.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (chunk) => {
        text += chunk;
      },
    });
    const temperature = Number(settings.temperature ?? 0);
    await this.model.generate({
      ...inputs,
      max_new_tokens: Number(settings.max_tokens ?? 48),
      do_sample: temperature > 0,
      temperature: temperature > 0 ? temperature : undefined,
      top_p: Number(settings.top_p ?? 0.9),
      streamer,
    });
    if (!text.trim()) {
      throw new Error("Transformers.js returned no text");
    }
    return text.trim();
  }

  async unload() {
    this.cancelLoad();
    await this.model?.dispose?.();
    this.model = null;
    this.tokenizer = null;
    this.TextStreamer = null;
    this.random = null;
    this.modelKey = null;
  }

  cancelLoad() {
    this.loadController?.abort();
  }
}

function modelFileOptions(config) {
  if (!config.file) {
    return {};
  }
  const path = config.file.replace(/^\/+/, "").replace(/\.onnx$/i, "");
  const separator = path.lastIndexOf("/");
  const subfolder = separator >= 0 ? path.slice(0, separator) : "";
  let modelFileName = separator >= 0 ? path.slice(separator + 1) : path;
  const dtypeSuffix = `_${config.dtype}`.toLowerCase();
  if (modelFileName.toLowerCase().endsWith(dtypeSuffix)) {
    modelFileName = modelFileName.slice(0, -dtypeSuffix.length);
  }
  return {
    ...(subfolder ? { subfolder } : {}),
    model_file_name: modelFileName,
  };
}

function renderTransformersProgress(event, onProgress) {
  if (event.status === "progress_total") {
    onProgress({
      phase: "download",
      loaded: event.loaded ?? event.progress ?? 0,
      total: event.total ?? 100,
      file: "Model cache",
    });
    return;
  }
  if (event.status === "progress") {
    onProgress({
      phase: "download",
      loaded: event.loaded,
      total: event.total,
      file: event.file,
    });
  }
}
