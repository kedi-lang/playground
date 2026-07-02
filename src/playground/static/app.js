import { AdapterClient, fetchJSON } from "./adapter-client.js";
import {
  clearKediExecutionDiagnostic,
  createKediEditor,
  setKediExecutionDiagnostic,
} from "./kedi-editor.js";
import { MODEL_REGISTRY, modelConfig as builtinModelConfig } from "./model-registry.js";
import { PyodideRuntime } from "./pyodide-runtime.js";
import { assertWebGPU, formatBytes } from "./runtimes/runtime.js";
import { TransformersRuntime } from "./runtimes/transformers-runtime.js";
import { WllamaRuntime } from "./runtimes/wllama-runtime.js";

const DEFAULT_SOURCE = `>> Give me a [country]
>> What's the [capital] of <country>?
= Capital of <country> is <capital>.`;
const ENV_KEY = "kedi.playground.env";
const SESSION_KEY = "kedi.playground.session";
const DEFAULT_MODEL_SETTINGS = Object.freeze({
  temperature: 0.7,
  max_tokens: 48,
  top_p: 0.95,
  seed: null,
});

const ui = {
  source: document.querySelector("#source"),
  output: document.querySelector("#output"),
  error: document.querySelector("#error"),
  model: document.querySelector("#model"),
  byokControls: document.querySelector("#byok-controls"),
  localControls: document.querySelector("#local-controls"),
  byokModel: document.querySelector("#byok-model"),
  byokCustomModelField: document.querySelector("#byok-custom-model-field"),
  byokCustomModel: document.querySelector("#byok-custom-model"),
  sourceControls: document.querySelector("#model-source-controls"),
  modelFile: document.querySelector("#model-file"),
  modelFileName: document.querySelector("#model-file-name"),
  filePicker: document.querySelector("#file-picker"),
  downloadLocal: document.querySelector("#download-local"),
  run: document.querySelector("#run"),
  cancel: document.querySelector("#cancel"),
  status: document.querySelector("#status"),
  progress: document.querySelector("#progress"),
  timing: document.querySelector("#timing"),
  statusDot: document.querySelector(".status-dot"),
  envDialog: document.querySelector("#env-dialog"),
  envFeedback: document.querySelector("#env-feedback"),
  customEnvList: document.querySelector("#custom-env-list"),
  runSettingsPanel: document.querySelector("#run-settings-panel"),
  modelSettingsPanel: document.querySelector("#model-settings-panel"),
  modelsPanel: document.querySelector("#models-panel"),
  settingTemperature: document.querySelector("#setting-temperature"),
  settingMaxTokens: document.querySelector("#setting-max-tokens"),
  settingTopP: document.querySelector("#setting-top-p"),
  settingSeed: document.querySelector("#setting-seed"),
  resetModelSettings: document.querySelector("#reset-model-settings"),
  byokModelLabel: document.querySelector("#byok-model-label"),
  byokModelId: document.querySelector("#byok-model-id"),
  addByokModel: document.querySelector("#add-byok-model"),
  byokModelFeedback: document.querySelector("#byok-model-feedback"),
  byokModelList: document.querySelector("#byok-model-list"),
  browserModelLabel: document.querySelector("#browser-model-label"),
  browserModelRepo: document.querySelector("#browser-model-repo"),
  browserModelFile: document.querySelector("#browser-model-file"),
  browserModelDtype: document.querySelector("#browser-model-dtype"),
  addBrowserModel: document.querySelector("#add-browser-model"),
  browserModelFeedback: document.querySelector("#browser-model-feedback"),
  browserModelList: document.querySelector("#browser-model-list"),
  stdinForm: document.querySelector("#stdin-form"),
  stdinInput: document.querySelector("#stdin-input"),
  stdinEof: document.querySelector("#stdin-eof"),
};

let activeRuntime = null;
let activeRuntimeKey = null;
let activeRun = null;
let activeDownload = null;
let activeModelSource = "cache";
let cacheCheckId = 0;
let pendingStdin = null;
const pythonRuntime = new PyodideRuntime(setStatus, browserIo());

const initialSession = sessionValues();
const sourceEditor = await createKediEditor(
  ui.source,
  initialSession.source || DEFAULT_SOURCE,
  (source) => saveSession({ source }),
);
let customByokModels = Array.isArray(initialSession.customByokModels)
  ? initialSession.customByokModels.filter(isByokModel)
  : [];
let customBrowserModels = Array.isArray(initialSession.customBrowserModels)
  ? initialSession.customBrowserModels.filter(isBrowserModel)
  : [];
let activeByokModelSource =
  initialSession.byokModelSource === "custom" && customByokModels.length
    ? "custom"
    : "builtin";
renderBrowserModelOptions(initialSession.browserModel);
renderCustomByokModels(initialSession.byokCustomModel);
renderCustomBrowserModels();
applyModelSettings(initialSession.modelSettings);
const initialControlTab =
  initialSession.controlTab === "model" ? "settings" : initialSession.controlTab;
setControlTab(["run", "settings", "models"].includes(initialControlTab) ? initialControlTab : "run");
installHuggingFaceAuth();
bindEvents();
const initialMode = initialSession.mode === "byok" ? "byok" : "local";
document.querySelector(`input[name="mode"][value="${initialMode}"]`).checked = true;
setMode(initialMode);
void pythonRuntime.preload().then(
  () => {
    if (!activeRun) {
      setStatus("Ready");
    }
  },
  (error) => {
    if (!activeRun) {
      setStatus("Sandbox unavailable");
      setProgress(error?.message ?? String(error));
    }
  },
);
loadByokModels();

function bindEvents() {
  for (const tab of document.querySelectorAll("[data-control-tab]")) {
    tab.addEventListener("click", () => {
      setControlTab(tab.dataset.controlTab);
      saveSession({ controlTab: tab.dataset.controlTab });
    });
  }
  for (const input of document.querySelectorAll('input[name="mode"]')) {
    input.addEventListener("change", () => {
      setMode(input.value);
      saveSession({ mode: input.value });
    });
  }
  ui.model.addEventListener("change", handleBrowserModelChange);
  ui.modelFile.addEventListener("change", () => {
    const file = ui.modelFile.files?.[0];
    ui.modelFileName.textContent = file?.name || "";
    if (file) {
      activeModelSource = "file";
      setStatus("Local GGUF selected");
      setProgress("");
      updateSourceControls();
    }
  });
  ui.downloadLocal.addEventListener("click", cacheModel);
  ui.run.addEventListener("click", run);
  ui.cancel.addEventListener("click", cancel);
  ui.stdinForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitStdin(ui.stdinInput.value);
  });
  ui.stdinEof.addEventListener("click", () => submitStdin(null));
  document.querySelector("#open-env").addEventListener("click", openEnv);
  document.querySelector("#close-env").addEventListener("click", closeEnv);
  document.querySelector("#save-env").addEventListener("click", saveEnv);
  document.querySelector("#clear-env").addEventListener("click", clearEnv);
  document.querySelector("#add-env").addEventListener("click", () => addCustomEnvRow());
  for (const input of ui.envDialog.querySelectorAll(
    '[data-env="LOGFIRE_ENABLED"], [data-env="LOGFIRE_TOKEN"]',
  )) {
    input.addEventListener("input", clearEnvFeedback);
  }
  ui.byokModel.addEventListener("focus", () => activateByokModelSource("builtin"));
  ui.byokModel.addEventListener("change", () => {
    activateByokModelSource("builtin");
    saveSession({
      byokBuiltinModel: ui.byokModel.value,
    });
  });
  ui.byokCustomModel.addEventListener("focus", () =>
    activateByokModelSource("custom"),
  );
  ui.byokCustomModel.addEventListener("change", () => {
    activateByokModelSource("custom");
    saveSession({
      byokCustomModel: ui.byokCustomModel.value,
    });
  });
  document.querySelector("#open-models-tab").addEventListener("click", () => {
    setControlTab("models");
    saveSession({ controlTab: "models" });
    ui.byokModelLabel.focus();
  });
  ui.addByokModel.addEventListener("click", addByokModel);
  ui.addBrowserModel.addEventListener("click", addBrowserModel);
  ui.browserModelFile.addEventListener("input", updateBrowserDtypeState);
  for (const [, , input] of modelSettingFields()) {
    input.addEventListener("change", saveModelSettings);
  }
  ui.resetModelSettings.addEventListener("click", () => {
    applyModelSettings();
    saveModelSettings();
  });
}

function selectedMode() {
  return document.querySelector('input[name="mode"]:checked')?.value ?? "local";
}

function setMode(mode) {
  ui.localControls.hidden = mode !== "local";
  ui.byokControls.hidden = mode !== "byok";
  ui.sourceControls.hidden = mode !== "local";
  if (mode === "local") {
    updateSourceControls();
  }
  setStatus("Ready");
  setProgress("");
}

function setControlTab(tabName) {
  const activeTab = ["run", "settings", "models"].includes(tabName) ? tabName : "run";
  ui.runSettingsPanel.hidden = activeTab !== "run";
  ui.modelSettingsPanel.hidden = activeTab !== "settings";
  ui.modelsPanel.hidden = activeTab !== "models";
  for (const tab of document.querySelectorAll("[data-control-tab]")) {
    const active = tab.dataset.controlTab === activeTab;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  }
}

async function run() {
  const started = performance.now();
  setRunning(true);
  clearResult();
  try {
    const result = selectedMode() === "local" ? await runLocal() : await runByok();
    ui.output.textContent = result.result;
    setStatus("Complete");
  } catch (error) {
    ui.error.hidden = false;
    ui.error.textContent = error?.message ?? String(error);
    setKediExecutionDiagnostic(sourceEditor, error?.diagnostic);
    setStatus("Failed");
  } finally {
    cancelStdin();
    ui.timing.textContent = `${Math.round(performance.now() - started)}ms`;
    setRunning(false);
    activeRun = null;
  }
}

async function runLocal() {
  await assertWebGPU();
  const modelId = ui.model.value;
  const baseConfig = browserModelConfig(modelId);
  const config = selectedRuntimeConfig(baseConfig);
  const runtime = await runtimeFor(config);
  const values = envValues();
  validateLogfireEnvironment(values);
  const secrets = Object.fromEntries(
    ["LOGFIRE_ENABLED", "LOGFIRE_TOKEN"]
      .filter((name) => typeof values[name] === "string" && values[name])
      .map((name) => [name, values[name]]),
  );
  const runId = crypto.randomUUID();
  const controller = new AbortController();
  activeRun = { runId, controller, bridgeStarted: false };
  setStatus("Loading model");
  await runtime.load(config, renderProgress, controller.signal);
  if (config.source === "cache") {
    setDownloadState(true);
  }
  setProgress("Model loaded");
  setStatus("Running Kedi");

  const client = new AdapterClient(runtime, setStatus, browserIo(), pythonRuntime);
  const bridge = client.serve(runId, controller.signal);
  activeRun.bridgeStarted = true;
  try {
    const result = await fetchJSON("/api/run/local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: sourceEditor.getValue(),
        modelId,
        modelConfig: MODEL_REGISTRY[modelId] ? null : baseConfig,
        runId,
        secrets,
        settings: modelSettings(),
      }),
    });
    controller.abort();
    await bridge;
    return result;
  } finally {
    controller.abort();
  }
}

async function runByok() {
  const model =
    activeByokModelSource === "custom"
      ? ui.byokCustomModel.value
      : ui.byokModel.value;
  if (!model) {
    throw new Error("Enter a BYOK model");
  }
  const values = envValues();
  validateLogfireEnvironment(values);
  const secrets = Object.fromEntries(
    Object.entries(values).filter(
      ([name, value]) => name !== "HF_TOKEN" && typeof value === "string" && value,
    ),
  );
  setStatus("Running Kedi");
  const runId = crypto.randomUUID();
  const controller = new AbortController();
  activeRun = { runId, controller, bridgeStarted: true };
  const client = new AdapterClient(null, setStatus, browserIo(), pythonRuntime);
  const bridge = client.serve(runId, controller.signal);
  try {
    const result = await fetchJSON("/api/run/byok", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: sourceEditor.getValue(),
        model,
        runId,
        secrets,
        settings: modelSettings(),
      }),
      signal: controller.signal,
    });
    controller.abort();
    await bridge;
    return result;
  } finally {
    controller.abort();
  }
}

function validateLogfireEnvironment(values) {
  if (values.LOGFIRE_ENABLED === "true" && !values.LOGFIRE_TOKEN) {
    throw new Error("LOGFIRE_TOKEN is required when Logfire is enabled");
  }
}

async function cancel() {
  if (activeDownload) {
    const download = activeDownload;
    activeDownload = null;
    download.controller.abort();
    activeRuntime?.cancelLoad?.();
    setStatus("Cancelled");
    setProgress("");
    setRunning(false);
    return;
  }
  if (!activeRun) {
    return;
  }
  cancelStdin();
  activeRun.controller.abort();
  activeRuntime?.cancelLoad?.();
  if (activeRun.bridgeStarted) {
    try {
      await fetchJSON("/api/run/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ runId: activeRun.runId }),
      });
    } catch {
      // The run can finish between the local abort and the cancellation request.
    }
  }
  setStatus("Cancelled");
  setRunning(false);
}

async function handleBrowserModelChange() {
  if (activeRuntime) {
    await activeRuntime.unload();
    activeRuntime = null;
    activeRuntimeKey = null;
  }
  activeModelSource = "cache";
  ui.modelFile.value = "";
  ui.modelFileName.textContent = "";
  setStatus("Ready");
  setProgress("");
  updateSourceControls();
  saveSession({ browserModel: ui.model.value });
}

function browserModelConfig(modelId) {
  const custom = customBrowserModels.find((model) => model.id === modelId);
  return custom ?? builtinModelConfig(modelId);
}

function renderBrowserModelOptions(preferredId) {
  const selectedId = preferredId || ui.model.value;
  ui.model.replaceChildren();

  const builtinGroup = document.createElement("optgroup");
  builtinGroup.label = "Builtin models";
  for (const model of Object.values(MODEL_REGISTRY)) {
    builtinGroup.append(new Option(model.label, model.id));
  }
  ui.model.append(builtinGroup);

  if (customBrowserModels.length) {
    const customGroup = document.createElement("optgroup");
    customGroup.label = "Custom WebGPU models";
    for (const model of customBrowserModels) {
      customGroup.append(new Option(model.label, model.id));
    }
    ui.model.append(customGroup);
  }

  const preferred = [...ui.model.options].find((option) => option.value === selectedId);
  if (preferred) {
    ui.model.value = preferred.value;
  }
}

function renderCustomBrowserModels() {
  ui.browserModelList.replaceChildren();
  for (const model of customBrowserModels) {
    appendModelRow(
      ui.browserModelList,
      model.label,
      `${model.engine} | ${model.repo}/${model.file}`,
      () => removeBrowserModel(model.id),
    );
  }
}

async function addBrowserModel() {
  const label = ui.browserModelLabel.value.trim();
  const repo = ui.browserModelRepo.value.trim();
  const file = ui.browserModelFile.value.trim();
  const lowerFile = file.toLowerCase();
  if (!label || !repo || !file) {
    setModelFeedback(ui.browserModelFeedback, "Display name, repo, and file are required.", "error");
    return;
  }
  const engine = lowerFile.endsWith(".gguf")
    ? "wllama"
    : lowerFile.endsWith(".onnx")
      ? "transformers"
      : null;
  if (!engine) {
    setModelFeedback(ui.browserModelFeedback, "Model file must end in .gguf or .onnx.", "error");
    return;
  }

  const dtype = ui.browserModelDtype.value;
  const existing = customBrowserModels.find(
    (model) =>
      model.repo === repo &&
      model.file === file &&
      (engine === "wllama" || model.dtype === dtype),
  );
  const model = {
    id: existing?.id ?? `custom-webgpu-${crypto.randomUUID()}`,
    kind: "webgpu",
    label,
    engine,
    repo,
    file,
    device: "webgpu",
    ...(engine === "transformers" ? { model: repo, dtype } : {}),
  };
  customBrowserModels = existing
    ? customBrowserModels.map((item) => (item.id === existing.id ? model : item))
    : [...customBrowserModels, model];
  saveSession({ customBrowserModels, browserModel: model.id });
  renderBrowserModelOptions(model.id);
  renderCustomBrowserModels();
  await handleBrowserModelChange();
  ui.browserModelLabel.value = "";
  ui.browserModelRepo.value = "";
  ui.browserModelFile.value = "";
  updateBrowserDtypeState();
  setModelFeedback(ui.browserModelFeedback, `${label} added.`, "success");
}

async function removeBrowserModel(modelId) {
  const removingSelected = ui.model.value === modelId;
  customBrowserModels = customBrowserModels.filter((model) => model.id !== modelId);
  saveSession({ customBrowserModels });
  renderBrowserModelOptions(removingSelected ? undefined : ui.model.value);
  renderCustomBrowserModels();
  if (removingSelected) {
    await handleBrowserModelChange();
  }
}

function updateBrowserDtypeState() {
  const gguf = ui.browserModelFile.value.trim().toLowerCase().endsWith(".gguf");
  ui.browserModelDtype.disabled = gguf;
}

async function runtimeFor(config) {
  const key = runtimeKey(config);
  if (activeRuntime && activeRuntimeKey === key) {
    return activeRuntime;
  }
  await activeRuntime?.unload();
  activeRuntime = createRuntime(config);
  activeRuntimeKey = key;
  return activeRuntime;
}

function selectedRuntimeConfig(config) {
  if (activeModelSource === "file") {
    const fileObject = ui.modelFile.files?.[0];
    if (!fileObject) {
      throw new Error("Choose a GGUF file");
    }
    return {
      ...config,
      engine: "wllama",
      file: fileObject.name,
      source: "file",
      fileObject,
    };
  }
  return { ...config, source: "cache" };
}

function updateSourceControls() {
  ui.sourceControls.hidden = selectedMode() !== "local";
  ui.modelFile.disabled = false;
  ui.filePicker.hidden = false;
  ui.filePicker.dataset.active = String(activeModelSource === "file");
  void refreshDownloadState();
}

function createRuntime(config) {
  return config.engine === "wllama" ? new WllamaRuntime() : new TransformersRuntime();
}

function runtimeKey(config) {
  if (config.source === "file" && config.fileObject) {
    const file = config.fileObject;
    return `wllama:file:${file.name}:${file.size}:${file.lastModified}`;
  }
  return `${config.engine}:cache:${config.id}`;
}

async function refreshDownloadState() {
  if (selectedMode() !== "local") {
    return;
  }
  const checkId = ++cacheCheckId;
  const config = browserModelConfig(ui.model.value);
  setDownloadState(null);
  const cacheConfig = { ...config, source: "cache" };
  const runtime =
    activeRuntime && activeRuntimeKey === runtimeKey(cacheConfig)
      ? activeRuntime
      : createRuntime(cacheConfig);
  const temporary = runtime !== activeRuntime;
  try {
    const cached = await runtime.isCached(config);
    if (checkId === cacheCheckId) {
      setDownloadState(cached);
    }
  } catch {
    if (checkId === cacheCheckId) {
      setDownloadState(false);
    }
  } finally {
    if (temporary) {
      await runtime.unload();
    }
  }
}

function setDownloadState(cached) {
  if (cached === null) {
    ui.downloadLocal.textContent = "Checking";
    ui.downloadLocal.dataset.state = "checking";
    ui.downloadLocal.disabled = true;
    return;
  }
  ui.downloadLocal.textContent = cached ? "Downloaded" : "Download";
  ui.downloadLocal.dataset.state = cached ? "downloaded" : "download";
  ui.downloadLocal.disabled = false;
}

async function cacheModel() {
  const config = browserModelConfig(ui.model.value);
  activeModelSource = "cache";
  ui.filePicker.dataset.active = "false";
  if (ui.downloadLocal.dataset.state === "downloaded") {
    setStatus("Using downloaded model");
    setProgress("");
    return;
  }
  const download = {
    controller: new AbortController(),
  };
  activeDownload = download;
  ui.downloadLocal.disabled = true;
  setRunning(true);
  setStatus("Caching model in this browser");
  setProgress("Checking browser cache");
  try {
    const runtime = await runtimeFor(config);
    await runtime.load(
      { ...config, source: "cache" },
      renderProgress,
      download.controller.signal,
    );
    setStatus("Model loaded");
    setProgress("Stored in this browser");
    setDownloadState(true);
  } catch (error) {
    if (error?.name === "AbortError") {
      return;
    }
    setStatus("Download failed");
    setProgress(error?.message ?? String(error));
  } finally {
    if (activeDownload === download) {
      activeDownload = null;
      setRunning(false);
    }
    void refreshDownloadState();
  }
}

function renderProgress(event) {
  if (event.phase === "download") {
    const label = event.file ? `${event.file} ` : "";
    if (event.total) {
      const percent = Math.round((event.loaded / event.total) * 100);
      setProgress(
        `${label}${percent}% ${formatBytes(event.loaded)} / ${formatBytes(event.total)}`,
      );
    } else {
      setProgress(`${label}${formatBytes(event.loaded || 0)}`);
    }
    return;
  }
  setProgress(event.message || "Loading model");
}

function setRunning(running) {
  ui.run.disabled = running;
  ui.cancel.disabled = !running;
}

function clearResult() {
  clearKediExecutionDiagnostic(sourceEditor);
  ui.output.textContent = "";
  ui.error.textContent = "";
  ui.error.hidden = true;
  ui.timing.textContent = "";
  closeStdin();
}

function browserIo() {
  return {
    onStdout: appendProgramOutput,
    onStderr: appendProgramOutput,
    onStdin: requestStdin,
  };
}

function appendProgramOutput(value) {
  ui.output.textContent += value;
  ui.output.scrollTop = ui.output.scrollHeight;
}

function requestStdin() {
  if (pendingStdin) {
    return Promise.reject(new Error("A stdin request is already active"));
  }
  setStatus("Waiting for input");
  ui.stdinForm.hidden = false;
  ui.stdinInput.value = "";
  ui.stdinInput.focus();
  return new Promise((resolve, reject) => {
    pendingStdin = { resolve, reject };
  });
}

function submitStdin(value) {
  if (!pendingStdin) {
    return;
  }
  const request = pendingStdin;
  pendingStdin = null;
  closeStdin();
  setStatus("Running Python sandbox");
  request.resolve(value);
}

function cancelStdin() {
  if (!pendingStdin) {
    closeStdin();
    return;
  }
  const request = pendingStdin;
  pendingStdin = null;
  closeStdin();
  request.reject(new Error("Standard input was cancelled"));
}

function closeStdin() {
  ui.stdinForm.hidden = true;
  ui.stdinInput.value = "";
}

function setStatus(message) {
  ui.status.textContent = message;
  const normalized = message.toLowerCase();
  ui.statusDot.dataset.state =
    normalized.includes("fail") || normalized.includes("error")
      ? "error"
      : normalized.includes("ready") ||
          normalized.includes("complete") ||
          normalized.includes("loaded")
        ? "ready"
        : normalized.includes("cancel")
          ? "idle"
          : "busy";
}

function setProgress(message) {
  ui.progress.textContent = message;
  ui.progress.hidden = !message;
}

function envValues() {
  try {
    const parsed = JSON.parse(localStorage.getItem(ENV_KEY) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function sessionValues() {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(SESSION_KEY) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveSession(patch) {
  sessionStorage.setItem(
    SESSION_KEY,
    JSON.stringify({ ...sessionValues(), ...patch }),
  );
}

function modelSettingFields() {
  return [
    ["temperature", "Temperature", ui.settingTemperature],
    ["max_tokens", "Max tokens", ui.settingMaxTokens],
    ["top_p", "Top P", ui.settingTopP],
    ["seed", "Seed", ui.settingSeed],
  ];
}

function applyModelSettings(settings = {}) {
  const saved = settings && typeof settings === "object" ? settings : {};
  const values = { ...DEFAULT_MODEL_SETTINGS, ...saved };
  for (const [name, , input] of modelSettingFields()) {
    input.value = values[name] ?? "";
  }
}

function modelSettings() {
  const settings = {};
  for (const [name, label, input] of modelSettingFields()) {
    if (!input.value.trim()) {
      continue;
    }
    if (!input.checkValidity()) {
      throw new Error(`${label} is outside the supported range`);
    }
    const value = Number(input.value);
    if (!Number.isFinite(value)) {
      throw new Error(`${label} must be a number`);
    }
    settings[name] = value;
  }
  return settings;
}

function saveModelSettings() {
  try {
    saveSession({ modelSettings: modelSettings() });
    setStatus("Ready");
    setProgress("");
  } catch (error) {
    setStatus("Invalid model settings");
    setProgress(error?.message ?? String(error));
  }
}

function openEnv() {
  const values = envValues();
  for (const input of ui.envDialog.querySelectorAll("[data-env]")) {
    if (input.type === "checkbox") {
      input.checked = values[input.dataset.env] === "true";
    } else {
      input.value = values[input.dataset.env] || "";
    }
  }
  renderCustomEnv(values);
  clearEnvFeedback();
  ui.envDialog.showModal();
}

function saveEnv() {
  const values = {};
  for (const input of ui.envDialog.querySelectorAll("[data-env]")) {
    if (input.type === "checkbox") {
      values[input.dataset.env] = input.checked ? "true" : "false";
    } else if (input.value.trim()) {
      values[input.dataset.env] = input.value.trim();
    }
  }
  for (const row of ui.customEnvList.querySelectorAll(".custom-env-row")) {
    const name = row.querySelector("[data-custom-name]").value.trim();
    const value = row.querySelector("[data-custom-value]").value.trim();
    if (name && value) {
      values[name] = value;
    }
  }
  if (values.LOGFIRE_ENABLED === "true" && !values.LOGFIRE_TOKEN) {
    setEnvFeedback("Add LOGFIRE_TOKEN before enabling Logfire.");
    ui.envDialog.querySelector('[data-env="LOGFIRE_TOKEN"]').focus();
    return;
  }
  localStorage.setItem(ENV_KEY, JSON.stringify(values));
  clearEnvFeedback();
  ui.envDialog.close();
  setStatus("Environment saved");
}

function closeEnv() {
  clearEnvFeedback();
  ui.envDialog.close();
}

function setEnvFeedback(message) {
  ui.envFeedback.textContent = message;
  ui.envFeedback.dataset.state = "error";
  ui.envFeedback.hidden = false;
}

function clearEnvFeedback() {
  ui.envFeedback.textContent = "";
  ui.envFeedback.hidden = true;
}

function clearEnv() {
  localStorage.removeItem(ENV_KEY);
  for (const input of ui.envDialog.querySelectorAll("[data-env]")) {
    input.value = "";
    input.checked = false;
  }
  ui.customEnvList.replaceChildren();
  clearEnvFeedback();
  setStatus("Environment cleared");
}

function isByokModel(model) {
  return (
    model &&
    typeof model === "object" &&
    (model.kind === undefined || model.kind === "byok") &&
    typeof model.label === "string" &&
    typeof model.model === "string"
  );
}

function isBrowserModel(model) {
  return (
    model &&
    typeof model === "object" &&
    (model.kind === undefined || model.kind === "webgpu") &&
    typeof model.id === "string" &&
    typeof model.label === "string" &&
    typeof model.repo === "string" &&
    typeof model.file === "string" &&
    (model.engine === "wllama" || model.engine === "transformers")
  );
}

async function loadByokModels() {
  try {
    const payload = await fetchJSON("/api/byok/models");
    for (const model of payload.models) {
      ui.byokModel.add(new Option(model.label, model.id));
    }
    const preferredId =
      initialSession.byokBuiltinModel ||
      initialSession.byokModel ||
      "openai:gpt-4o-mini";
    const preferred = [...ui.byokModel.options].find(
      (option) => option.value === preferredId,
    );
    if (preferred) {
      ui.byokModel.value = preferred.value;
    }
    updateByokModelSelection();
  } catch (error) {
    setStatus("Model list failed");
    setProgress(error?.message ?? String(error));
  }
}

async function addByokModel() {
  const label = ui.byokModelLabel.value.trim();
  const modelId = ui.byokModelId.value.trim();
  if (!label || !modelId) {
    setModelFeedback(ui.byokModelFeedback, "Display name and model ID are required.", "error");
    return;
  }

  ui.addByokModel.disabled = true;
  setModelFeedback(ui.byokModelFeedback, "Checking provider...");
  try {
    const result = await fetchJSON("/api/byok/models/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelId }),
    });
    const model = { kind: "byok", label, model: result.model };
    const existing = customByokModels.find((item) => item.model === result.model);
    customByokModels = existing
      ? customByokModels.map((item) => (item.model === result.model ? model : item))
      : [...customByokModels, model];
    activeByokModelSource = "custom";
    saveSession({
      customByokModels,
      byokModelSource: "custom",
      byokCustomModel: result.model,
    });
    renderCustomByokModels(result.model);
    ui.byokModelLabel.value = "";
    ui.byokModelId.value = "";
    setModelFeedback(ui.byokModelFeedback, `${label} added.`, "success");
  } catch (error) {
    setModelFeedback(
      ui.byokModelFeedback,
      error?.message ?? String(error),
      "error",
    );
  } finally {
    ui.addByokModel.disabled = false;
  }
}

function removeByokModel(modelId) {
  const removingSelected =
    activeByokModelSource === "custom" && ui.byokCustomModel.value === modelId;
  customByokModels = customByokModels.filter((model) => model.model !== modelId);
  if (removingSelected || !customByokModels.length) {
    activeByokModelSource = "builtin";
  }
  saveSession({
    customByokModels,
    byokModelSource: activeByokModelSource,
  });
  renderCustomByokModels();
}

function renderCustomByokModels(preferredId) {
  ui.byokCustomModel.replaceChildren();
  for (const model of customByokModels) {
    ui.byokCustomModel.add(new Option(model.label, model.model));
  }
  ui.byokCustomModelField.hidden = !customByokModels.length;
  const selectedId = preferredId || initialSession.byokCustomModel;
  const preferred = [...ui.byokCustomModel.options].find(
    (option) => option.value === selectedId,
  );
  if (preferred) {
    ui.byokCustomModel.value = preferred.value;
  }
  if (!customByokModels.length) {
    activeByokModelSource = "builtin";
  }
  updateByokModelSelection();

  ui.byokModelList.replaceChildren();
  for (const model of customByokModels) {
    appendModelRow(
      ui.byokModelList,
      model.label,
      model.model,
      () => removeByokModel(model.model),
    );
  }
}

function updateByokModelSelection() {
  ui.byokModel.dataset.active = String(activeByokModelSource === "builtin");
  ui.byokCustomModel.dataset.active = String(activeByokModelSource === "custom");
}

function activateByokModelSource(source) {
  if (source === "custom" && !customByokModels.length) {
    return;
  }
  activeByokModelSource = source;
  updateByokModelSelection();
  saveSession({ byokModelSource: source });
}

function appendModelRow(container, label, modelId, onRemove) {
  const row = document.createElement("div");
  row.className = "model-list-row";

  const labelElement = document.createElement("span");
  labelElement.className = "model-list-label";
  labelElement.textContent = label;

  const idElement = document.createElement("span");
  idElement.className = "model-list-id";
  idElement.textContent = modelId;
  idElement.title = modelId;

  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "Remove";
  remove.addEventListener("click", onRemove);

  row.append(labelElement, idElement, remove);
  container.append(row);
}

function setModelFeedback(element, message, state = "") {
  element.textContent = message;
  element.dataset.state = state;
}

function renderCustomEnv(values) {
  ui.customEnvList.replaceChildren();
  const fixed = new Set(
    [...ui.envDialog.querySelectorAll("[data-env]")].map((input) => input.dataset.env),
  );
  for (const [name, value] of Object.entries(values)) {
    if (!fixed.has(name)) {
      addCustomEnvRow(name, value);
    }
  }
}

function addCustomEnvRow(name = "", value = "") {
  const row = document.createElement("div");
  row.className = "custom-env-row";
  row.innerHTML = `
    <input data-custom-name aria-label="Variable name" placeholder="PROVIDER_API_KEY">
    <input data-custom-value type="password" autocomplete="off" aria-label="Variable value">
    <button type="button" aria-label="Remove variable">Remove</button>
  `;
  row.querySelector("[data-custom-name]").value = name;
  row.querySelector("[data-custom-value]").value = value;
  row.querySelector("button").addEventListener("click", () => row.remove());
  ui.customEnvList.append(row);
}

function installHuggingFaceAuth() {
  const originalFetch = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const url = typeof input === "string" || input instanceof URL ? String(input) : input?.url;
    const token = envValues().HF_TOKEN;
    if (!token || !url || !new URL(url, location.href).hostname.endsWith("huggingface.co")) {
      return originalFetch(input, init);
    }
    const sourceHeaders = init.headers || (input instanceof Request ? input.headers : undefined);
    const headers = new Headers(sourceHeaders);
    headers.set("Authorization", `Bearer ${token}`);
    if (input instanceof Request) {
      return originalFetch(new Request(input, { ...init, headers }));
    }
    return originalFetch(input, { ...init, headers });
  };
}
