import { AdapterClient, fetchJSON } from "./adapter-client.js";
import {
  clearKediExecutionDiagnostic,
  createKediEditor,
  setKediExecutionDiagnostic,
  setKediTips,
} from "./kedi-editor.js";
import { MODEL_REGISTRY, modelConfig as builtinModelConfig } from "./model-registry.js";
import { PyodideRuntime } from "./pyodide-runtime.js";
import { assertWebGPU, formatBytes } from "./runtimes/runtime.js";
import { TransformersRuntime } from "./runtimes/transformers-runtime.js";
import { WllamaRuntime } from "./runtimes/wllama-runtime.js";

const EXAMPLES = Object.freeze({
  capital: `>> Give me a [country].
>> Capital of <country> is [capital].
= Capital of <country> is <capital>.`,
  contact: `>> In "Aylin Kaya is a product designer based in Berlin", the person is [name], the role is [role], and the city is [city].
= <name> works as a <role> in <city>.`,
  delivery: `@delivery_estimate(city: str) -> str:
  = Delivery to <city> takes 3 business days.

> use:
  delivery_estimate

>> Pick a European [city].
>> Use delivery_estimate. Delivery estimate for <city> is [estimate].
= <estimate>`,
  codemode: `> import: sandbox
> use: sandbox

>> Use the sandbox to run Python that finds all prime numbers under 20 and sums their squares. The numeric result is [answer: int].
= Sum of squared primes under 20 is <answer>.`,
});
const BLANK_EXAMPLE = "blank";
const DEFAULT_SOURCE = EXAMPLES.capital;
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
  modelSettingsNotice: document.querySelector("#model-settings-notice"),
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
  tipPanel: document.querySelector("#tip-panel"),
  tipList: document.querySelector("#tip-list"),
  fallbackNotice: document.querySelector("#fallback-notice"),
  executionStream: document.querySelector("#execution-stream"),
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
let activeExample = "capital";
let blankSource = "";
let fallbackNoticeTimer = null;
const pythonRuntime = new PyodideRuntime(setStatus, browserIo());

const initialSession = sessionValues();
const initialSource = initialSession.source || DEFAULT_SOURCE;
blankSource =
  typeof initialSession.blankSource === "string"
    ? initialSession.blankSource
    : exampleKeyForSource(initialSource)
      ? ""
      : initialSource;
activeExample =
  initialSession.activeExample === BLANK_EXAMPLE
    ? BLANK_EXAMPLE
    : EXAMPLES[initialSession.activeExample]
      ? initialSession.activeExample
      : exampleKeyForSource(initialSource) || "capital";
const sourceEditor = await createKediEditor(
  ui.source,
  activeExample === BLANK_EXAMPLE ? blankSource : EXAMPLES[activeExample],
  (source) => {
    handleSourceChange(source);
  },
);
updateExampleTabs();
updateTips(sourceEditor.getValue());
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
const initialPythonRuntime = initialSession.pythonRuntime === "browser" ? "browser" : "server";
document.querySelector(
  `input[name="python-runtime"][value="${initialPythonRuntime}"]`,
).checked = true;
setMode(initialMode);
if (initialPythonRuntime === "browser") {
  schedulePythonPreload();
}
loadByokModels();

function schedulePythonPreload() {
  const preload = () => {
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
  };
  if (globalThis.requestIdleCallback) {
    globalThis.requestIdleCallback(preload, { timeout: 1_500 });
    return;
  }
  globalThis.setTimeout(preload, 750);
}

function bindEvents() {
  for (const tab of document.querySelectorAll("[data-example]")) {
    tab.addEventListener("click", () => {
      selectExample(tab.dataset.example);
    });
  }
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
      updateTips(sourceEditor.getValue());
    });
  }
  for (const input of document.querySelectorAll('input[name="python-runtime"]')) {
    input.addEventListener("change", () => {
      saveSession({ pythonRuntime: input.value });
      if (input.value === "browser") {
        schedulePythonPreload();
      } else {
        ui.fallbackNotice.hidden = true;
      }
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

function selectExample(key) {
  if (key === BLANK_EXAMPLE) {
    activeExample = BLANK_EXAMPLE;
    sourceEditor.setValue(blankSource);
    saveSession({ activeExample, source: blankSource, blankSource });
    updateExampleTabs();
    updateTips(blankSource);
    sourceEditor.focus();
    return;
  }
  const source = EXAMPLES[key];
  if (!source) {
    return;
  }
  activeExample = key;
  sourceEditor.setValue(source);
  saveSession({ activeExample, source });
  updateExampleTabs();
  updateTips(source);
  sourceEditor.focus();
}

function handleSourceChange(source) {
  if (activeExample !== BLANK_EXAMPLE && EXAMPLES[activeExample] !== source) {
    activeExample = BLANK_EXAMPLE;
  }
  if (activeExample === BLANK_EXAMPLE) {
    blankSource = source;
  }
  saveSession({ source, activeExample, blankSource });
  updateExampleTabs();
  updateTips(source);
}

function exampleKeyForSource(source) {
  return Object.entries(EXAMPLES).find(([, value]) => value === source)?.[0] ?? null;
}

function updateExampleTabs() {
  for (const tab of document.querySelectorAll("[data-example]")) {
    const selected = tab.dataset.example === activeExample;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
  }
}

function selectedMode() {
  return document.querySelector('input[name="mode"]:checked')?.value ?? "local";
}

function selectedPythonRuntime() {
  return document.querySelector('input[name="python-runtime"]:checked')?.value ?? "server";
}

function setMode(mode) {
  ui.localControls.hidden = mode !== "local";
  ui.byokControls.hidden = mode !== "byok";
  ui.sourceControls.hidden = mode !== "local";
  updateModelSettingsLock(mode);
  if (mode === "local") {
    updateSourceControls();
  }
  setStatus("Ready");
  setProgress("");
  updateTips(sourceEditor.getValue());
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
  const mode = selectedMode();
  try {
    if (mode === "byok") {
      await ensureSelectedByokSecrets();
    }
  } catch (error) {
    clearResult();
    ui.error.hidden = false;
    ui.error.textContent = error?.message ?? String(error);
    addExecutionCard({
      kind: "error",
      title: "Missing environment key",
      body: error?.message ?? String(error),
      state: "error",
    });
    setStatus("Failed");
    ui.timing.textContent = `${Math.round(performance.now() - started)}ms`;
    return;
  }
  setRunning(true);
  clearResult();
  addExecutionCard({
    kind: "run",
    title: "Kedi run",
    body: mode === "local" ? "Browser-backed execution" : "BYOK execution",
    state: "running",
  });
  try {
    const result = mode === "local" ? await runLocal() : await runByok();
    ui.output.textContent = result.result;
    addExecutionCard({
      kind: "output",
      title: "Output",
      body: result.result || "(no output)",
      state: "done",
    });
    setStatus("Complete");
  } catch (error) {
    ui.error.hidden = false;
    ui.error.textContent = error?.message ?? String(error);
    addExecutionCard({
      kind: "error",
      title: "Error",
      body: error?.message ?? String(error),
      state: "error",
    });
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
  const source = sourceEditor.getValue();
  const modelId = ui.model.value;
  const baseConfig = browserModelConfig(modelId);
  const config = selectedRuntimeConfig(baseConfig);
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
  setStatus("Running Kedi");

  const client = new AdapterClient(
    null,
    setStatus,
    browserIo(),
    pythonRuntime,
    sourceDefinitelyDoesNotNeedModel(source)
      ? null
      : async (signal) => {
          await assertWebGPU();
          const runtime = await runtimeFor(config);
          setStatus("Loading model");
          await runtime.load(config, renderProgress, signal);
          if (config.source === "cache") {
            setDownloadState(true);
          }
          setProgress("Model loaded");
          return runtime;
        },
    executionEvents(),
  );
  const bridge = client.serve(runId, controller.signal);
  activeRun.bridgeStarted = true;
  try {
    const result = await fetchJSON("/api/run/local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source,
        modelId,
        modelConfig: MODEL_REGISTRY[modelId] ? null : baseConfig,
        runId,
        pythonRuntime: selectedPythonRuntime(),
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
  const model = selectedByokModel();
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
  const client = new AdapterClient(
    null,
    setStatus,
    browserIo(),
    pythonRuntime,
    null,
    executionEvents(),
  );
  const bridge = client.serve(runId, controller.signal);
  try {
    const result = await fetchJSON("/api/run/byok", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: sourceEditor.getValue(),
        model,
        runId,
        pythonRuntime: selectedPythonRuntime(),
        secrets,
        settings: {},
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

async function ensureSelectedByokSecrets() {
  const model = selectedByokModel();
  if (!model) {
    throw new Error("Enter a BYOK model");
  }
  const values = envValues();
  validateLogfireEnvironment(values);
  await ensureByokSecrets(model, values);
}

function selectedByokModel() {
  return activeByokModelSource === "custom"
    ? ui.byokCustomModel.value
    : ui.byokModel.value;
}

function sourceDefinitelyDoesNotNeedModel(source) {
  const meaningfulLines = source
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
  if (!meaningfulLines.length) {
    return true;
  }
  if (meaningfulLines.some((line) => line.startsWith(">>"))) {
    return false;
  }
  const importLines = meaningfulLines.filter((line) => line.startsWith("> import:"));
  return importLines.every((line) =>
    /^>\s*import:\s*(this|errors|require)\s*$/.test(line),
  );
}

function validateLogfireEnvironment(values) {
  if (values.LOGFIRE_ENABLED === "true" && !values.LOGFIRE_TOKEN) {
    throw new Error("LOGFIRE_TOKEN is required when Logfire is enabled");
  }
}

async function ensureByokSecrets(model, values) {
  const payload = await fetchJSON("/api/byok/models/requirements", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  const missing = (payload.requirements ?? []).filter(
    (requirement) =>
      !Array.isArray(requirement.anyOf) ||
      !requirement.anyOf.some((name) => typeof values[name] === "string" && values[name]),
  );
  if (!missing.length) {
    return;
  }
  const names = missing.map((requirement) => requirement.label || requirement.anyOf?.join(" or "));
  throw new Error(`Set ${names.join(", ")} in Environment before running ${model}.`);
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
  ui.executionStream.replaceChildren();
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
  const existing = ui.executionStream.querySelector('[data-card-id="stdio"]');
  if (existing) {
    const body = existing.querySelector(".execution-card-body");
    body.textContent += value;
    ui.executionStream.scrollTop = ui.executionStream.scrollHeight;
    return;
  }
  addExecutionCard({
    id: "stdio",
    kind: "python",
    title: "Program output",
    body: value,
    state: "done",
  });
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

function executionEvents() {
  return {
    onEvent: addExecutionCard,
    onPython: showPyodideRuntimeNotice,
  };
}

function addExecutionCard(event) {
  const id = event.id || crypto.randomUUID();
  let card = ui.executionStream.querySelector(`[data-card-id="${CSS.escape(id)}"]`);
  if (!card) {
    card = document.createElement("article");
    card.className = "execution-card";
    card.dataset.cardId = id;
    card.innerHTML = [
      '<div class="execution-card-title">',
      "  <span data-title></span>",
      '  <span class="execution-card-kind" data-kind></span>',
      "</div>",
      '<div class="execution-card-body" data-body></div>',
    ].join("");
    ui.executionStream.append(card);
  }
  card.dataset.state = event.state || "running";
  card.querySelector("[data-title]").textContent = event.title || "Step";
  card.querySelector("[data-kind]").textContent = event.kind || "";
  card.querySelector("[data-body]").textContent = event.body || "";
  ui.executionStream.scrollTop = ui.executionStream.scrollHeight;
  return id;
}

function showPyodideRuntimeNotice() {
  if (selectedPythonRuntime() !== "browser") {
    return;
  }
  ui.fallbackNotice.hidden = false;
  clearTimeout(fallbackNoticeTimer);
  fallbackNoticeTimer = setTimeout(() => {
    ui.fallbackNotice.hidden = true;
  }, 4_000);
}

function updateTips(source) {
  const tips = analyzeSourceTips(source);
  ui.tipPanel.hidden = tips.length === 0;
  ui.tipList.replaceChildren(
    ...tips.map((tip) => {
      const card = document.createElement("article");
      card.className = "tip-card";
      const title = document.createElement("strong");
      title.textContent = tip.title;
      const body = document.createElement("span");
      body.textContent = tip.message;
      card.append(title, body);
      return card;
    }),
  );
  setKediTips(sourceEditor, tips);
}

function analyzeSourceTips(source) {
  const lines = source.split(/\r?\n/);
  const tips = [];
  const localWebGpu = selectedMode() === "local";
  const useLine = lines.findIndex((line) => /^\s*>\s*use:/.test(line));
  const importsSandbox = lines.some((line) => /^\s*>\s*import:\s*sandbox\s*$/.test(line));
  const usesSandbox = lines.some((line) => /^\s*>\s*use:\s*sandbox\s*$/.test(line));
  if (localWebGpu && useLine >= 0) {
    tips.push({
      line: useLine + 1,
      column: 1,
      title: "Tool calls on WebGPU",
      message:
        "Local WebGPU tool calls can be slow and less accurate. Use BYOK for tool-heavy programs.",
    });
  }
  if (localWebGpu && (importsSandbox || usesSandbox)) {
    const importLine = lines.findIndex((item) =>
      /^\s*>\s*import:\s*sandbox\s*$/.test(item),
    );
    const useLine = lines.findIndex((item) => /^\s*>\s*use:\s*sandbox\s*$/.test(item));
    const line = importLine >= 0 ? importLine : useLine;
    tips.push({
      line: line >= 0 ? line + 1 : 1,
      column: 1,
      title: "Sandbox accuracy",
      message:
        "CodeMode and sandbox tasks are more reliable with BYOK models than small local WebGPU models.",
    });
  }
  const templateLines = lines
    .map((line, index) => ({ line, index }))
    .filter(({ line }) => /^\s*>>/.test(line));
  if (templateLines.length > 1) {
    tips.push({
      line: templateLines[1].index + 1,
      column: 1,
      title: "Batch independent prompts",
      message:
        "If these generations are independent, combining them into one template block reduces model calls.",
    });
  }
  return tips;
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

function updateModelSettingsLock(mode = selectedMode()) {
  const locked = mode === "byok";
  ui.modelSettingsPanel.dataset.locked = String(locked);
  ui.modelSettingsNotice.hidden = !locked;
  for (const [, , input] of modelSettingFields()) {
    input.disabled = locked;
  }
  ui.resetModelSettings.disabled = locked;
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
