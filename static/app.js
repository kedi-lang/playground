import { AdapterClient, fetchJSON } from "./adapter-client.js";
import { MODEL_REGISTRY, modelConfig } from "./model-registry.js";
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
  customModelField: document.querySelector("#custom-model-field"),
  customModel: document.querySelector("#custom-model"),
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
  customEnvList: document.querySelector("#custom-env-list"),
  fileSourceOption: document.querySelector('[data-source-option="file"]'),
  runSettingsPanel: document.querySelector("#run-settings-panel"),
  modelSettingsPanel: document.querySelector("#model-settings-panel"),
  settingTemperature: document.querySelector("#setting-temperature"),
  settingMaxTokens: document.querySelector("#setting-max-tokens"),
  settingTopP: document.querySelector("#setting-top-p"),
  settingSeed: document.querySelector("#setting-seed"),
  resetModelSettings: document.querySelector("#reset-model-settings"),
};

let activeRuntime = null;
let activeModelId = null;
let activeRun = null;
let activeDownload = null;

const initialSession = sessionValues();
ui.source.value = initialSession.source || DEFAULT_SOURCE;
for (const model of Object.values(MODEL_REGISTRY)) {
  ui.model.add(new Option(model.label, model.id));
}
if (MODEL_REGISTRY[initialSession.browserModel]) {
  ui.model.value = initialSession.browserModel;
}
const initialModelSource = initialSession.modelSource === "file" ? "file" : "cache";
const initialSource = document.querySelector(
  `input[name="model-source"][value="${initialModelSource}"]`,
);
if (initialSource) {
  initialSource.checked = true;
}
ui.customModel.value = initialSession.customModel || "";
applyModelSettings(initialSession.modelSettings);
setControlTab(initialSession.controlTab === "model" ? "model" : "run");
installHuggingFaceAuth();
bindEvents();
const initialMode = initialSession.mode === "byok" ? "byok" : "local";
document.querySelector(`input[name="mode"][value="${initialMode}"]`).checked = true;
setMode(initialMode);
setStatus("Ready");
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
  ui.model.addEventListener("change", async () => {
    if (activeRuntime && activeModelId !== ui.model.value) {
      await activeRuntime.unload();
      activeRuntime = null;
      activeModelId = null;
    }
    setStatus("Ready");
    setProgress("");
    updateSourceControls();
    saveSession({ browserModel: ui.model.value });
  });
  for (const input of document.querySelectorAll('input[name="model-source"]')) {
    input.addEventListener("change", () => {
      setStatus("Ready");
      setProgress("");
      updateSourceControls();
      saveSession({ modelSource: input.value });
    });
  }
  ui.modelFile.addEventListener("change", () => {
    const file = ui.modelFile.files?.[0];
    ui.modelFileName.textContent = file?.name || "";
    if (file) {
      document.querySelector('input[name="model-source"][value="file"]').checked = true;
      saveSession({ modelSource: "file" });
      updateSourceControls();
    }
  });
  ui.downloadLocal.addEventListener("click", cacheModel);
  ui.run.addEventListener("click", run);
  ui.cancel.addEventListener("click", cancel);
  document.querySelector("#open-env").addEventListener("click", openEnv);
  document.querySelector("#close-env").addEventListener("click", () => ui.envDialog.close());
  document.querySelector("#save-env").addEventListener("click", saveEnv);
  document.querySelector("#clear-env").addEventListener("click", clearEnv);
  document.querySelector("#add-env").addEventListener("click", () => addCustomEnvRow());
  ui.byokModel.addEventListener("change", () => {
    updateCustomModelField();
    saveSession({ byokModel: ui.byokModel.value });
  });
  ui.customModel.addEventListener("input", () => {
    saveSession({ customModel: ui.customModel.value });
  });
  ui.source.addEventListener("input", () => {
    saveSession({ source: ui.source.value });
  });
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
  const modelTab = tabName === "model";
  ui.runSettingsPanel.hidden = modelTab;
  ui.modelSettingsPanel.hidden = !modelTab;
  for (const tab of document.querySelectorAll("[data-control-tab]")) {
    const active = tab.dataset.controlTab === (modelTab ? "model" : "run");
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
    setStatus("Failed");
  } finally {
    ui.timing.textContent = `${Math.round(performance.now() - started)}ms`;
    setRunning(false);
    activeRun = null;
  }
}

async function runLocal() {
  await assertWebGPU();
  const modelId = ui.model.value;
  const config = selectedRuntimeConfig(modelConfig(modelId));
  const runtime = await runtimeFor(config);
  const runId = crypto.randomUUID();
  const controller = new AbortController();
  activeRun = { runId, controller, bridgeStarted: false };
  setStatus("Loading model");
  await runtime.load(config, renderProgress, controller.signal);
  setProgress("Model loaded");
  setStatus("Running Kedi");

  const client = new AdapterClient(runtime, setStatus);
  const bridge = client.serve(runId, controller.signal);
  activeRun.bridgeStarted = true;
  try {
    const result = await fetchJSON("/api/run/local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: ui.source.value,
        modelId,
        runId,
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
    ui.byokModel.value === "__custom__"
      ? ui.customModel.value.trim()
      : ui.byokModel.value;
  if (!model) {
    throw new Error("Enter a BYOK model");
  }
  const values = envValues();
  if (values.LOGFIRE_ENABLED === "true" && !values.LOGFIRE_TOKEN) {
    throw new Error("LOGFIRE_TOKEN is required when Logfire is enabled");
  }
  const secrets = Object.fromEntries(
    Object.entries(values).filter(
      ([name, value]) => name !== "HF_TOKEN" && typeof value === "string" && value,
    ),
  );
  setStatus("Running Kedi");
  const runId = crypto.randomUUID();
  const controller = new AbortController();
  activeRun = { runId, controller, bridgeStarted: true };
  const client = new AdapterClient(null, setStatus);
  const bridge = client.serve(runId, controller.signal);
  try {
    const result = await fetchJSON("/api/run/byok", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: ui.source.value,
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

async function runtimeFor(config) {
  if (activeRuntime && activeModelId === config.id) {
    return activeRuntime;
  }
  await activeRuntime?.unload();
  activeRuntime = config.engine === "wllama" ? new WllamaRuntime() : new TransformersRuntime();
  activeModelId = config.id;
  return activeRuntime;
}

function selectedRuntimeConfig(config) {
  const source = selectedModelSource();
  if (config.engine === "transformers") {
    if (source === "file") {
      throw new Error("File source is only available for GGUF models");
    }
    return {
      ...config,
      source: "cache",
    };
  }
  if (source === "file") {
    const fileObject = ui.modelFile.files?.[0];
    if (!fileObject) {
      throw new Error("Choose a GGUF file");
    }
    return { ...config, source, fileObject };
  }
  return { ...config, source: "cache" };
}

function selectedModelSource() {
  return document.querySelector('input[name="model-source"]:checked')?.value ?? "cache";
}

function updateSourceControls() {
  const config = modelConfig(ui.model.value);
  const wllama = config.engine === "wllama";
  const source = selectedModelSource();
  ui.sourceControls.hidden = selectedMode() !== "local";
  ui.fileSourceOption.hidden = !wllama;
  ui.modelFile.disabled = !wllama;
  ui.filePicker.hidden = !wllama || source !== "file";
  ui.downloadLocal.hidden = source === "file";
  ui.downloadLocal.textContent = "Cache model";
  if (!wllama && selectedModelSource() === "file") {
    document.querySelector('input[name="model-source"][value="cache"]').checked = true;
    ui.filePicker.hidden = true;
    ui.downloadLocal.hidden = false;
    saveSession({ modelSource: "cache" });
  }
}

async function cacheModel() {
  const config = modelConfig(ui.model.value);
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
    ui.downloadLocal.disabled = false;
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
  ui.output.textContent = "";
  ui.error.textContent = "";
  ui.error.hidden = true;
  ui.timing.textContent = "";
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
  localStorage.setItem(ENV_KEY, JSON.stringify(values));
  ui.envDialog.close();
  setStatus("Environment saved");
}

function clearEnv() {
  localStorage.removeItem(ENV_KEY);
  for (const input of ui.envDialog.querySelectorAll("[data-env]")) {
    input.value = "";
    input.checked = false;
  }
  ui.customEnvList.replaceChildren();
  setStatus("Environment cleared");
}

async function loadByokModels() {
  try {
    const payload = await fetchJSON("/api/byok/models");
    for (const model of payload.models) {
      ui.byokModel.add(new Option(model.label, model.id));
    }
    ui.byokModel.add(new Option("Custom model...", "__custom__"));
    const preferredId = initialSession.byokModel || "openai:gpt-4o-mini";
    const preferred = [...ui.byokModel.options].find(
      (option) => option.value === preferredId,
    );
    if (preferred) {
      ui.byokModel.value = preferred.value;
    }
    updateCustomModelField();
  } catch (error) {
    setStatus("Model list failed");
    setProgress(error?.message ?? String(error));
  }
}

function updateCustomModelField() {
  ui.customModelField.hidden = ui.byokModel.value !== "__custom__";
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
