export class BrowserModelRuntime {
  async load(_config, _onProgress) {
    throw new Error("BrowserModelRuntime.load must be implemented");
  }

  async generate(_request) {
    throw new Error("BrowserModelRuntime.generate must be implemented");
  }

  async isCached(_config) {
    return false;
  }

  async unload() {}
}

export function formatBytes(value) {
  if (!Number.isFinite(value)) {
    return "?";
  }
  if (value < 1024) {
    return `${value}B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)}KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)}MB`;
}

export function completionPrompt(messages) {
  return `${messages
    .map((message) => {
      const name = message.name ? ` (${message.name})` : "";
      return `${message.role.toUpperCase()}${name}:\n${message.content}`;
    })
    .join("\n\n")}\n\nASSISTANT:\n`;
}

export async function assertWebGPU() {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("The browser could not create a WebGPU adapter");
  }
}
