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

export const DEFAULT_CONTEXT_SOFT_LIMIT_PERCENT = 80;

export function estimateMessagesTokens(messages) {
  const bytes = new TextEncoder().encode(completionPrompt(messages)).length;
  return Math.max(1, Math.ceil(bytes / 3) + messages.length * 8);
}

export function nextContextSize(
  currentSize,
  requiredTokens,
  softLimitPercent = DEFAULT_CONTEXT_SOFT_LIMIT_PERCENT,
  maximumSize = Number.POSITIVE_INFINITY,
) {
  const current = positiveInteger(currentSize, 512);
  const required = positiveInteger(requiredTokens, 1);
  const maximum = positiveInteger(maximumSize, Number.POSITIVE_INFINITY);
  if (required <= softTokenLimit(current, softLimitPercent) || current >= maximum) {
    return current;
  }

  const desired = Math.ceil(
    required / normalizedSoftLimitRatio(softLimitPercent),
  );
  let target = current;
  while (target < desired && target < maximum) {
    target *= 2;
  }
  return Math.min(target, maximum);
}

export function softMaxNewTokens(
  inputTokens,
  requestedTokens,
  contextSize,
  softLimitPercent = DEFAULT_CONTEXT_SOFT_LIMIT_PERCENT,
) {
  const requested = positiveInteger(requestedTokens, 1);
  if (!Number.isFinite(contextSize) || contextSize <= 0) {
    return requested;
  }

  const input = Math.max(0, Math.floor(Number(inputTokens) || 0));
  const hardAvailable = Math.floor(contextSize) - input;
  if (hardAvailable <= 0) {
    throw new Error(
      `WebGPU request uses ${input} tokens, but the model context holds ${Math.floor(contextSize)} tokens`,
    );
  }

  const softAvailable = softTokenLimit(contextSize, softLimitPercent) - input;
  return Math.min(requested, Math.max(1, softAvailable), hardAvailable);
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

function softTokenLimit(contextSize, softLimitPercent) {
  return Math.floor(contextSize * normalizedSoftLimitRatio(softLimitPercent));
}

function normalizedSoftLimitRatio(value) {
  const percent = Number(value);
  if (!Number.isFinite(percent) || percent <= 0 || percent >= 100) {
    return DEFAULT_CONTEXT_SOFT_LIMIT_PERCENT / 100;
  }
  return percent / 100;
}

function positiveInteger(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : fallback;
}
