import { PyodideRuntime } from "./pyodide-runtime.js";

export class AdapterClient {
  constructor(runtime, onStatus) {
    this.runtime = runtime;
    this.onStatus = onStatus;
    this.python = new PyodideRuntime(onStatus);
  }

  async serve(runId, signal) {
    try {
      while (!signal.aborted) {
        let payload;
        try {
          payload = await fetchJSON(
            `/api/bridge/request?runId=${encodeURIComponent(runId)}`,
            { signal },
          );
        } catch (error) {
          if (signal.aborted) {
            return;
          }
          throw error;
        }
        if (payload.cancelled || payload.done) {
          return;
        }
        if (payload.request) {
          await this.handle(runId, payload.request);
        }
      }
    } finally {
      this.python.dispose();
    }
  }

  async handle(runId, request) {
    if (request.operation === "python") {
      this.onStatus("Running Python sandbox");
      try {
        await postResponse(runId, request.id, await this.python.execute(request));
      } catch (error) {
        await postResponse(runId, request.id, {
          ok: false,
          errorType: error?.name || "PyodideError",
          error: error?.message || String(error),
        });
      }
      return;
    }
    if (!this.runtime) {
      await postResponse(runId, request.id, {
        kind: "error",
        error: "This run has no browser model runtime",
      });
      return;
    }
    this.onStatus(`Generating ${fieldNames(request.outputSchema)}`);
    try {
      const text = await this.runtime.generate({
        messages: browserMessages(request),
        settings: request.settings,
      });
      const response = parseModelResponse(text, request.outputSchema);
      await postResponse(runId, request.id, response);
    } catch (error) {
      await postResponse(runId, request.id, {
        kind: "error",
        error: error?.message ?? String(error),
      });
    }
  }
}

function browserMessages(request) {
  const outputKeys = Object.keys(request.outputSchema.properties ?? {});
  const base = [
    request.instructions || "Complete the Kedi template accurately.",
    "Return only one JSON object with no markdown or explanation.",
  ];
  let contract;
  if (request.tools.length) {
    contract = [
      'For a final answer return {"type":"final","data":{...}}.',
      `The data object must match this JSON Schema: ${JSON.stringify(request.outputSchema)}`,
      'To call one tool return {"type":"tool_call","name":"tool_name","arguments":{...}}.',
      `Available tools: ${JSON.stringify(request.tools)}`,
    ];
  } else {
    const hints = Object.fromEntries(
      Object.entries(request.outputSchema.properties ?? {}).map(([name, schema]) => [
        name,
        schemaType(schema),
      ]),
    );
    contract = [
      `Return exactly these keys: ${outputKeys.join(", ")}.`,
      `Output type hints: ${JSON.stringify(hints)}.`,
    ];
  }
  const system = [...base, ...contract, request.effort ? `Reasoning effort: ${request.effort}.` : ""]
    .filter(Boolean)
    .join("\n");
  return [{ role: "system", content: system }, ...request.messages];
}

function schemaType(schema) {
  if (schema?.$ref) {
    return schema.$ref.split("/").at(-1);
  }
  if (Array.isArray(schema?.enum)) {
    return schema.enum.join(" | ");
  }
  if (schema?.type === "array") {
    return `list[${schemaType(schema.items)}]`;
  }
  if (Array.isArray(schema?.anyOf)) {
    return schema.anyOf.map(schemaType).join(" | ");
  }
  return schema?.type || "value";
}

function parseModelResponse(text, schema) {
  const parsed = parseJsonObject(text);
  if (parsed) {
    if (parsed.type === "tool_call") {
      return {
        kind: "tool_call",
        name: parsed.name,
        arguments: parsed.arguments ?? {},
        callId: parsed.callId,
      };
    }
    if (parsed.type === "final") {
      return { kind: "final", data: parsed.data };
    }
    return { kind: "final", data: parsed };
  }
  const properties = Object.keys(schema.properties ?? {});
  const only = properties[0];
  if (properties.length === 1 && schema.properties[only]?.type === "string") {
    return { kind: "final", data: { [only]: text.trim() } };
  }
  throw new Error(`Model did not return JSON: ${text.slice(0, 300)}`);
}

function parseJsonObject(text) {
  const start = text.indexOf("{");
  if (start < 0) {
    return null;
  }
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (escaped) {
      escaped = false;
    } else if (char === "\\") {
      escaped = true;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (!quoted && char === "{") {
      depth += 1;
    } else if (!quoted && char === "}") {
      depth -= 1;
      if (depth === 0) {
        try {
          return JSON.parse(text.slice(start, index + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function fieldNames(schema) {
  const names = Object.keys(schema.properties ?? {});
  return names.length ? names.join(", ") : "model output";
}

async function postResponse(runId, requestId, response) {
  await fetchJSON("/api/bridge/response", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ runId, requestId, response }),
  });
}

export async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new Error(`Expected JSON from ${url}, received ${text.slice(0, 180)}`);
  }
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status} from ${url}`);
  }
  return payload;
}
