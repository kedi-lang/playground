import { PyodideRuntime } from "./pyodide-runtime.js";

export class AdapterClient {
  constructor(runtime, onStatus, io = {}, python = null, loadRuntime = null) {
    this.runtime = runtime;
    this.onStatus = onStatus;
    this.python = python ?? new PyodideRuntime(onStatus, io);
    this.loadRuntime = loadRuntime;
  }

  async serve(runId, signal) {
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
        await this.handle(runId, payload.request, signal);
      }
    }
  }

  async handle(runId, request, signal) {
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
    const runtime = await this.modelRuntime(signal);
    if (!runtime) {
      await postResponse(runId, request.id, {
        kind: "error",
        error: "This run has no browser model runtime",
      });
      return;
    }
    this.onStatus(`Generating ${fieldNames(request.outputSchema)}`);
    const messages = browserMessages(request);
    try {
      const generation = normalizeGeneration(
        await runtime.generate({
          messages,
          settings: request.settings,
          responseFormat:
            request.outputSchema.type === "object" || request.tools.length
              ? "json_object"
              : "text",
        }),
      );
      const response = parseModelResponse(
        generation.text,
        request.outputSchema,
        request.tools,
        request.messages.some((message) => message.role === "tool"),
      );
      response.telemetry = {
        inputMessages: messages,
        outputText: generation.text,
        model: generation.model || request.model,
        finishReason:
          response.kind === "tool_call"
            ? "tool_call"
            : generation.finishReason || "stop",
        responseId: generation.responseId || null,
        usage: generation.usage || null,
      };
      await postResponse(runId, request.id, response);
    } catch (error) {
      await postResponse(runId, request.id, {
        kind: "error",
        error: error?.message ?? String(error),
        telemetry: {
          inputMessages: messages,
          model: request.model,
          finishReason: "error",
        },
      });
    }
  }

  async modelRuntime(signal) {
    if (this.runtime || !this.loadRuntime) {
      return this.runtime;
    }
    this.runtime = await this.loadRuntime(signal);
    return this.runtime;
  }
}

function normalizeGeneration(value) {
  if (typeof value === "string") {
    return { text: value };
  }
  if (value && typeof value.text === "string") {
    return value;
  }
  throw new Error("Browser runtime returned an invalid generation result");
}

function browserMessages(request) {
  const objectOutput = request.outputSchema.type === "object";
  const outputKeys = Object.keys(request.outputSchema.properties ?? {});
  const requiredTools = request.requiredTools ?? [];
  const outputContract =
    request.outputPrompt || formatOutputContract(request.outputSchema);
  const toolCatalog = formatToolCatalog(request.tools);
  const messages = canonicalTemplateMessages(request.messages, outputKeys);
  const base = [request.instructions];
  let contract;
  if (request.tools.length) {
    const toolResults = request.messages
      .filter((message) => message.role === "tool")
      .map((message) => parseToolResult(message.content));
    const hasToolResult = toolResults.some((result) => result?.ok === true);
    const hasToolFailure = toolResults.some((result) => result?.ok === false);
    contract = [
      "You are in an agent loop. Return exactly one JSON object for this turn.",
      ...(requiredTools.length
        ? [
            `You must call each explicitly requested tool before returning a final answer: ${requiredTools.join(", ")}.`,
            "Return only a call_tool action now.",
          ]
        : []),
      ...(hasToolResult
        ? [
            "A tool result is present in the conversation. Use its actual result.",
            "Return a final action unless another tool call is genuinely required.",
          ]
        : []),
      ...(hasToolFailure
        ? [
            "A previous tool call failed. Correct its arguments using the XML tool catalog and call it again.",
            "A failed tool call does not provide a value and must never be used in a final answer.",
          ]
        : []),
      'To ask the host to execute a tool, return {"action":"call_tool","name":"tool_name","arguments":{...}}.',
      "After a call_tool action, stop immediately. The host will execute it and send back a tool result.",
      "Never simulate, predict, or invent a tool result.",
      "Available tools are defined by this XML catalog:",
      toolCatalog,
      "Tool arguments must contain concrete runtime values taken from the conversation.",
      "Use the exact argument names and value types shown in the XML tool catalog.",
      'For a string argument named country, use {"country":"Turkey"}.',
      objectOutput
        ? 'When the answer is ready, return {"action":"final","data":{...}}.'
        : 'When the answer is ready, return {"action":"final","data":"..."}.',
      "Never return call_tool and final in the same turn.",
      "The final data must match this XML output contract:",
      outputContract,
      "The XML contract describes the answer shape only. Never copy its tags or field metadata into final data.",
      'Type labels and placeholders such as "string", "integer", "value", and "example" are never valid field values.',
      "Generate concrete semantic values that complete the template.",
    ];
  } else if (!objectOutput) {
    contract = [
      "Return only that placeholder's value with no JSON, label, markdown, or explanation.",
    ];
  } else {
    contract = [
      "Return only one JSON object with no markdown or explanation.",
      `Return exactly these keys: ${outputKeys.join(", ")}.`,
      "Match this XML output contract:",
      outputContract,
      "Return values, not the XML field metadata.",
      'Type labels and placeholders such as "string", "integer", "value", and "example" are never valid field values.',
      "Generate concrete semantic values that complete the template.",
    ];
  }
  const system = [...base, ...contract, request.effort ? `Reasoning effort: ${request.effort}.` : ""]
    .filter(Boolean)
    .join("\n");
  return [{ role: "system", content: system }, ...messages];
}

function canonicalTemplateMessages(messages, outputKeys) {
  if (!outputKeys.length) {
    return messages;
  }
  const userIndex = messages.findIndex((message) => message.role === "user");
  if (userIndex < 0) {
    return messages;
  }
  const content = messages[userIndex].content;
  if (
    typeof content !== "string" ||
    content.includes("Fields to be substituted into the template string for final construction:")
  ) {
    return messages;
  }
  const instruction =
    "The template string must maintain grammatical and semantic integrity once the output fields are substituted into" +
    " their corresponding locations in the template after the final construction.\n";
  const fields =
    "Fields to be substituted into the template string for final construction: " +
    `(${outputKeys.join(", ")}).`;
  const normalized = `${instruction}${fields}\n\nThe template string:\n${content}`;
  return messages.map((message, index) =>
    index === userIndex ? { ...message, content: normalized } : message,
  );
}

function parseToolResult(content) {
  try {
    return JSON.parse(content);
  } catch {
    return null;
  }
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

function parseModelResponse(text, schema, tools = [], hasToolResult = false) {
  const parsedObjects = parseJsonObjects(text);
  const toolCall = parsedObjects.find(
    (value) => value.type === "tool_call" || value.action === "call_tool",
  );
  const final = parsedObjects.find(
    (value) => value.type === "final" || value.action === "final",
  );
  if (hasToolResult && final) {
    return { kind: "final", data: final.data };
  }
  if (toolCall) {
    const tool = tools.find((candidate) => candidate.name === toolCall.name);
    if (!tool) {
      return {
        kind: "retry",
        error: `Unknown tool ${JSON.stringify(toolCall.name)}. Use one exact tool name from the XML catalog.`,
      };
    }
    const argumentError = validateToolArguments(tool, toolCall.arguments);
    if (argumentError) {
      return {
        kind: "retry",
        requiredTool: tool.name,
        error: [
          `Invalid arguments for ${tool.name}: ${argumentError}.`,
          `Expected ${toolSignature(tool)}.`,
          "Return a corrected call_tool action using concrete values only.",
        ].join(" "),
      };
    }
    return {
      kind: "tool_call",
      name: toolCall.name,
      arguments: toolCall.arguments ?? {},
      callId: toolCall.callId,
    };
  }
  if (final) {
    return { kind: "final", data: final.data };
  }
  const parsed = parsedObjects[0];
  if (parsed) {
    if (schema.type !== "object") {
      return { kind: "final", data: text.trim() };
    }
    return { kind: "final", data: parsed };
  }
  if (tools.length) {
    const mentionedTool = tools.find((tool) =>
      new RegExp(`(^|\\W)${escapeRegExp(tool.name)}($|\\W)`).test(text),
    );
    return {
      kind: "retry",
      error: mentionedTool
        ? `You said ${mentionedTool.name} should be used but did not call it. Return only {"action":"call_tool","name":"${mentionedTool.name}","arguments":{...}} now.`
        : 'Return exactly one valid {"action":"call_tool",...} or {"action":"final","data":...} JSON object.',
      requiredTool: mentionedTool?.name,
    };
  }
  if (schema.type === "string") {
    return { kind: "final", data: text.trim() };
  }
  const properties = Object.keys(schema.properties ?? {});
  const only = properties[0];
  if (properties.length === 1 && schema.properties[only]?.type === "string") {
    return { kind: "final", data: { [only]: text.trim() } };
  }
  throw new Error(`Model did not return JSON: ${text.slice(0, 300)}`);
}

function formatOutputContract(schema) {
  const properties = schema?.properties;
  if (!properties || typeof properties !== "object") {
    return `<Output>\n  <type>${escapeXml(schemaType(schema))}</type>\n</Output>`;
  }
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  const fields = formatXmlFields(
    Object.entries(properties).filter(([name]) => required.has(name)),
  );
  const optionalFields = formatXmlFields(
    Object.entries(properties).filter(([name]) => !required.has(name)),
  );
  const parts = [
    `<fields>${fields ? `\n${indentXml(fields, 2)}\n` : ""}</fields>`,
    optionalFields
      ? `<optional_fields>\n${indentXml(optionalFields, 2)}\n</optional_fields>`
      : "",
  ].filter(Boolean);
  return `<Output>\n${indentXml(parts.join("\n"), 2)}\n</Output>`;
}

function formatToolCatalog(tools) {
  if (!tools.length) {
    return "<Tools />";
  }
  const rendered = tools
    .map((tool) => {
      const schema = tool.inputSchema ?? {};
      const properties = schema.properties ?? {};
      const required = new Set(Array.isArray(schema.required) ? schema.required : []);
      const argumentsXml = formatXmlFields(
        Object.entries(properties).filter(([name]) => required.has(name)),
      );
      const optionalArgumentsXml = formatXmlFields(
        Object.entries(properties).filter(([name]) => !required.has(name)),
      );
      const parts = [
        `<name>${escapeXml(tool.name)}</name>`,
        tool.description
          ? `<description>${escapeXml(tool.description)}</description>`
          : "",
        `<arguments>${argumentsXml ? `\n${indentXml(argumentsXml, 2)}\n` : ""}</arguments>`,
        optionalArgumentsXml
          ? `<optional_arguments>\n${indentXml(optionalArgumentsXml, 2)}\n</optional_arguments>`
          : "",
        tool.returnSchema
          ? `<returns>${escapeXml(schemaType(tool.returnSchema))}</returns>`
          : "",
      ].filter(Boolean);
      return `<Tool>\n${indentXml(parts.join("\n"), 2)}\n</Tool>`;
    })
    .join("\n");
  return `<Tools>\n${indentXml(rendered, 2)}\n</Tools>`;
}

function formatXmlFields(entries) {
  return entries
    .map(([name, schema]) => {
      const description =
        typeof schema?.description === "string" && schema.description
          ? ` description="${escapeXml(schema.description)}"`
          : "";
      return `<field name="${escapeXml(name)}" type="${escapeXml(schemaType(schema))}"${description} />`;
    })
    .join("\n");
}

function indentXml(value, spaces) {
  const prefix = " ".repeat(spaces);
  return value
    .split("\n")
    .map((line) => `${prefix}${line}`)
    .join("\n");
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function toolSignature(tool) {
  const schema = tool.inputSchema ?? {};
  const properties = schema.properties ?? {};
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  const parameters = Object.entries(properties).map(([name, field]) => {
    const suffix = required.has(name) ? "" : "?";
    return `${name}${suffix}: ${schemaType(field)}`;
  });
  return `${tool.name}(${parameters.join(", ")})`;
}

function validateToolArguments(tool, argumentsValue) {
  if (
    !argumentsValue ||
    typeof argumentsValue !== "object" ||
    Array.isArray(argumentsValue)
  ) {
    return "arguments must be an object";
  }
  const schema = tool.inputSchema ?? {};
  const properties = schema.properties ?? {};
  const required = Array.isArray(schema.required) ? schema.required : [];
  const missing = required.filter(
    (name) => !Object.prototype.hasOwnProperty.call(argumentsValue, name),
  );
  const unexpected = Object.keys(argumentsValue).filter(
    (name) => !Object.prototype.hasOwnProperty.call(properties, name),
  );
  const invalid = Object.entries(argumentsValue)
    .filter(
      ([name, value]) =>
        Object.prototype.hasOwnProperty.call(properties, name) &&
        !matchesSchemaType(value, properties[name]),
    )
    .map(([name]) => `${name} must be ${schemaType(properties[name])}`);
  const errors = [];
  if (missing.length) {
    errors.push(`missing ${missing.join(", ")}`);
  }
  if (unexpected.length) {
    errors.push(`unexpected ${unexpected.join(", ")}`);
  }
  errors.push(...invalid);
  return errors.join("; ");
}

function matchesSchemaType(value, schema) {
  if (Array.isArray(schema?.enum)) {
    return schema.enum.includes(value);
  }
  if (Array.isArray(schema?.anyOf)) {
    return schema.anyOf.some((option) => matchesSchemaType(value, option));
  }
  if (Array.isArray(schema?.oneOf)) {
    return schema.oneOf.some((option) => matchesSchemaType(value, option));
  }
  if (schema?.$ref) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }
  switch (schema?.type) {
    case "null":
      return value === null;
    case "string":
      return typeof value === "string";
    case "boolean":
      return typeof value === "boolean";
    case "integer":
      return Number.isInteger(value);
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "array":
      return (
        Array.isArray(value) &&
        value.every((item) => matchesSchemaType(item, schema.items ?? {}))
      );
    case "object":
      return value !== null && typeof value === "object" && !Array.isArray(value);
    default:
      return true;
  }
}
function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function parseJsonObjects(text) {
  const values = [];
  let depth = 0;
  let quoted = false;
  let escaped = false;
  let start = -1;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (escaped) {
      escaped = false;
    } else if (char === "\\") {
      escaped = true;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (!quoted && char === "{") {
      if (depth === 0) {
        start = index;
      }
      depth += 1;
    } else if (!quoted && char === "}") {
      depth -= 1;
      if (depth === 0 && start >= 0) {
        try {
          values.push(JSON.parse(text.slice(start, index + 1)));
        } catch {
          // Keep scanning: a later object may still contain a valid action.
        }
        start = -1;
      }
    }
  }
  return values;
}

function fieldNames(schema) {
  if (schema.type !== "object") {
    return schema.type || "model output";
  }
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
    const error = new Error(payload.error || `HTTP ${response.status} from ${url}`);
    error.diagnostic = payload.diagnostic ?? null;
    throw error;
  }
  return payload;
}
