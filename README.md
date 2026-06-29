# Kedi Playground PoC

This proof of concept runs the same Kedi source through:

- Bonsai 1.7B Q1 in the browser with wllama/WebGPU;
- Ternary Bonsai 1.7B q2 in the browser with Transformers.js/WebGPU;
- Pydantic AI on the backend with a browser-supplied provider key.

Runtime selection is owned by the model registry. The UI exposes model names,
not browser inference engines.

Run from the repository root:

```bash
uv run --extra playground python -m playground.server
```

Then open <http://127.0.0.1:8787>.

Environment values and `HF_TOKEN` are stored in browser local storage. BYOK
runs pass the saved non-Hugging Face values to a short-lived worker process,
without mutating the server environment. The BYOK model list is sourced from
Pydantic AI's `KnownModelName`. The Models tab accepts labelled custom
`provider:model` entries, checks their provider through Pydantic AI, and
normalizes the selected ID before adapter construction.

The selected execution mode, models, custom model registries, and Kedi source
are retained for the browser session. Custom browser models use a Hugging Face
repository and model filename: `.gguf` selects wllama, while `.onnx` selects
Transformers.js and exposes a dtype choice. The Model settings tab exposes the
shared Pydantic AI and WebGPU controls: temperature, maximum output tokens,
top-p, and an optional seed. These values are adapter defaults for the run;
Kedi `> settings:` directives still override them inside their active scope.
Logfire is configured and Pydantic AI is instrumented only when
`LOGFIRE_ENABLED` is set and a `LOGFIRE_TOKEN` is present.

Local model storage is browser-owned:

- Bonsai GGUF uses wllama's OPFS-backed cache. A cache miss downloads from
  Hugging Face directly into the requesting browser.
- Ternary Bonsai uses the Transformers.js browser cache with the same
  cache-first behavior.
- A user-selected GGUF file can be loaded without copying it to the server.

The playground server has no model download or model-file endpoint. Downloads
report live progress and can be cancelled in the browser. User-authored Python
and type expressions execute in an isolated Pyodide Web Worker; they never run
in the server's Python process. Browser adapter MCP profiles accept remote HTTP
and SSE servers only; `stdio` is rejected. Kedi procedure and Python tools are
registered through the same `WebGPUAdapter` tool protocol and executed by the
server-side Kedi runtime after the browser model requests them.
