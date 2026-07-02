# Kedi Playground

This proof of concept runs the same Kedi source through:

- Bonsai 1.7B Q1 in the browser with wllama/WebGPU;
- Ternary Bonsai 1.7B q2 in the browser with Transformers.js/WebGPU;
- Pydantic AI on the backend with a browser-supplied provider key.

Runtime selection is owned by the model registry. The UI exposes model names,
not browser inference engines.

## Hugging Face Space

This repository is ready for a Docker Space. The container listens on port
`7860` and runs as user `1000`. The `KEDI_INSTALL_MODE` Space build variable
selects the Kedi source:

- `prod` installs the latest `kedi[playground]` release from PyPI.
- `dev` is the current default and installs `KEDI_REVISION` from
  `KEDI_REPOSITORY`. The defaults select the private `kedi_playground` branch,
  so dev builds require a read-only `GITHUB_TOKEN` Space secret.

Use `dev` until the PyPI release contains the playground adapter APIs. The
Docker build verifies those imports and rejects an incompatible production
release instead of creating a broken image.

The GitHub token is mounted as a BuildKit secret only for the dependency-install
step and is not stored in an image layer.

The model files are intentionally not copied into the image. WebGPU inference
downloads them directly into each visitor's browser cache.

Build and run locally:

```bash
# Development: Kedi's kedi_playground branch (default)
GH_TOKEN="$(gh auth token)" docker build \
  --secret id=GITHUB_TOKEN,env=GH_TOKEN \
  -t kedi-playground:dev .

# Production: latest Kedi release from PyPI
docker build \
  --build-arg KEDI_INSTALL_MODE=prod \
  -t kedi-playground .

docker run --rm -p 7860:7860 kedi-playground
```

Then open <http://127.0.0.1:7860>.

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
Both WebGPU and BYOK runs execute in a short-lived worker. BYOK uses
`PydanticAdapter`; browser inference uses `WebGPUAdapter` directly. Logfire is
configured inside the worker only when `LOGFIRE_ENABLED` is set and a
`LOGFIRE_TOKEN` is present, so per-user tokens never mutate the long-lived
FastAPI process. Pydantic AI instrumentation covers BYOK runs, while
WebGPU runs emit the same GenAI-shaped trace hierarchy: an agent run span,
`chat <model>` spans with messages, settings, tool definitions, finish reasons,
and token usage, plus child spans for actual tool executions.

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
registered by `WebGPUAdapter`, which owns tool execution, remote MCP discovery,
profile overrides, output validation, and the browser model loop. The browser
selects wllama or Transformers.js from the model registry. BYOK remains a
separate `PydanticAdapter` execution path.

WebGPU context management keeps a percentage-based soft limit instead of a
fixed prompt ceiling. wllama starts with a small context and expands it
automatically, without redownloading cached weights, when the estimated request
approaches 80% of the active window. If the estimate is low, wllama's exact
overflow count triggers one corrected retry. Transformers.js uses its tokenizer
count to reserve output space against the same soft threshold. BYOK requests are
not altered by this browser-only policy.

The Monaco editor parses Kedi incrementally in the browser with the
`tree-sitter-kedi` WebAssembly grammar and the grammar's canonical
`queries/highlights.scm`. Hover requests continue to use the server's
`kedi.lsp` and Pyright integration. Rebuild the checked-in browser parser
assets after changing the grammar or highlight query:

```bash
make playground-tree-sitter
```
