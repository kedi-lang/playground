# Kedi Playground

Kedi Playground runs Kedi programs with browser-local WebGPU models or a
Pydantic AI provider selected through BYOK. User-authored Python executes in an
isolated Pyodide worker. Model weights download directly to the visitor's
browser cache and are not stored by the server.

Runtime selection is owned by the model registry. The UI exposes model names,
not browser inference engines.

## Run locally

```bash
uv sync --group dev
gh run download \
  --name pydantic-monty-pyodide \
  --dir src/playground/static/vendor
uv run kedi-playground
```

Open <http://127.0.0.1:8787>.

The wheel artifact is produced by the latest successful `space.yml` workflow.
Run that workflow manually first when no artifact is available.

## Docker

The default development image installs the `stable` branch from the private
Kedi repository:

```bash
KEDI_GITHUB_TOKEN="$(gh auth token)" \
  docker build \
  --secret id=KEDI_GITHUB_TOKEN,env=KEDI_GITHUB_TOKEN \
  -t kedi-playground .
docker run --rm -p 7860:7860 kedi-playground
```

Select another development revision explicitly:

```bash
docker build \
  --build-arg KEDI_INSTALL_MODE=dev \
  --build-arg KEDI_REVISION=stable \
  -t kedi-playground:dev .
```

Production mode installs the latest published `kedi[playground]` from PyPI:

```bash
docker build --build-arg KEDI_INSTALL_MODE=prod -t kedi-playground:prod .
```

The GitHub token is mounted as a BuildKit secret only for dependency
installation and is not stored in an image layer. The container listens on port
`7860` and runs as user `1000`.

## Hugging Face Space

The repository is configured as a Docker Space on port `7860`. GitHub Actions
runs tests, builds the package and container, then syncs successful `main`
pushes to Hugging Face.

Configure the GitHub repository with:

- Repository secret `HF_TOKEN`: a Hugging Face token with write access.
- Repository secret `KEDI_GITHUB_TOKEN`: a read-only token for the private
  `kedi-lang/kedi` repository.
- Repository variable `HF_SPACE_ID`: the target Space as `owner/space`.

The workflow builds and tests a Pyodide-compatible `pydantic-monty` wheel,
creates the Docker Space when needed, installs `KEDI_GITHUB_TOKEN` as an HF
Space build secret, then uploads a production staging directory. Hugging Face
rebuilds the Space after each upload.

## Runtime behavior

- Bonsai GGUF uses wllama and browser-owned OPFS storage.
- Ternary Bonsai ONNX uses Transformers.js and the browser cache.
- Custom `.gguf` and `.onnx` models remain browser-local.
- Embedded Python uses Pyodide 314 and the workflow-built Monty WebAssembly
  wheel.
- BYOK credentials and `HF_TOKEN` remain in browser local storage.
- Model settings and source code remain in browser session storage.
- Remote MCP supports HTTP and SSE; `stdio` is rejected.

Both WebGPU and BYOK runs execute in a short-lived worker. BYOK uses
`PydanticAdapter`; browser inference uses `WebGPUAdapter`. Logfire is configured
inside the worker only when `LOGFIRE_ENABLED` is set and a `LOGFIRE_TOKEN` is
present, so per-user tokens never mutate the long-lived FastAPI process.

The playground server has no model download or model-file endpoint.
User-authored Python and type expressions execute in an isolated Pyodide Web
Worker and never run in the server's Python process. `WebGPUAdapter` owns tool
registration and execution, remote MCP discovery, profile overrides, output
validation and the browser model loop.

WebGPU context management uses a percentage-based soft limit. wllama expands
its context automatically when the request approaches 80% of the active
window. Transformers.js reserves output space using its tokenizer count. BYOK
requests are not altered by this browser-only policy.

The Monaco editor parses Kedi incrementally with the `tree-sitter-kedi`
WebAssembly grammar and its canonical highlight query. Hover requests use the
server's `kedi.lsp` and Pyright integration. Rebuild the checked-in browser
parser assets after changing the grammar or highlight query:

```bash
make playground-tree-sitter
```
