from __future__ import annotations

import asyncio
import os
import secrets
import threading
from argparse import ArgumentParser
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, cast, get_args

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from playground.bridge import BridgeManager
from playground.execution import execution_error_payload
from playground.models import MODEL_BY_ID, browser_model, public_model_registry
from playground.pyright import PyrightServer
from playground.worker import run_provider, run_webgpu

PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PACKAGE_ROOT / "static"
BRIDGES = BridgeManager()
POLL_SECONDS = 25
_INTERNAL_BRIDGE_LOCK = threading.Lock()
_INTERNAL_BRIDGE_TOKENS: dict[str, str] = {}
PYRIGHT = PyrightServer()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    PYRIGHT.close()


app = FastAPI(title="Kedi Playground", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.middleware("http")
async def add_browser_isolation_headers(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok"}


class _CamelPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class ModelSettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    temperature: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1, le=512)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class BrowserModelPayload(_CamelPayload):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    engine: Literal["wllama", "transformers"]
    repo: str = Field(min_length=1)
    file: str = Field(min_length=1)
    model: str | None = None
    dtype: Literal["q2", "q4", "q4f16", "q8", "fp16", "fp32"] | None = None
    device: Literal["webgpu"] = "webgpu"

    @model_validator(mode="after")
    def validate_runtime_fields(self) -> BrowserModelPayload:
        suffix = Path(self.file).suffix.lower()
        if self.engine == "wllama" and suffix != ".gguf":
            raise ValueError("wllama models require a .gguf file")
        if self.engine == "transformers":
            if suffix != ".onnx":
                raise ValueError("Transformers.js models require a .onnx file")
            if not self.dtype:
                raise ValueError("Transformers.js models require a dtype")
            if self.model != self.repo:
                raise ValueError("Transformers.js model must match the repository")
        return self


class LocalRunPayload(_CamelPayload):
    source: str
    model_id: str = Field(alias="modelId")
    browser_model: BrowserModelPayload | None = Field(default=None, alias="modelConfig")
    run_id: str = Field(alias="runId")
    secrets: dict[str, str] = Field(default_factory=dict)
    settings: ModelSettingsPayload = Field(default_factory=ModelSettingsPayload)


class ByokRunPayload(BaseModel):
    source: str
    model: str
    run_id: str = Field(alias="runId")
    secrets: dict[str, str] = Field(default_factory=dict)
    settings: ModelSettingsPayload = Field(default_factory=ModelSettingsPayload)


class PydanticModelValidationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)


class LspSourcePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source: str = Field(max_length=200_000)


class HoverPayload(LspSourcePayload):
    line: int = Field(ge=0)
    character: int = Field(ge=0)


class CancelPayload(_CamelPayload):
    run_id: str = Field(alias="runId")


class BridgeResponsePayload(_CamelPayload):
    run_id: str = Field(alias="runId")
    request_id: str = Field(alias="requestId")
    response: dict[str, Any]


class InternalBridgePayload(_CamelPayload):
    run_id: str = Field(alias="runId")
    token: str
    request: dict[str, Any]
    timeout: float = 60


@app.get("/api/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/models")
async def models() -> dict[str, Any]:
    return {"ok": True, "models": public_model_registry()}


@app.get("/api/byok/models")
async def byok_models() -> dict[str, Any]:
    return {"ok": True, "models": _known_pydantic_models()}


@app.post("/api/lsp/hover")
async def hover(payload: HoverPayload) -> dict[str, Any]:
    from lsprotocol import types as lsp

    from kedi.lsp.features import compute_hover

    python_hover = await asyncio.to_thread(
        PYRIGHT.hover,
        payload.source,
        payload.line,
        payload.character,
    )
    if python_hover is not None:
        return {"ok": True, "hover": python_hover, "provider": "pyright"}

    result = compute_hover(
        payload.source,
        source_path=None,
        pos=lsp.Position(line=payload.line, character=payload.character),
    )
    if result is None:
        return {"ok": True, "hover": None, "provider": "kedi"}

    contents = cast(lsp.MarkupContent, result.contents)
    range_ = result.range
    return {
        "ok": True,
        "provider": "kedi",
        "hover": {
            "contents": {
                "kind": contents.kind,
                "value": contents.value,
            },
            "range": (
                {
                    "start": {
                        "line": range_.start.line,
                        "character": range_.start.character,
                    },
                    "end": {
                        "line": range_.end.line,
                        "character": range_.end.character,
                    },
                }
                if range_ is not None
                else None
            ),
        },
    }


@app.post("/api/byok/models/validate")
async def validate_byok_model(payload: PydanticModelValidationPayload) -> JSONResponse:
    try:
        model = _validate_pydantic_model_name(payload.model)
        return JSONResponse({"ok": True, "model": model})
    except Exception as exc:  # noqa: BLE001 - validation errors cross an API boundary.
        return _error_response(exc)


@app.get("/api/bridge/request")
async def bridge_request(runId: str) -> dict[str, Any]:
    run = BRIDGES.get_or_create(runId)
    request = await asyncio.to_thread(run.next_request, timeout=POLL_SECONDS)
    return {
        "ok": True,
        "done": run.done and request is None,
        "cancelled": run.cancelled,
        "request": request,
    }


@app.post("/api/bridge/response")
async def bridge_response(payload: BridgeResponsePayload) -> JSONResponse:
    try:
        BRIDGES.get_or_create(payload.run_id).submit_response(
            payload.request_id,
            payload.response,
        )
        return JSONResponse({"ok": True})
    except Exception as exc:  # noqa: BLE001 - bridge validation is an API boundary.
        return _error_response(exc)


@app.post("/api/internal/bridge/request")
async def internal_bridge_request(payload: InternalBridgePayload) -> JSONResponse:
    try:
        with _INTERNAL_BRIDGE_LOCK:
            expected = _INTERNAL_BRIDGE_TOKENS.get(payload.run_id)
        if expected is None or not secrets.compare_digest(expected, payload.token):
            return JSONResponse(
                {"ok": False, "error": "Invalid internal bridge token"},
                status_code=403,
            )
        response = await asyncio.to_thread(
            BRIDGES.get_or_create(payload.run_id).request,
            payload.request,
            timeout=payload.timeout,
        )
        return JSONResponse(dict(response))
    except Exception as exc:  # noqa: BLE001 - internal bridge is a process boundary.
        return _error_response(exc)


@app.post("/api/run/cancel")
async def cancel_run(payload: CancelPayload) -> dict[str, bool]:
    BRIDGES.cancel(payload.run_id)
    return {"ok": True}


@app.post("/api/run/local")
async def local_run(payload: LocalRunPayload, request: Request) -> JSONResponse:
    bridge = BRIDGES.get_or_create(payload.run_id)
    try:
        if payload.browser_model is not None and payload.browser_model.id != payload.model_id:
            raise ValueError("Custom model ID does not match the selected model")
        supported_models = tuple(MODEL_BY_ID)
        if payload.browser_model is None:
            browser_model(payload.model_id)
        else:
            supported_models = (*supported_models, payload.model_id)
        with _internal_bridge(payload.run_id, request) as (bridge_url, bridge_token):
            result = await asyncio.to_thread(
                run_webgpu,
                source=payload.source,
                model=payload.model_id,
                supported_models=supported_models,
                secrets=payload.secrets,
                settings=payload.settings.as_dict(),
                run_id=payload.run_id,
                bridge_url=bridge_url,
                bridge_token=bridge_token,
            )
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)
    except Exception as exc:  # noqa: BLE001 - API boundary formats Kedi failures.
        return _error_response(exc)
    finally:
        bridge.finish()
        BRIDGES.cleanup()


@app.post("/api/run/byok")
async def byok_run(payload: ByokRunPayload, request: Request) -> JSONResponse:
    bridge = BRIDGES.get_or_create(payload.run_id)
    try:
        with _internal_bridge(payload.run_id, request) as (bridge_url, bridge_token):
            result = await asyncio.to_thread(
                run_provider,
                source=payload.source,
                model=payload.model,
                secrets=payload.secrets,
                settings=payload.settings.as_dict(),
                run_id=payload.run_id,
                bridge_url=bridge_url,
                bridge_token=bridge_token,
            )
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)
    except Exception as exc:  # noqa: BLE001 - API boundary returns a compact error.
        return _error_response(exc)
    finally:
        bridge.finish()
        BRIDGES.cleanup()


@contextmanager
def _internal_bridge(run_id: str, request: Request) -> Iterator[tuple[str, str]]:
    token = secrets.token_urlsafe(32)
    with _INTERNAL_BRIDGE_LOCK:
        _INTERNAL_BRIDGE_TOKENS[run_id] = token
    try:
        url = os.environ.get(
            "PLAYGROUND_INTERNAL_BRIDGE_URL",
            f"{str(request.base_url).rstrip('/')}/api/internal/bridge/request",
        )
        yield url, token
    finally:
        with _INTERNAL_BRIDGE_LOCK:
            _INTERNAL_BRIDGE_TOKENS.pop(run_id, None)


def _error_response(exc: Exception) -> JSONResponse:
    return JSONResponse(
        execution_error_payload(exc),
        status_code=400,
    )


def _known_pydantic_models() -> list[dict[str, str]]:
    from pydantic_ai.models import KnownModelName

    names = get_args(KnownModelName.__value__)
    models: list[dict[str, str]] = []
    for name in names:
        if not isinstance(name, str):
            continue
        provider, separator, model_id = name.partition(":")
        label = f"{model_id} ({provider})" if separator else name
        models.append({"id": name, "label": label, "provider": provider})
    return models


def _validate_pydantic_model_name(model: str) -> str:
    from pydantic_ai.models import infer_model

    model = model.strip()
    provider, separator, model_name = model.partition(":")
    if not separator or not provider or not model_name:
        raise ValueError("Model ID must use provider:model format")
    try:
        infer_model(model)
    except Exception as exc:
        message = str(exc)
        if message.startswith(("Unknown provider:", "Unknown model:")):
            raise ValueError(message) from exc
    return model


app.mount("/", StaticFiles(directory=STATIC_ROOT, html=True), name="static")


def main() -> None:
    parser = ArgumentParser(description="Run the Kedi browser playground.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8787")))
    args = parser.parse_args()

    import uvicorn

    print(f"Kedi playground: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
