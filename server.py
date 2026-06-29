from __future__ import annotations

import asyncio
import os
import secrets
import sys
import threading
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, get_args

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from playground.bridge import BridgeManager
from playground.byok import run_byok
from playground.models import MODEL_BY_ID, browser_model, public_model_registry

ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
REPO_ROOT = ROOT.parent
BRIDGES = BridgeManager()
POLL_SECONDS = 25
_INTERNAL_BRIDGE_LOCK = threading.Lock()
_INTERNAL_BRIDGE_TOKENS: dict[str, str] = {}

app = FastAPI(title="Kedi Playground", docs_url=None, redoc_url=None)


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


class LocalRunPayload(_CamelPayload):
    source: str
    model_id: str = Field(alias="modelId")
    run_id: str = Field(alias="runId")
    settings: ModelSettingsPayload = Field(default_factory=ModelSettingsPayload)


class ByokRunPayload(BaseModel):
    source: str
    model: str
    run_id: str = Field(alias="runId")
    secrets: dict[str, str] = Field(default_factory=dict)
    settings: ModelSettingsPayload = Field(default_factory=ModelSettingsPayload)


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


def _ensure_repo_importable() -> None:
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def run_local(
    *,
    source: str,
    model_id: str,
    run_id: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    _ensure_repo_importable()
    from kedi.agent_adapter.webgpu import WebGPUAdapter
    from kedi.executors import PyodideExecutor
    from kedi.lang import compile_program, parse_program

    browser_model(model_id)
    bridge = BRIDGES.get_or_create(run_id)
    program = parse_program(source, source_path="<playground>")
    adapter = WebGPUAdapter(
        bridge,
        model=model_id,
        model_settings=settings,
        supported_models=tuple(MODEL_BY_ID),
    )
    runtime = compile_program(
        program,
        adapter=adapter,
        executor=PyodideExecutor(bridge),
    )
    result = runtime.run_main()
    return {"ok": True, "result": "" if result is None else str(result)}


@app.get("/api/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/models")
async def models() -> dict[str, Any]:
    return {"ok": True, "models": public_model_registry()}


@app.get("/api/byok/models")
async def byok_models() -> dict[str, Any]:
    return {"ok": True, "models": _known_pydantic_models()}


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
async def local_run(payload: LocalRunPayload) -> JSONResponse:
    bridge = BRIDGES.get_or_create(payload.run_id)
    try:
        result = await asyncio.to_thread(
            run_local,
            source=payload.source,
            model_id=payload.model_id,
            run_id=payload.run_id,
            settings=payload.settings.as_dict(),
        )
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001 - API boundary formats Kedi failures.
        return _error_response(exc)
    finally:
        bridge.finish()
        BRIDGES.cleanup()


@app.post("/api/run/byok")
async def byok_run(payload: ByokRunPayload, request: Request) -> JSONResponse:
    bridge = BRIDGES.get_or_create(payload.run_id)
    token = secrets.token_urlsafe(32)
    with _INTERNAL_BRIDGE_LOCK:
        _INTERNAL_BRIDGE_TOKENS[payload.run_id] = token
    try:
        bridge_url = os.environ.get(
            "PLAYGROUND_INTERNAL_BRIDGE_URL",
            f"{str(request.base_url).rstrip('/')}/api/internal/bridge/request",
        )
        result = await asyncio.to_thread(
            run_byok,
            source=payload.source,
            model=payload.model,
            secrets=payload.secrets,
            settings=payload.settings.as_dict(),
            run_id=payload.run_id,
            bridge_url=bridge_url,
            bridge_token=token,
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)
    except Exception as exc:  # noqa: BLE001 - API boundary returns a compact error.
        return _error_response(exc)
    finally:
        with _INTERNAL_BRIDGE_LOCK:
            _INTERNAL_BRIDGE_TOKENS.pop(payload.run_id, None)
        bridge.finish()
        BRIDGES.cleanup()


def _error_response(exc: Exception) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
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
