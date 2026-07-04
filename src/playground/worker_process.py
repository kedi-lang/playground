from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
from collections.abc import Mapping
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from typing import Any

from opentelemetry.trace import get_tracer

from playground.execution import execution_error_payload, format_execution_output
from playground.telemetry import WebGPUTelemetry


class HttpBridge:
    def __init__(
        self,
        *,
        url: str,
        run_id: str,
        token: str,
        telemetry: WebGPUTelemetry | None = None,
    ) -> None:
        self._url = url
        self._run_id = run_id
        self._token = token
        self._telemetry = telemetry

    def request(self, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]:
        span = (
            self._telemetry.model_request_span(payload)
            if self._telemetry is not None and payload.get("operation") != "python"
            else nullcontext(lambda _: None)
        )
        with span as finish:
            body = json.dumps(
                {
                    "runId": self._run_id,
                    "token": self._token,
                    "request": dict(payload),
                    "timeout": timeout,
                }
            ).encode()
            request = urllib.request.Request(
                self._url,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout + 5) as response:
                result = json.loads(response.read())
            if not isinstance(result, dict):
                raise TypeError("Internal playground bridge response must be an object")
            finish(result)
        return result


def _configure_logfire(*, instrument_pydantic_ai: bool) -> bool:
    if os.environ.get("LOGFIRE_ENABLED", "").lower() in {"1", "on", "true", "yes"}:
        token = os.environ.get("LOGFIRE_TOKEN")
        if not token:
            raise ValueError("LOGFIRE_TOKEN is required when LOGFIRE_ENABLED is true")
        import logfire

        logfire.configure(
            token=token,
            send_to_logfire=True,
            service_name="kedi-playground",
            console=False,
        )
        if instrument_pydantic_ai:
            logfire.instrument_pydantic_ai()
        return True
    return False


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    from kedi.agent_adapter.adapters import PydanticAdapter
    from kedi.agent_adapter.webgpu import WebGPUAdapter
    from kedi.executors import PyodideExecutor
    from kedi.lang import compile_program, parse_program
    from kedi.model_normalization import normalize_for_pydantic_ai

    source = payload.get("source")
    mode = payload.get("mode")
    model = payload.get("model")
    supported_models = payload.get("supportedModels", [])
    settings = payload.get("settings", {})
    run_id = payload.get("runId")
    bridge_url = payload.get("bridgeUrl")
    bridge_token = payload.get("bridgeToken")
    if (
        not isinstance(source, str)
        or mode not in {"provider", "webgpu"}
        or not isinstance(model, str)
        or not isinstance(run_id, str)
        or not isinstance(bridge_url, str)
        or not isinstance(bridge_token, str)
    ):
        raise TypeError("mode, source, model, and bridge settings are invalid")
    if not isinstance(settings, dict):
        raise TypeError("settings must be an object")
    if not isinstance(supported_models, list) or not all(
        isinstance(item, str) for item in supported_models
    ):
        raise TypeError("supportedModels must be a list of strings")

    logfire_enabled = _configure_logfire(instrument_pydantic_ai=mode == "provider")
    telemetry = (
        WebGPUTelemetry(
            get_tracer("kedi.playground.webgpu"),
            run_id=run_id,
            model=model,
        )
        if logfire_enabled and mode == "webgpu"
        else None
    )
    bridge = HttpBridge(
        url=bridge_url,
        run_id=run_id,
        token=bridge_token,
        telemetry=telemetry,
    )
    if mode == "webgpu":
        adapter = WebGPUAdapter(
            bridge,
            model=model,
            supported_models=supported_models,
            model_settings=settings,
            allowed_mcp_transports={"http", "sse"},
            instrumentation=telemetry,
        )
    else:
        adapter = PydanticAdapter(
            model=normalize_for_pydantic_ai(model),
            model_settings=settings,
        )
    executor = PyodideExecutor(bridge)

    program = parse_program(source, source_path="<playground>")
    runtime = compile_program(program, adapter=adapter, executor=executor)
    run_span = telemetry.run_span() if telemetry is not None else nullcontext(lambda _: None)
    with run_span as finish:
        result = runtime.run_main()
        finish(result)
    return {
        "ok": True,
        "result": format_execution_output(executor.drain_stdout(), result),
    }


def main() -> None:
    stdout = sys.stdout
    try:
        payload = json.loads(sys.stdin.read())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            response = execute(payload)
    except Exception as exc:  # noqa: BLE001 - process boundary returns a compact error.
        response = execution_error_payload(exc)
    stdout.write(json.dumps(response))


if __name__ == "__main__":
    main()
