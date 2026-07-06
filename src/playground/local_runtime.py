from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

from opentelemetry.trace import get_tracer

from playground.execution import format_execution_output
from playground.nsjail import NsJailPool
from playground.telemetry import WebGPUTelemetry
from playground.worker_process import HttpBridge, _configure_logfire

_SANDBOX_IMPORT_RE = re.compile(r"(?m)^\s*>\s*import:\s*sandbox\s*$")
_ENV_LOCK = threading.Lock()


def run_webgpu_host(
    *,
    source: str,
    model: str,
    supported_models: Sequence[str],
    secrets: Mapping[str, object],
    settings: Mapping[str, object],
    run_id: str,
    bridge_url: str,
    bridge_token: str,
    nsjail_pool: NsJailPool | None,
) -> dict[str, Any]:
    with _temporary_environment(secrets):
        return _run_webgpu_host_in_environment(
            source=source,
            model=model,
            supported_models=supported_models,
            settings=settings,
            run_id=run_id,
            bridge_url=bridge_url,
            bridge_token=bridge_token,
            nsjail_pool=nsjail_pool,
        )


def _run_webgpu_host_in_environment(
    *,
    source: str,
    model: str,
    supported_models: Sequence[str],
    settings: Mapping[str, object],
    run_id: str,
    bridge_url: str,
    bridge_token: str,
    nsjail_pool: NsJailPool | None,
) -> dict[str, Any]:
    from kedi.agent_adapter.webgpu import WebGPUAdapter
    from kedi.executors import PlaygroundExecutor
    from kedi.lang import compile_program, parse_program

    logfire_enabled = _configure_logfire(instrument_pydantic_ai=False)
    telemetry = (
        WebGPUTelemetry(
            get_tracer("kedi.playground.webgpu"),
            run_id=run_id,
            model=model,
        )
        if logfire_enabled
        else None
    )
    bridge = HttpBridge(
        url=bridge_url,
        run_id=run_id,
        token=bridge_token,
        telemetry=telemetry,
    )
    adapter = WebGPUAdapter(
        bridge,
        model=model,
        supported_models=supported_models,
        model_settings=settings,
        allowed_mcp_transports={"http", "sse"},
        instrumentation=telemetry,
    )

    python_bridge = _RunPythonBridge(
        browser_bridge=bridge,
        nsjail_pool=nsjail_pool,
        requires_native_sandbox=_requires_native_sandbox(source),
    )
    try:
        executor = PlaygroundExecutor(python_bridge)
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
    finally:
        python_bridge.close()


class _RunPythonBridge:
    def __init__(
        self,
        *,
        browser_bridge: HttpBridge,
        nsjail_pool: NsJailPool | None,
        requires_native_sandbox: bool,
    ) -> None:
        self._browser_bridge = browser_bridge
        self._pool = nsjail_pool
        self._requires_native_sandbox = requires_native_sandbox
        self._lease_context: Any = None
        self._worker: Any = None
        self._fallback_to_browser = False

    def request(self, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]:
        if self._worker is None and not self._fallback_to_browser:
            self._select_backend()
        if self._worker is not None:
            return self._worker.request(payload, timeout=timeout)
        return self._browser_bridge.request(payload, timeout=timeout)

    def close(self) -> None:
        if self._lease_context is None:
            return
        self._lease_context.__exit__(None, None, None)
        self._lease_context = None
        self._worker = None

    def _select_backend(self) -> None:
        if self._pool is not None:
            lease_context = self._pool.lease()
            worker = lease_context.__enter__()
            if worker is not None:
                self._lease_context = lease_context
                self._worker = worker
                return
            lease_context.__exit__(None, None, None)
        if self._requires_native_sandbox:
            raise RuntimeError(
                "pydantic_monty is not supported by the Pyodide fallback; "
                "server sandbox execution is unavailable"
            )
        self._fallback_to_browser = True


@contextmanager
def _temporary_environment(secrets: Mapping[str, object]):
    updates = {name: value for name, value in secrets.items() if isinstance(value, str) and value}
    original = {name: os.environ.get(name) for name in updates}
    with _ENV_LOCK:
        try:
            os.environ.update(updates)
            yield
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _requires_native_sandbox(source: str) -> bool:
    return "pydantic_monty" in source or _SANDBOX_IMPORT_RE.search(source) is not None


__all__ = ["run_webgpu_host"]
