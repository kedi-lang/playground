from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Mapping

from playground.execution import format_execution_output


class HttpPyodideBridge:
    def __init__(self, *, url: str, run_id: str, token: str) -> None:
        self._url = url
        self._run_id = run_id
        self._token = token

    def request(self, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]:
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
            raise TypeError("Internal Pyodide bridge response must be an object")
        return result


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("LOGFIRE_ENABLED", "").lower() in {"1", "on", "true", "yes"}:
        token = os.environ.get("LOGFIRE_TOKEN")
        if not token:
            raise ValueError("LOGFIRE_TOKEN is required when LOGFIRE_ENABLED is true")
        import logfire

        logfire.configure(token=token, send_to_logfire=True, console=False)
        logfire.instrument_pydantic_ai()

    from kedi.agent_adapter.adapters import PydanticAdapter
    from kedi.executors import PyodideExecutor
    from kedi.lang import compile_program, parse_program
    from kedi.model_normalization import normalize_for_pydantic_ai

    source = payload.get("source")
    model = payload.get("model")
    settings = payload.get("settings", {})
    run_id = payload.get("runId")
    bridge_url = payload.get("bridgeUrl")
    bridge_token = payload.get("bridgeToken")
    if (
        not isinstance(source, str)
        or not isinstance(model, str)
        or not isinstance(run_id, str)
        or not isinstance(bridge_url, str)
        or not isinstance(bridge_token, str)
    ):
        raise TypeError("source, model, and bridge settings must be strings")
    if not isinstance(settings, dict):
        raise TypeError("settings must be an object")
    adapter = PydanticAdapter(
        model=normalize_for_pydantic_ai(model),
        model_settings=settings,
    )
    executor = PyodideExecutor(
        HttpPyodideBridge(
            url=bridge_url,
            run_id=run_id,
            token=bridge_token,
        )
    )

    program = parse_program(source, source_path="<playground-byok>")
    runtime = compile_program(program, adapter=adapter, executor=executor)
    result = runtime.run_main()
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
        response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    stdout.write(json.dumps(response))


if __name__ == "__main__":
    main()
