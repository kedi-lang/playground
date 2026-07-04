from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from kedi.errors import KediExecutionError, KediPythonTrace, KediTraceFrame
from kedi.lang import parse_program
from kedi.lang.ast import Span

from playground import server, worker, worker_process
from playground.bridge import BridgeCancelled, BridgeManager, BridgeRun
from playground.execution import (
    execution_error_diagnostic,
    execution_error_payload,
    format_execution_output,
)
from playground.models import browser_model, public_model_registry
from playground.pyright import PyrightServer
from playground.telemetry import WebGPUTelemetry


def _run_provider(
    *,
    source: str,
    model: str,
    secrets: dict[str, object],
    settings: dict[str, object] | None = None,
) -> dict[str, Any]:
    return worker.run_provider(
        source=source,
        model=model,
        secrets=secrets,
        settings=settings or {},
        run_id="run",
        bridge_url="http://127.0.0.1/internal",
        bridge_token="bridge-token",
    )


def _run_webgpu(
    *,
    source: str,
    secrets: dict[str, object],
    settings: dict[str, object] | None = None,
) -> dict[str, Any]:
    return worker.run_webgpu(
        source=source,
        model="bonsai",
        supported_models=("bonsai",),
        secrets=secrets,
        settings=settings or {},
        run_id="run",
        bridge_url="http://127.0.0.1/internal",
        bridge_token="bridge-token",
    )


def test_model_registry_is_public_and_rejects_unknown_model() -> None:
    models = public_model_registry()
    assert [model["engine"] for model in models] == ["wllama", "transformers"]
    assert browser_model("bonsai-1.7b-q1").file == "Bonsai-1.7B-Q1_0.gguf"
    with pytest.raises(ValueError, match="Unknown browser model"):
        browser_model("missing")


def test_health_endpoint_and_space_container_configuration() -> None:
    response = server.healthz()
    playground_root = Path(server.__file__).resolve().parents[2]
    dockerfile = (playground_root / "Dockerfile").read_text(encoding="utf-8")
    readme = (playground_root / "README.md").read_text(encoding="utf-8")
    workflow = (playground_root / ".github" / "workflows" / "space.yml").read_text(
        encoding="utf-8"
    )

    assert response == {"status": "ok"}
    assert "ARG KEDI_INSTALL_MODE=dev" in dockerfile
    assert "ARG KEDI_REVISION=stable" in dockerfile
    assert "build-essential" in dockerfile
    assert "--mount=type=secret,id=KEDI_GITHUB_TOKEN,required=false" in dockerfile
    assert '"kedi[playground]"' in dockerfile
    assert '"kedi[playground] @ git+${KEDI_REPOSITORY}@${KEDI_REVISION}"' in dockerfile
    assert "KEDI_INSTALL_MODE must be prod or dev" in dockerfile
    assert "from kedi.agent_adapter import WebGPUAdapter" in dockerfile
    assert "from kedi.executors import PyodideExecutor" in dockerfile
    assert "USER user" in dockerfile
    assert "PORT=7860" in dockerfile
    assert "PYTHONPATH=/home/user/app/src" in dockerfile
    assert "COPY --chown=user:user src ./src" in dockerfile
    assert "models" in (playground_root / ".dockerignore").read_text(encoding="utf-8")
    assert not readme.startswith("---")
    assert "`HF_TOKEN`" in readme
    assert "`HF_SPACE_ID`" in readme
    assert "`KEDI_GITHUB_TOKEN`" in readme
    assert "huggingface/hub-sync@v0.1.0" in workflow
    assert workflow.count("uses: actions/checkout@v4") == 2
    assert "needs: verify" in workflow
    assert "space_sdk: docker" in workflow
    assert "api.add_space_secret(" in workflow


def test_tutorial_examples_are_parseable_and_tool_usage_stays_focused() -> None:
    app_source = (Path(server.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    examples_match = re.search(
        r"const EXAMPLES = Object\.freeze\(\{(?P<body>.*?)\}\);",
        app_source,
        re.DOTALL,
    )

    assert examples_match is not None
    examples = dict(
        re.findall(
            r"^\s*(\w+): `(.*?)`,?$",
            examples_match.group("body"),
            re.DOTALL | re.MULTILINE,
        )
    )
    assert set(examples) == {"capital", "contact", "delivery"}
    assert sum("> use:" in source for source in examples.values()) == 1
    assert all("[" in source and "]" in source for source in examples.values())
    assert "Contact extraction" in (
        Path(server.__file__).parent / "static" / "index.html"
    ).read_text(encoding="utf-8")
    assert "Delivery tool" in (Path(server.__file__).parent / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    for source in examples.values():
        parse_program(source, source_path="<tutorial>")


def test_model_settings_payload_accepts_only_shared_adapter_settings() -> None:
    settings = server.ModelSettingsPayload(
        temperature=0.2,
        max_tokens=64,
        top_p=0.8,
        seed=42,
    )

    assert settings.as_dict() == {
        "temperature": 0.2,
        "max_tokens": 64,
        "top_p": 0.8,
        "seed": 42,
    }
    assert server.ModelSettingsPayload().as_dict() == {}
    with pytest.raises(ValueError):
        server.ModelSettingsPayload(temperature=1.01)
    with pytest.raises(ValueError):
        server.ModelSettingsPayload.model_validate({"frequency_penalty": 1})


def test_playground_lsp_hover_uses_kedi_hover_content() -> None:
    source = "@greet(name: str):\n  = Hello <name>\n"
    result = asyncio.run(
        server.hover(
            server.HoverPayload(
                source=source,
                line=0,
                character=2,
            )
        )
    )
    missing = asyncio.run(
        server.hover(
            server.HoverPayload(
                source=source,
                line=3,
                character=0,
            )
        )
    )

    assert result["hover"]["contents"]["kind"] == "markdown"
    assert result["provider"] == "kedi"
    assert "@greet" in result["hover"]["contents"]["value"]
    assert result["hover"]["range"]["start"] == {"line": 0, "character": 1}
    assert missing == {"ok": True, "hover": None, "provider": "kedi"}


def test_playground_pyright_hover_maps_virtual_python_back_to_kedi() -> None:
    pyright = PyrightServer(timeout=10)
    try:
        hover = pyright.hover(
            '[name: str] = `"Ada"`\n= <`name.upper()`>\n',
            1,
            5,
        )
    finally:
        pyright.close()

    assert hover is not None
    assert "(variable) name:" in hover["contents"]["value"]
    assert hover["range"] == {
        "start": {"line": 1, "character": 4},
        "end": {"line": 1, "character": 8},
    }


def test_custom_browser_model_payload_validates_runtime_contract() -> None:
    gguf = server.BrowserModelPayload(
        id="custom-gguf",
        label="Local",
        engine="wllama",
        repo="organization/model",
        file="model.gguf",
    )
    onnx = server.BrowserModelPayload(
        id="custom-onnx",
        label="Browser",
        engine="transformers",
        repo="organization/model",
        file="onnx/model_q4.onnx",
        model="organization/model",
        dtype="q4",
    )

    assert gguf.device == "webgpu"
    assert onnx.dtype == "q4"
    with pytest.raises(ValueError, match=r"\.gguf"):
        server.BrowserModelPayload(
            id="bad",
            label="Bad",
            engine="wllama",
            repo="organization/model",
            file="model.onnx",
        )
    with pytest.raises(ValueError, match=r"\.onnx"):
        server.BrowserModelPayload(
            id="bad",
            label="Bad",
            engine="transformers",
            repo="organization/model",
            file="model.gguf",
            model="organization/model",
            dtype="q4",
        )
    with pytest.raises(ValueError, match="dtype"):
        server.BrowserModelPayload(
            id="bad",
            label="Bad",
            engine="transformers",
            repo="organization/model",
            file="model.onnx",
            model="organization/model",
        )
    with pytest.raises(ValueError, match="match the repository"):
        server.BrowserModelPayload(
            id="bad",
            label="Bad",
            engine="transformers",
            repo="organization/model",
            file="model.onnx",
            model="other/model",
            dtype="q4",
        )


def test_pydantic_model_name_validation_uses_infer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai import models

    seen: list[str] = []

    def accept(model: str) -> object:
        seen.append(model)
        return object()

    monkeypatch.setattr(models, "infer_model", accept)
    assert (
        server._validate_pydantic_model_name(" openrouter:google/gemini-3-flash-preview ")
        == "openrouter:google/gemini-3-flash-preview"
    )
    assert seen == ["openrouter:google/gemini-3-flash-preview"]

    monkeypatch.setattr(
        models,
        "infer_model",
        lambda _model: (_ for _ in ()).throw(RuntimeError("API key required")),
    )
    assert server._validate_pydantic_model_name("openai:gpt-4o-mini") == ("openai:gpt-4o-mini")

    monkeypatch.setattr(
        models,
        "infer_model",
        lambda _model: (_ for _ in ()).throw(ValueError("Unknown provider: nope")),
    )
    with pytest.raises(ValueError, match="Unknown provider"):
        server._validate_pydantic_model_name("nope:model")
    with pytest.raises(ValueError, match="provider:model"):
        server._validate_pydantic_model_name("missing-provider")


def test_execution_output_preserves_stdout_and_return_values() -> None:
    assert format_execution_output("The Zen\n", None) == "The Zen\n"
    assert format_execution_output("", "done") == "done"
    assert format_execution_output("printed", "done") == "printed\ndone"
    assert format_execution_output("printed\n", "done") == "printed\ndone"


def test_execution_error_payload_prefers_precise_playground_python_line() -> None:
    playground_span = Span(0, 20, 2, 1, 5, 4, source_path="<playground>")
    imported_span = Span(0, 20, 9, 1, 9, 8, source_path="/modules/helper.kedi")
    error = KediExecutionError(
        message="Python block error: ValueError: broken",
        frames=(
            KediTraceFrame(kind="return_block", label="return", span=playground_span),
            KediTraceFrame(kind="python_block", label="python", span=imported_span),
        ),
        python_traces=(
            KediPythonTrace(
                filename="<helper>",
                kind="block",
                code="raise ValueError('broken')",
                span=imported_span,
                traceback_lineno=1,
                python_lineno=1,
                kedi_lineno=9,
            ),
            KediPythonTrace(
                filename="<playground-python>",
                kind="block",
                code="raise ValueError('broken')",
                span=playground_span,
                traceback_lineno=3,
                python_lineno=3,
                kedi_lineno=4,
            ),
        ),
    )

    payload = execution_error_payload(error)

    assert payload["diagnostic"] == {
        "source": "<playground>",
        "line": 4,
        "column": 1,
        "message": "Python block error: ValueError: broken",
        "kind": "block",
    }


def test_execution_error_diagnostic_falls_back_to_playground_frame() -> None:
    playground_span = Span(0, 10, 3, 2, 3, 8, source_path="<playground-byok>")
    error = KediExecutionError(
        message="broken",
        frames=(
            KediTraceFrame(kind="python_block", label="missing", span=None),
            KediTraceFrame(
                kind="python_block",
                label="external",
                span=Span(0, 4, 8, 1, 8, 4, source_path="/module.kedi"),
            ),
            KediTraceFrame(kind="return", label="return", span=playground_span),
        ),
        python_traces=(
            KediPythonTrace(
                filename="<python>",
                kind="block",
                code="broken",
                span=None,
                traceback_lineno=1,
                python_lineno=1,
            ),
        ),
    )

    assert execution_error_diagnostic(error) == {
        "source": "<playground-byok>",
        "line": 3,
        "column": 2,
        "message": "broken",
        "kind": "return",
    }
    assert execution_error_diagnostic(error, source_paths={"<other>"}) is None
    assert execution_error_payload(ValueError("plain")) == {
        "ok": False,
        "error": "ValueError: plain",
    }


def test_playground_responses_enable_cross_origin_isolation() -> None:
    async def call_next(_request: Any) -> server.Response:
        return server.Response()

    response = asyncio.run(server.add_browser_isolation_headers(SimpleNamespace(), call_next))

    assert response.headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert response.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
    assert response.headers["Cross-Origin-Resource-Policy"] == "cross-origin"
    assert response.headers["Cache-Control"] == "no-store"


def test_bridge_request_response_and_duplicate_rejection() -> None:
    run = BridgeRun("run")
    result: list[dict[str, Any]] = []

    def request() -> None:
        result.append(dict(run.request({"model": "bonsai"}, timeout=1)))

    thread = threading.Thread(target=request)
    thread.start()
    pending = run.next_request(timeout=1)
    assert pending is not None
    assert pending["model"] == "bonsai"
    run.submit_response(pending["id"], {"kind": "final", "data": {"answer": "ok"}})
    with pytest.raises(ValueError, match="active bridge request"):
        run.submit_response("missing", {})
    thread.join()
    assert result == [{"kind": "final", "data": {"answer": "ok"}}]


def test_bridge_timeout_cancel_finish_and_cleanup() -> None:
    timed = BridgeRun("timeout")
    with pytest.raises(TimeoutError):
        timed.request({}, timeout=0)
    assert timed.next_request(timeout=0) is not None

    cancelled = BridgeRun("cancel")
    cancelled.cancel()
    with pytest.raises(BridgeCancelled):
        cancelled.request({}, timeout=1)

    finished = BridgeRun("finished")
    finished.finish()
    with pytest.raises(RuntimeError, match="already complete"):
        finished.request({}, timeout=1)
    assert finished.next_request(timeout=0) is None

    manager = BridgeManager()
    managed = manager.get_or_create("managed")
    assert manager.get_or_create("managed") is managed
    manager.cancel("managed")
    managed.updated_at = time.monotonic() - 20
    manager.cleanup(max_age=10)
    assert manager.get_or_create("managed") is not managed


def test_worker_validates_input_and_redacts_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="uppercase identifiers"):
        _run_provider(
            source="= ok",
            model="mistral:model",
            secrets={"bad-key": "secret"},
        )
    with pytest.raises(TypeError, match="values must be strings"):
        _run_provider(
            source="= ok",
            model="openai/gpt",
            secrets={"OPENAI_API_KEY": 1},
        )

    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="",
            stderr="provider rejected secret",
        ),
    )
    with pytest.raises(RuntimeError, match=r"provider rejected \[REDACTED\]"):
        _run_provider(
            source="= ok",
            model="openai/gpt",
            secrets={"OPENAI_API_KEY": "secret"},
        )


def test_provider_worker_contract_and_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def run(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(stdout=json.dumps({"ok": True, "result": "done"}), stderr="")

    monkeypatch.setattr(worker.subprocess, "run", run)
    result = _run_provider(
        source="= ok",
        model="groq/llama",
        secrets={
            "CUSTOM_PROVIDER_KEY": "token",
            "LOGFIRE_ENABLED": "true",
            "LOGFIRE_TOKEN": "logfire-token",
        },
    )

    assert result == {"ok": True, "result": "done"}
    assert captured["env"]["CUSTOM_PROVIDER_KEY"] == "token"
    assert captured["env"]["LOGFIRE_ENABLED"] == "true"
    assert captured["env"]["LOGFIRE_TOKEN"] == "logfire-token"
    assert "OPENAI_API_KEY" not in captured["env"]
    assert json.loads(captured["input"]) == {
        "mode": "provider",
        "source": "= ok",
        "model": "groq/llama",
        "supportedModels": [],
        "settings": {},
        "runId": "run",
        "bridgeUrl": "http://127.0.0.1/internal",
        "bridgeToken": "bridge-token",
    }

    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="[]", stderr=""),
    )
    with pytest.raises(TypeError, match="must be an object"):
        _run_provider(
            source="= ok",
            model="openai/gpt",
            secrets={"OPENAI_API_KEY": "token"},
        )


def test_webgpu_worker_contract_and_logfire_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def run(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(stdout='{"ok": true, "result": "done"}', stderr="")

    monkeypatch.setattr(worker.subprocess, "run", run)
    result = _run_webgpu(
        source="= ok",
        secrets={
            "LOGFIRE_ENABLED": "true",
            "LOGFIRE_TOKEN": "logfire-token",
        },
    )

    assert result == {"ok": True, "result": "done"}
    assert json.loads(captured["input"]) == {
        "mode": "webgpu",
        "source": "= ok",
        "model": "bonsai",
        "supportedModels": ["bonsai"],
        "settings": {},
        "runId": "run",
        "bridgeUrl": "http://127.0.0.1/internal",
        "bridgeToken": "bridge-token",
    }
    assert captured["env"]["LOGFIRE_ENABLED"] == "true"
    assert captured["env"]["LOGFIRE_TOKEN"] == "logfire-token"


def test_local_run_returns_http_error_for_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "run_webgpu",
        lambda **_: {
            "ok": False,
            "error": "KediNameError: Unknown variable: country",
            "diagnostic": {
                "source": "<playground>",
                "line": 7,
                "column": 1,
                "message": "Unknown variable: country",
                "kind": "return",
            },
        },
    )
    payload = server.LocalRunPayload(
        source="= <country>",
        modelId="bonsai-1.7b-q1",
        runId="failed-local-run",
    )

    response = asyncio.run(
        server.local_run(
            payload,
            SimpleNamespace(base_url="http://127.0.0.1:59863/"),
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "ok": False,
        "error": "KediNameError: Unknown variable: country",
        "diagnostic": {
            "source": "<playground>",
            "line": 7,
            "column": 1,
            "message": "Unknown variable: country",
            "kind": "return",
        },
    }


def test_worker_process_configures_logfire_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    logfire = ModuleType("logfire")
    logfire.configure = lambda **kwargs: events.append(("configure", kwargs))  # type: ignore[attr-defined]
    logfire.instrument_pydantic_ai = lambda: events.append(("instrument", {}))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "logfire", logfire)

    monkeypatch.delenv("LOGFIRE_ENABLED", raising=False)
    assert worker_process._configure_logfire(instrument_pydantic_ai=True) is False
    assert events == []

    monkeypatch.setenv("LOGFIRE_ENABLED", "true")
    with pytest.raises(ValueError, match="LOGFIRE_TOKEN is required"):
        worker_process._configure_logfire(instrument_pydantic_ai=True)

    monkeypatch.setenv("LOGFIRE_TOKEN", "token")
    assert worker_process._configure_logfire(instrument_pydantic_ai=False) is True
    assert events == [
        (
            "configure",
            {
                "token": "token",
                "send_to_logfire": True,
                "service_name": "kedi-playground",
                "console": False,
            },
        ),
    ]

    events.clear()
    assert worker_process._configure_logfire(instrument_pydantic_ai=True) is True
    assert events == [
        (
            "configure",
            {
                "token": "token",
                "send_to_logfire": True,
                "service_name": "kedi-playground",
                "console": False,
            },
        ),
        ("instrument", {}),
    ]


def test_worker_process_uses_direct_webgpu_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kedi.agent_adapter.webgpu as webgpu_module

    captured: dict[str, Any] = {}

    class Adapter:
        def __init__(self, bridge: Any, **kwargs: Any) -> None:
            captured["bridge"] = bridge
            captured["kwargs"] = kwargs

    monkeypatch.setattr(webgpu_module, "WebGPUAdapter", Adapter)
    monkeypatch.setattr(
        worker_process,
        "_configure_logfire",
        lambda **_: False,
    )

    result = worker_process.execute(
        {
            "mode": "webgpu",
            "source": "= ok",
            "model": "bonsai",
            "supportedModels": ["bonsai"],
            "settings": {"temperature": 0.2},
            "runId": "run",
            "bridgeUrl": "http://127.0.0.1/internal",
            "bridgeToken": "bridge-token",
        }
    )

    assert result == {"ok": True, "result": "ok"}
    assert isinstance(captured["bridge"], worker_process.HttpBridge)
    assert captured["kwargs"] == {
        "model": "bonsai",
        "supported_models": ["bonsai"],
        "model_settings": {"temperature": 0.2},
        "allowed_mcp_transports": {"http", "sse"},
        "instrumentation": None,
    }


def test_http_bridge_records_webgpu_model_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any]] = []

    class Telemetry:
        @contextmanager
        def model_request_span(
            self,
            payload: dict[str, Any],
        ) -> Iterator[Any]:
            events.append(("request", payload))
            yield lambda response: events.append(("response", response))

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return b'{"kind": "final", "data": {"answer": "ok"}}'

    monkeypatch.setattr(worker_process.urllib.request, "urlopen", lambda *_a, **_k: Response())

    bridge = worker_process.HttpBridge(
        url="http://127.0.0.1/internal",
        run_id="run",
        token="token",
        telemetry=Telemetry(),  # type: ignore[arg-type]
    )
    result = bridge.request(
        {"model": "bonsai", "step": 2, "tools": [{"name": "lookup"}]},
        timeout=3,
    )

    assert result["kind"] == "final"
    assert events == [
        ("request", {"model": "bonsai", "step": 2, "tools": [{"name": "lookup"}]}),
        ("response", {"kind": "final", "data": {"answer": "ok"}}),
    ]

    events.clear()
    result = bridge.request(
        {"operation": "python", "action": "evaluate_inline"},
        timeout=3,
    )
    assert result["kind"] == "final"
    assert events == []


def test_webgpu_telemetry_records_agent_model_and_tool_spans() -> None:
    class Span:
        def __init__(self, attributes: dict[str, Any]) -> None:
            self.attributes = dict(attributes)
            self.exceptions: list[BaseException] = []
            self.status: Any = None

        def set_attribute(self, key: str, value: Any) -> None:
            self.attributes[key] = value

        def set_attributes(self, attributes: dict[str, Any]) -> None:
            self.attributes.update(attributes)

        def record_exception(self, exception: BaseException, **_: Any) -> None:
            self.exceptions.append(exception)

        def set_status(self, status: Any) -> None:
            self.status = status

    class Tracer:
        def __init__(self) -> None:
            self.spans: list[tuple[str, dict[str, Any], Span]] = []

        @contextmanager
        def start_as_current_span(
            self,
            name: str,
            *,
            attributes: dict[str, Any],
            **kwargs: Any,
        ) -> Iterator[Span]:
            span = Span(attributes)
            self.spans.append((name, kwargs, span))
            yield span

    tracer = Tracer()
    telemetry = WebGPUTelemetry(
        tracer,  # type: ignore[arg-type]
        run_id="run-1",
        model="bonsai",
    )
    payload = {
        "model": "bonsai",
        "step": 1,
        "instructions": "Be exact.",
        "settings": {"temperature": 0.2, "max_tokens": 32},
        "outputSchema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
        "requiredTools": ["lookup"],
        "tools": [
            {
                "name": "lookup",
                "description": "Lookup a value",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ],
        "messages": [{"role": "user", "content": "Find Kedi"}],
    }
    browser_messages = [
        {"role": "system", "content": "Be exact.\nReturn JSON."},
        {"role": "user", "content": "Find Kedi"},
    ]

    with telemetry.run_span() as finish_run:
        with telemetry.model_request_span(payload) as finish_model:
            finish_model(
                {
                    "kind": "tool_call",
                    "name": "lookup",
                    "arguments": {"query": "Kedi"},
                    "callId": "call-1",
                    "telemetry": {
                        "inputMessages": browser_messages,
                        "outputText": '{"type":"tool_call"}',
                        "model": "bonsai-q1",
                        "finishReason": "tool_call",
                        "responseId": "response-1",
                        "usage": {"inputTokens": 12, "outputTokens": 4},
                    },
                }
            )
        with telemetry.tool_span("lookup", "call-1", {"query": "Kedi"}) as finish_tool:
            finish_tool({"title": "Kedi"})
        finish_run({"answer": "done"})

    assert [name for name, _, _ in tracer.spans] == [
        "invoke_agent kedi-webgpu",
        "chat bonsai",
        "execute_tool lookup",
    ]
    run_span = tracer.spans[0][2]
    model_span = tracer.spans[1][2]
    tool_span = tracer.spans[2][2]
    assert run_span.attributes["gen_ai.aggregated_usage.input_tokens"] == 12
    assert '"type": "tool_call"' in run_span.attributes["pydantic_ai.all_messages"]
    assert model_span.attributes["gen_ai.request.temperature"] == 0.2
    assert model_span.attributes["gen_ai.response.model"] == "bonsai-q1"
    assert model_span.attributes["gen_ai.response.finish_reasons"] == ["tool_call"]
    assert model_span.attributes["gen_ai.usage.output_tokens"] == 4
    assert '"name": "lookup"' in model_span.attributes["gen_ai.tool.definitions"]
    assert tool_span.attributes["gen_ai.tool.call.result"] == '{"title": "Kedi"}'


def test_provider_worker_runs_as_a_real_subprocess() -> None:
    result = _run_provider(
        source="= ok",
        model="test",
        secrets={},
    )

    assert result == {"ok": True, "result": "ok"}


def test_webgpu_worker_runs_as_a_real_subprocess_without_model_call() -> None:
    result = _run_webgpu(source="= ok", secrets={})

    assert result == {"ok": True, "result": "ok"}


def test_model_downloads_are_browser_owned() -> None:
    static = Path(server.__file__).parent / "static"
    app_source = (static / "app.js").read_text(encoding="utf-8")
    adapter_source = (static / "adapter-client.js").read_text(encoding="utf-8")
    index_source = (static / "index.html").read_text(encoding="utf-8")
    editor_source = (static / "kedi-editor.js").read_text(encoding="utf-8")
    highlighter_source = (static / "tree-sitter-highlighter.js").read_text(encoding="utf-8")
    styles_source = (static / "styles.css").read_text(encoding="utf-8")
    wllama_source = (static / "runtimes" / "wllama-runtime.js").read_text(encoding="utf-8")
    transformers_source = (static / "runtimes" / "transformers-runtime.js").read_text(
        encoding="utf-8"
    )
    route_paths = {getattr(route, "path", "") for route in server.app.routes}

    assert "/api/model/download" not in route_paths
    assert "/api/model/download/cancel" not in route_paths
    assert "/api/lsp/semantic-tokens" not in route_paths
    assert "/api/lsp/semantic-tokens/legend" not in route_paths
    assert not any(path.startswith("/models/") for path in route_paths)
    assert "/api/model/download" not in app_source
    assert "JSON.stringify(request.outputSchema)" not in adapter_source
    assert "formatOutputContract(request.outputSchema)" in adapter_source
    assert "formatToolCatalog(request.tools)" in adapter_source
    assert '<field name="${escapeXml(name)}" type="${escapeXml(schemaType(schema))}"' in (
        adapter_source
    )
    assert "`<${name}>${escapeXml(schemaType(schema))}</${name}>`" not in adapter_source
    assert "validateToolArguments(tool, toolCall.arguments)" in adapter_source
    assert "Return a corrected call_tool action using concrete values only." in adapter_source
    assert "loadModelFromUrl(modelUrl(config), params)" in wllama_source
    assert "useCache: true" in wllama_source
    assert "`model:${config.id}`" in wllama_source
    assert "env.useBrowserCache = true" in transformers_source
    assert "async isCached(config)" in transformers_source
    assert "async isCached(config)" in wllama_source
    assert "this.random.seed(Number(settings.seed))" in transformers_source
    assert "contextlib.redirect_stdout" in (static / "pyodide-worker.js").read_text(
        encoding="utf-8"
    )
    worker_source = (static / "pyodide-worker.js").read_text(encoding="utf-8")
    pyodide_runtime_source = (static / "pyodide-runtime.js").read_text(encoding="utf-8")
    assert "getattr(builtins, value.__name__, None) is value" in worker_source
    assert '"__name__": "__kedi_pyodide__"' in worker_source
    assert "pyodide.setStdin({ stdin: readStdin, isatty: true })" in worker_source
    assert 'pyodide.loadPackage(["micropip", "pydantic", "protobuf", "pygments"])' in worker_source
    assert '"opentelemetry-api==1.41.1"' in worker_source
    assert '"opentelemetry-semantic-conventions==0.62b1"' in worker_source
    assert '"logfire==4.33.0"' in worker_source
    assert "[name for name in namespace if not name.startswith" not in worker_source
    assert "new SharedArrayBuffer" in worker_source
    assert 'type: "stdin"' in worker_source
    assert "Atomics.wait(control, 0, 0)" in worker_source
    assert "respondToStdin(message.buffer, value)" in pyodide_runtime_source
    assert "Atomics.notify(control, 0)" in pyodide_runtime_source
    assert "pending.stdout += echo" in pyodide_runtime_source
    assert "stdout: pending.stdout" in pyodide_runtime_source
    assert "pyodide-worker.js?v=${Date.now()}" in pyodide_runtime_source
    assert 'event.data?.type === "ready"' in pyodide_runtime_source
    assert 'event.data?.type === "ready_error"' in pyodide_runtime_source
    assert "await this.preload()" in pyodide_runtime_source
    assert "this.executionQueue.then(() => this.#execute(request))" in pyodide_runtime_source
    assert 'self.postMessage({ type: "ready" })' in worker_source
    assert 'type: "ready_error"' in worker_source
    assert "const pythonRuntime = new PyodideRuntime(setStatus, browserIo())" in app_source
    assert "pythonRuntime.preload()" in app_source
    assert "browserIo(), pythonRuntime" in app_source
    assert "this.python.dispose()" not in adapter_source
    assert 'id="stdin-form"' in index_source
    assert 'id="stdin-input"' in index_source
    assert 'id="stdin-eof"' in index_source
    assert "onStdin: requestStdin" in app_source
    assert 'id="source" class="source-editor"' in index_source
    assert '<textarea id="source"' not in index_source
    assert "monaco-editor@0.55.1" in index_source
    assert "registerDocumentSemanticTokensProvider" in editor_source
    assert "registerHoverProvider" in editor_source
    assert "createKediTreeSitterHighlighter" in editor_source
    assert "SemanticTokensEdits" not in editor_source
    assert "tokenEdits" in highlighter_source
    assert "column: row === startPosition.row ? startPosition.column : 0" in highlighter_source
    assert (
        "column: row === endPosition.row ? endPosition.column : line.length" in highlighter_source
    )
    assert "const absoluteStart = mapPoint(localStart)" in highlighter_source
    assert "const absoluteEnd = mapPoint(localEnd)" in highlighter_source
    assert "startIndex: start" in highlighter_source
    assert "column: prefix.length - lineStart" in highlighter_source
    assert "utf8Length" not in highlighter_source
    assert "tree-sitter-kedi.wasm" in highlighter_source
    assert "tree-sitter-python.wasm" in highlighter_source
    assert "pythonInjectionSpans" in highlighter_source
    assert "normalizePythonRegion" in highlighter_source
    assert "PYTHON_PRIORITY_OFFSET" in highlighter_source
    assert (
        (static / "grammars" / "kedi" / "tree-sitter-kedi.wasm").read_bytes().startswith(b"\0asm")
    )
    assert (
        (static / "grammars" / "python" / "tree-sitter-python.wasm")
        .read_bytes()
        .startswith(b"\0asm")
    )
    assert (static / "vendor" / "web-tree-sitter.wasm").read_bytes().startswith(b"\0asm")
    highlight_query = (static / "grammars" / "kedi" / "highlights.scm").read_text(encoding="utf-8")
    injection_query = (static / "grammars" / "kedi" / "injections.scm").read_text(encoding="utf-8")
    assert "(procedure_def name: (identifier) @function)" in highlight_query
    assert "(template_block_stmt" in highlight_query
    assert "(python_code) @string.special" in highlight_query
    assert "(python_code) @injection.content" in injection_query
    assert "(python_inline_body) @injection.content" in injection_query
    assert "/api/lsp/semantic-tokens" not in editor_source
    assert "/api/lsp/hover" in editor_source
    assert "setMonarchTokensProvider" not in editor_source
    assert "setKediExecutionDiagnostic" in editor_source
    assert 'glyphMarginClassName: "kedi-runtime-error-glyph"' in editor_source
    assert 'setModelMarkers(model, "kedi-runtime"' in editor_source
    assert "clearKediExecutionDiagnostic(sourceEditor)" in app_source
    assert "error.diagnostic = payload.diagnostic ?? null" in adapter_source
    assert ".kedi-runtime-error-line" in styles_source
    assert ".kedi-runtime-error-glyph::before" in styles_source
    assert 'id="download-local"' in index_source
    assert 'data-state="checking"' in index_source
    assert 'name="model-source"' not in index_source
    assert "Browser cache" not in index_source
    assert '"Downloaded" : "Download"' in app_source
    assert 'class="file-input"' in index_source
    assert 'for="model-file" class="file-button"' in index_source
    assert '<button id="run" class="primary" type="button">Run</button>' in index_source
    assert "<span>WebGPU</span>" in index_source
    assert "<span>Custom BYOK models</span>" in index_source
    assert '<div id="file-picker" class="file-picker">' in index_source
    assert 'customGroup.label = "Custom WebGPU models"' in app_source
    assert 'kind: "byok"' in app_source
    assert 'kind: "webgpu"' in app_source
    assert "Add LOGFIRE_TOKEN before enabling Logfire." in app_source
    assert 'id="env-feedback"' in index_source
    assert 'data-control-tab="settings"' in index_source
    assert 'data-control-tab="models"' in index_source
    assert 'id="byok-custom-model"' in index_source
    assert 'id="byok-model-id"' in index_source
    assert 'id="browser-model-repo"' in index_source
    assert 'id="browser-model-file"' in index_source
    assert 'id="browser-model-dtype"' in index_source
    assert "/api/byok/models/validate" in route_paths
    assert "customByokModels" in app_source
    assert "customBrowserModels" in app_source
    assert "model_file_name" in transformers_source
    assert 'id="setting-temperature"' in index_source
    assert 'id="setting-max-tokens"' in index_source
    assert 'id="setting-top-p"' in index_source
    assert 'id="setting-seed"' in index_source
    assert app_source.count("settings: modelSettings()") == 2
    assert 'class="example-tabs"' in index_source
    assert index_source.count('data-example="') == 3
    assert "sourceEditor.setValue(source)" in app_source


def test_wllama_uses_the_fast_demo_generation_path() -> None:
    static = Path(server.__file__).parent / "static"
    runtime_source = (static / "runtimes" / "wllama-runtime.js").read_text(encoding="utf-8")
    adapter_source = (static / "adapter-client.js").read_text(encoding="utf-8")

    assert "const DEFAULT_CONTEXT_SIZE = 1024" in runtime_source
    assert "estimateMessagesTokens(messages) + maxTokens" in runtime_source
    assert "actualRequestTokens + maxTokens" in runtime_source
    assert "contextSoftLimitPercent" in runtime_source
    assert "ctx_shift: true" in runtime_source
    assert "positiveInteger(settings.max_tokens, 48)" in runtime_source
    assert "generation.seed = Number(settings.seed)" in runtime_source
    assert "createChatCompletion" in runtime_source
    assert 'responseFormat === "json_object"' in runtime_source
    assert "kedi:wllama-cache-complete" in runtime_source
    assert "if (request.tools.length)" in adapter_source
    assert "request.requiredTools ?? []" in adapter_source
    assert "Never simulate, predict, or invent a tool result." in adapter_source
    assert "Available tools are defined by this XML catalog:" in adapter_source
    assert "formatToolCatalog(request.tools)" in adapter_source
    assert "request.toolsPrompt" not in adapter_source
    assert "A failed tool call does not provide a value" in adapter_source
    assert "Tool arguments must contain concrete runtime values" in adapter_source
    assert '{"country":"Turkey"}' in adapter_source
    assert "JSON Schema" not in adapter_source
    assert "Never return call_tool and final in the same turn." in adapter_source
    assert "The host will execute it and send back a tool result." in adapter_source
    assert "const toolCall = parsedObjects.find" in adapter_source
    assert 'kind: "retry"' in adapter_source
    assert "You said ${mentionedTool.name} should be used but did not call it." in adapter_source
    assert "responseFormat:" in adapter_source
    assert "canonicalTemplateMessages(request.messages, outputKeys)" in adapter_source
    assert (
        "Fields to be substituted into the template string for final construction:"
        in adapter_source
    )
    assert "Complete the Kedi template accurately." not in adapter_source
    assert "Replace the bracketed output placeholder" not in adapter_source
    assert "Return exactly these keys" in adapter_source
    assert '"string", "integer", "value", and "example"' in adapter_source
    assert "Generate concrete semantic values that complete the template." in adapter_source


def test_byok_rejects_invalid_worker_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="not-json", stderr=""),
    )
    with pytest.raises(RuntimeError, match="Invalid playground worker response"):
        _run_provider(
            source="= ok",
            model="openai/gpt",
            secrets={"OPENAI_API_KEY": "token"},
        )
