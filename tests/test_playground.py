from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from playground import byok, server
from playground.bridge import BridgeCancelled, BridgeManager, BridgeRun
from playground.execution import format_execution_output
from playground.models import browser_model, public_model_registry


def _run_byok(
    *,
    source: str,
    model: str,
    secrets: dict[str, object],
    settings: dict[str, object] | None = None,
) -> dict[str, Any]:
    return byok.run_byok(
        source=source,
        model=model,
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


def test_byok_validates_input_and_redacts_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="uppercase identifiers"):
        _run_byok(
            source="= ok",
            model="mistral:model",
            secrets={"bad-key": "secret"},
        )
    with pytest.raises(TypeError, match="values must be strings"):
        _run_byok(
            source="= ok",
            model="openai/gpt",
            secrets={"OPENAI_API_KEY": 1},
        )

    monkeypatch.setattr(
        byok.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="",
            stderr="provider rejected secret",
        ),
    )
    with pytest.raises(RuntimeError, match=r"provider rejected \[REDACTED\]"):
        _run_byok(
            source="= ok",
            model="openai/gpt",
            secrets={"OPENAI_API_KEY": "secret"},
        )


def test_byok_worker_contract_and_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def run(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(stdout=json.dumps({"ok": True, "result": "done"}), stderr="")

    monkeypatch.setattr(byok.subprocess, "run", run)
    result = _run_byok(
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
        "source": "= ok",
        "model": "groq/llama",
        "settings": {},
        "runId": "run",
        "bridgeUrl": "http://127.0.0.1/internal",
        "bridgeToken": "bridge-token",
    }

    monkeypatch.setattr(
        byok.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="[]", stderr=""),
    )
    with pytest.raises(TypeError, match="must be an object"):
        _run_byok(
            source="= ok",
            model="openai/gpt",
            secrets={"OPENAI_API_KEY": "token"},
        )


def test_byok_worker_runs_as_a_real_subprocess() -> None:
    result = _run_byok(
        source="= ok",
        model="test",
        secrets={},
    )

    assert result == {"ok": True, "result": "ok"}


def test_model_downloads_are_browser_owned() -> None:
    static = Path(server.__file__).parent / "static"
    app_source = (static / "app.js").read_text(encoding="utf-8")
    index_source = (static / "index.html").read_text(encoding="utf-8")
    wllama_source = (static / "runtimes" / "wllama-runtime.js").read_text(encoding="utf-8")
    transformers_source = (static / "runtimes" / "transformers-runtime.js").read_text(
        encoding="utf-8"
    )
    route_paths = {getattr(route, "path", "") for route in server.app.routes}

    assert "/api/model/download" not in route_paths
    assert "/api/model/download/cancel" not in route_paths
    assert not any(path.startswith("/models/") for path in route_paths)
    assert "/api/model/download" not in app_source
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


def test_wllama_uses_the_fast_demo_generation_path() -> None:
    static = Path(server.__file__).parent / "static"
    runtime_source = (static / "runtimes" / "wllama-runtime.js").read_text(encoding="utf-8")
    adapter_source = (static / "adapter-client.js").read_text(encoding="utf-8")

    assert "n_ctx: 512" in runtime_source
    assert "settings.max_tokens ?? 48" in runtime_source
    assert "generation.seed = Number(settings.seed)" in runtime_source
    assert "createChatCompletion" in runtime_source
    assert 'response_format: { type: "json_object" }' in runtime_source
    assert "if (request.tools.length)" in adapter_source
    assert "Return exactly these keys" in adapter_source


def test_byok_rejects_invalid_worker_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        byok.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="not-json", stderr=""),
    )
    with pytest.raises(RuntimeError, match="Invalid BYOK worker response"):
        _run_byok(
            source="= ok",
            model="openai/gpt",
            secrets={"OPENAI_API_KEY": "token"},
        )
