from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("byok_worker.py")

ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def run_byok(
    *,
    source: str,
    model: str,
    secrets: Mapping[str, object],
    settings: Mapping[str, object],
    run_id: str,
    bridge_url: str,
    bridge_token: str,
    timeout: float = 300,
) -> dict[str, Any]:
    if any(ENV_NAME_PATTERN.fullmatch(name) is None for name in secrets):
        raise ValueError("Environment names must be uppercase identifiers")
    if any(not isinstance(value, str) for value in secrets.values()):
        raise TypeError("Environment values must be strings")

    env = _worker_environment(secrets=secrets)
    payload = json.dumps(
        {
            "source": source,
            "model": model,
            "settings": dict(settings),
            "runId": run_id,
            "bridgeUrl": bridge_url,
            "bridgeToken": bridge_token,
        }
    )
    completed = subprocess.run(
        [sys.executable, str(WORKER)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout.strip()
    if not output:
        message = completed.stderr.strip() or "BYOK worker returned no output"
        raise RuntimeError(_redact(message, secrets))
    try:
        result = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(_redact(f"Invalid BYOK worker response: {output}", secrets)) from exc
    if not isinstance(result, dict):
        raise TypeError("BYOK worker response must be an object")
    return result


def _worker_environment(
    *,
    secrets: Mapping[str, object],
) -> dict[str, str]:
    kept = {
        name: os.environ[name]
        for name in (
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "PYTHONPATH",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TMPDIR",
        )
        if name in os.environ
    }
    kept["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT / "src"), kept.get("PYTHONPATH", "")) if value
    )
    for name, value in secrets.items():
        if isinstance(value, str) and value:
            kept[name] = value
    return kept


def _redact(message: str, secrets: Mapping[str, object]) -> str:
    for value in secrets.values():
        if isinstance(value, str) and value:
            message = message.replace(value, "[REDACTED]")
    return message


__all__ = [
    "ENV_NAME_PATTERN",
    "run_byok",
]
