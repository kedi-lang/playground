from __future__ import annotations

from typing import Any


def format_execution_output(stdout: str, result: Any) -> str:
    if result is None:
        return stdout
    separator = "\n" if stdout and not stdout.endswith("\n") else ""
    return f"{stdout}{separator}{result}"


__all__ = ["format_execution_output"]
