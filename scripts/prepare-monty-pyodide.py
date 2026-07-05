from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from io import BytesIO
from pathlib import Path

ASYNC_RUNTIME_VERSION = "0.28.0"
ASYNC_RUNTIME_SHA256 = "9e7364a95bf00e8377bbf9b0f09d7ff9715a29d8fcf93b47d1a967363b973178"
ASYNC_RUNTIME_URL = (
    f"https://crates.io/api/v1/crates/pyo3-async-runtimes/{ASYNC_RUNTIME_VERSION}/download"
)


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one matching Pyodide patch target in {path}")
    path.write_text(source.replace(old, new), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare-monty-pyodide.py PATH_TO_MONTY")

    monty_root = Path(sys.argv[1]).resolve()
    if not (monty_root / "crates/monty-python").is_dir():
        raise RuntimeError(f"Monty checkout not found at {monty_root}")

    with urllib.request.urlopen(ASYNC_RUNTIME_URL) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != ASYNC_RUNTIME_SHA256:
        raise RuntimeError("pyo3-async-runtimes source checksum mismatch")

    vendor_root = monty_root / "vendor"
    target = vendor_root / "pyo3-async-runtimes"
    shutil.rmtree(target, ignore_errors=True)
    vendor_root.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as package:
            package.extractall(temp_dir, filter="data")
        shutil.move(
            Path(temp_dir) / f"pyo3-async-runtimes-{ASYNC_RUNTIME_VERSION}",
            target,
        )

    replace_once(
        target / "Cargo.toml",
        'features = [\n    "rt",\n    "rt-multi-thread",\n    "time",\n]',
        'features = [\n    "rt",\n    "time",\n]',
    )
    replace_once(
        target / "Cargo.toml.orig",
        'features = ["rt", "rt-multi-thread", "time"]',
        'features = ["rt", "time"]',
    )
    replace_once(
        target / "src/tokio.rs",
        "Builder::new_multi_thread()",
        "Builder::new_current_thread()",
    )

    workspace = monty_root / "Cargo.toml"
    source = workspace.read_text(encoding="utf-8")
    if "[patch.crates-io]" in source:
        raise RuntimeError("Monty already defines a crates.io patch section")
    workspace.write_text(
        source
        + '\n[patch.crates-io]\npyo3-async-runtimes = { path = "vendor/pyo3-async-runtimes" }\n',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
