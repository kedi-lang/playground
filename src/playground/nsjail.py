from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from time import monotonic as _monotonic
from typing import Any

from playground import sandbox_worker

PACKAGE_ROOT = Path(__file__).resolve().parent
PLAYGROUND_ROOT = PACKAGE_ROOT.parents[1]
RESPONSE_PREFIX = sandbox_worker._RESPONSE_PREFIX

DEFAULT_POOL_SIZE = 10
DEFAULT_TIMEOUT = 60.0
DEFAULT_RLIMIT_AS_MB = "2048"
_NOBODY_MAP = "65534:65534:1"
_DEFAULT_CHROOT = "/tmp/kedi-nsjail-root"
_NETWORK_DENY_SECCOMP = (
    "ERRNO(1) { "
    "socket, socketpair, connect, accept, accept4, bind, listen, "
    "getsockname, getpeername, sendto, recvfrom, sendmsg, recvmsg, shutdown "
    "} DEFAULT ALLOW"
)
_MAX_NOISE_LINES = 3
_NAMESPACE_DISABLE_FLAGS = [
    "--disable_clone_newnet",
    "--disable_clone_newuser",
    "--disable_clone_newns",
    "--disable_clone_newpid",
    "--disable_clone_newipc",
    "--disable_clone_newuts",
    "--disable_clone_newcgroup",
]


class NsJailUnavailable(RuntimeError):
    pass


class NsJailWorker:
    def __init__(self, command: list[str], *, env: Mapping[str, str]) -> None:
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=dict(env),
        )
        self._lock = threading.Lock()
        self._closed = False

    def request(
        self,
        payload: Mapping[str, Any],
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("NsJail worker is closed")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("NsJail worker pipes are not available")
        if self._process.poll() is not None:
            message = self._worker_failure_message("NsJail worker exited before request")
            self.close()
            raise RuntimeError(message)
        with self._lock:
            try:
                self._process.stdin.write(json.dumps(dict(payload), separators=(",", ":")) + "\n")
                self._process.stdin.flush()
            except BrokenPipeError as exc:
                message = self._worker_failure_message("NsJail worker pipe closed")
                self.close()
                raise RuntimeError(message) from exc
            return self._read_response(timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._process.terminate()
        try:
            self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=1)

    def _read_stderr(self) -> str:
        stream = self._process.stderr
        if stream is None:
            return ""
        if self._process.poll() is None:
            return ""
        lines: list[str] = []
        try:
            while True:
                ready, _, _ = select.select([stream], [], [], 0)
                if not ready:
                    break
                line = stream.readline()
                if not line:
                    break
                lines.append(line.rstrip())
        except (OSError, TypeError, ValueError):
            text = stream.read()
            return text.strip() if text else ""
        return "\n".join(lines).strip()

    def _read_response(self, timeout: float) -> Mapping[str, Any]:
        assert self._process.stdout is not None
        deadline = timeout + _monotonic()
        noise: list[str] = []
        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                self.close()
                suffix = _protocol_noise_suffix(noise)
                raise TimeoutError(f"Timed out waiting for NsJail worker{suffix}")
            ready, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not ready:
                self.close()
                suffix = _protocol_noise_suffix(noise)
                raise TimeoutError(f"Timed out waiting for NsJail worker{suffix}")
            line = self._process.stdout.readline()
            if not line:
                stderr = self._read_stderr()
                self.close()
                suffix = _protocol_noise_suffix(noise)
                raise RuntimeError(stderr or f"NsJail worker exited without a response{suffix}")
            if not line.startswith(RESPONSE_PREFIX):
                noise.append(line.rstrip())
                continue
            response = json.loads(line.removeprefix(RESPONSE_PREFIX))
            if not isinstance(response, dict):
                self.close()
                raise RuntimeError(
                    f"NsJail worker returned a malformed response{_protocol_noise_suffix(noise)}"
                )
            return response

    def _worker_failure_message(self, fallback: str) -> str:
        stderr = self._read_stderr()
        if stderr:
            return stderr
        return_code = self._process.poll()
        if return_code is None:
            return fallback
        return f"{fallback} with exit code {return_code}"


class NsJailPool:
    def __init__(
        self,
        *,
        size: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.size = DEFAULT_POOL_SIZE if size is None else size
        self.timeout = timeout
        self._workers: list[NsJailWorker] = []
        self._lock = threading.Lock()
        self._enabled = False
        self._closed = False
        self._last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if self.size < 1:
            self._last_error = "NsJail pool size is less than 1"
            return
        if os.environ.get("KEDI_NSJAIL_ENABLED", "1").lower() in {"0", "false", "off", "no"}:
            self._last_error = "NsJail disabled by KEDI_NSJAIL_ENABLED"
            return
        try:
            worker = self._new_checked_worker()
        except Exception as exc:  # noqa: BLE001 - pool availability is best-effort.
            self._last_error = str(exc)
            return
        with self._lock:
            self._workers.append(worker)
            self._enabled = True
        for _ in range(self.size - 1):
            self._spawn_replacement()

    @contextmanager
    def lease(self) -> Iterator[NsJailWorker | None]:
        worker = self._take_worker()
        try:
            yield worker
        finally:
            if worker is not None:
                worker.close()
                self._spawn_replacement()

    def close(self) -> None:
        self._closed = True
        with self._lock:
            workers = self._workers
            self._workers = []
        for worker in workers:
            worker.close()

    def _take_worker(self) -> NsJailWorker | None:
        if not self._enabled:
            return None
        with self._lock:
            return self._workers.pop() if self._workers else None

    def _spawn_replacement(self) -> None:
        if self._closed or not self._enabled:
            return

        def target() -> None:
            try:
                worker = self._new_checked_worker()
            except Exception as exc:  # noqa: BLE001 - keep the pool alive with fewer workers.
                self._last_error = str(exc)
                return
            with self._lock:
                if self._closed:
                    worker.close()
                elif len(self._workers) < self.size:
                    self._workers.append(worker)
                else:
                    worker.close()

        threading.Thread(target=target, daemon=True).start()

    def _new_worker(self) -> NsJailWorker:
        env = worker_environment()
        command = nsjail_command(env)
        return NsJailWorker(command, env=env)

    def _new_checked_worker(self) -> NsJailWorker:
        worker = self._new_worker()
        try:
            self._self_test(worker)
        except Exception:
            worker.close()
            raise
        return worker

    def _self_test(self, worker: NsJailWorker) -> None:
        response = worker.request(
            {
                "operation": "python",
                "action": "evaluate_inline",
                "code": "1 + 1",
                "env": {},
                "syncNames": [],
                "kediLineOffset": 0,
            },
            timeout=min(self.timeout, 10),
        )
        if response.get("ok") is not True or response.get("result") != 2:
            raise NsJailUnavailable(f"NsJail self-test failed: {response!r}")


def nsjail_command(env: Mapping[str, str] | None = None) -> list[str]:
    executable = os.environ.get("KEDI_NSJAIL_BIN") or shutil.which("nsjail")
    if not executable:
        raise NsJailUnavailable("nsjail executable was not found")
    child_env = worker_environment() if env is None else dict(env)

    command = [
        executable,
        "-Mo",
        "--quiet",
        "--chroot",
        sandbox_root(),
        "--cwd",
        "/tmp",
        "--disable_proc",
        "--user",
        _NOBODY_MAP,
        "--group",
        _NOBODY_MAP,
        *_env_args(child_env),
        "--time_limit",
        os.environ.get("KEDI_NSJAIL_TIME_LIMIT", "300"),
        "--rlimit_as",
        os.environ.get("KEDI_NSJAIL_RLIMIT_AS", DEFAULT_RLIMIT_AS_MB),
        "--rlimit_cpu",
        os.environ.get("KEDI_NSJAIL_RLIMIT_CPU", "60"),
        "--rlimit_fsize",
        os.environ.get("KEDI_NSJAIL_RLIMIT_FSIZE", "16"),
        "--rlimit_nofile",
        os.environ.get("KEDI_NSJAIL_RLIMIT_NOFILE", "128"),
    ]
    if static_chroot_mode():
        command.extend(_NAMESPACE_DISABLE_FLAGS)
        command.extend(["--seccomp_string", _NETWORK_DENY_SECCOMP])
    else:
        command.extend(["--disable_clone_newcgroup", "--tmpfsmount", "/tmp"])
        for path in readonly_mounts():
            command.extend(["-R", path])
    command.extend(["--", sys.executable, str(Path(sandbox_worker.__file__).resolve())])
    return command


def _env_args(env: Mapping[str, str]) -> list[str]:
    args: list[str] = []
    for name, value in sorted(env.items()):
        args.extend(["--env", f"{name}={value}"])
    return args


def _protocol_noise_suffix(noise: list[str]) -> str:
    if not noise:
        return ""
    preview = "; ".join(repr(line[:200]) for line in noise[-_MAX_NOISE_LINES:])
    hidden = len(noise) - _MAX_NOISE_LINES
    if hidden > 0:
        return f" after ignoring {hidden} earlier protocol noise line(s): {preview}"
    return f" after protocol noise: {preview}"


def sandbox_root() -> str:
    root = Path(os.environ.get("KEDI_NSJAIL_CHROOT", _DEFAULT_CHROOT))
    if static_chroot_mode():
        if not root.is_dir():
            raise NsJailUnavailable(f"Static NsJail root does not exist: {root}")
        return str(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return str(root)


def static_chroot_mode() -> bool:
    return os.environ.get("KEDI_NSJAIL_STATIC_CHROOT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def readonly_mounts() -> list[str]:
    candidates = [
        "/usr/local",
        "/usr/lib",
        "/lib",
        "/lib64",
        "/opt/venv",
        str(PLAYGROUND_ROOT / "src"),
        "/dev/urandom",
        "/dev/null",
    ]
    return [path for path in candidates if Path(path).exists()]


def worker_environment() -> dict[str, str]:
    env: dict[str, str] = {
        "HOME": "/tmp",
        "LD_LIBRARY_PATH": "/usr/local/lib:/usr/lib:/lib",
    }
    for name in ("LANG", "LC_ALL", "PATH", "PYTHONPATH", "SSL_CERT_DIR", "SSL_CERT_FILE"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(PLAYGROUND_ROOT / "src"),
            os.environ.get("PYTHONPATH", ""),
        )
        if value
    )
    return env


__all__ = [
    "NsJailPool",
    "NsJailUnavailable",
    "NsJailWorker",
    "nsjail_command",
    "readonly_mounts",
    "static_chroot_mode",
    "worker_environment",
]
