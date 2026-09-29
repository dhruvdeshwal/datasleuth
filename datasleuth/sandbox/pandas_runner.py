"""Run Pandas code in an isolated subprocess (default) or a Docker container."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import psutil

from datasleuth.config import Settings
from datasleuth.state import ExecutionResult

WORKER = Path(__file__).with_name("pandas_worker.py")
_STDERR_TAIL = 500


def _payload(
    db_path: str, code: str, tables: list[str] | None, settings: Settings, immutable: bool
) -> str:
    return json.dumps(
        {
            "db_path": db_path,
            "code": code,
            "tables": tables,
            "row_cap": settings.row_cap,
            "cell_cap": settings.cell_cap_chars,
            "memory_mb": settings.pandas_memory_mb,
            "cpu_s": settings.pandas_cpu_s,
            "immutable": immutable,
        }
    )


def _parse(stdout: str, stderr: str, started: float) -> ExecutionResult:
    runtime = (time.perf_counter() - started) * 1000
    try:
        data = json.loads(stdout)
    except ValueError:
        tail = stderr.strip()[-_STDERR_TAIL:] or "no output"
        return ExecutionResult(
            ok=False, error_type="runtime_error",
            error_message=f"sandbox process failed: {tail}", runtime_ms=runtime,
        )  # fmt: skip
    rows = data.get("rows", [])
    return ExecutionResult(
        ok=bool(data.get("ok")),
        rows=rows,
        columns=data.get("columns", []),
        row_count=len(rows),
        truncated=bool(data.get("truncated")),
        error_type=data.get("error_type"),
        error_message=data.get("error_message"),
        runtime_ms=runtime,
    )


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def run_pandas_subprocess(
    db_path: Path, code: str, tables: list[str] | None, settings: Settings
) -> ExecutionResult:
    """Run code in a fresh interpreter: empty cwd, scrubbed env, RSS + wall-clock watchdog.

    POSIX additionally gets ``setrlimit`` (address space, CPU) applied inside the worker.
    """
    started = time.perf_counter()
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    for key in ("SYSTEMROOT", "PATH", "TEMP", "TMP"):  # required by Windows/DLL loading
        if key in os.environ:
            env[key] = os.environ[key]
    with tempfile.TemporaryDirectory(prefix="ds_sbx_") as cwd:
        proc = subprocess.Popen(
            [sys.executable, "-I", str(WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", cwd=cwd, env=env,
        )  # fmt: skip
        verdict: dict[str, str] = {}
        done = threading.Event()

        def watchdog() -> None:
            limit = settings.pandas_memory_mb * 1024 * 1024
            while not done.wait(0.05):
                try:
                    ps = psutil.Process(proc.pid)
                    rss = ps.memory_info().rss + sum(
                        c.memory_info().rss for c in ps.children(recursive=True)
                    )
                except psutil.Error:
                    return
                if rss > limit:
                    verdict["memory"] = "1"
                    _kill_tree(proc)
                    return

        threading.Thread(target=watchdog, daemon=True).start()
        try:
            stdout, stderr = proc.communicate(
                _payload(str(db_path.resolve()), code, tables, settings, False),
                timeout=settings.pandas_timeout_s,
            )
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.communicate()
            return ExecutionResult(
                ok=False, error_type="timeout",
                error_message=f"Pandas code exceeded {settings.pandas_timeout_s:.0f}s",
                runtime_ms=(time.perf_counter() - started) * 1000,
            )  # fmt: skip
        finally:
            done.set()
    if verdict.get("memory"):
        return ExecutionResult(
            ok=False, error_type="runtime_error",
            error_message=f"Pandas code exceeded the {settings.pandas_memory_mb} MB memory limit",
            runtime_ms=(time.perf_counter() - started) * 1000,
        )  # fmt: skip
    return _parse(stdout, stderr, started)


def docker_command(db_path: Path, settings: Settings, name: str) -> list[str]:
    """Build the locked-down ``docker run`` command (no network, read-only FS, non-root)."""
    mem = f"{settings.pandas_memory_mb}m"
    return [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none", "--read-only", "--user", "65534:65534",
        "--memory", mem, "--memory-swap", mem, "--cpus", "1", "--pids-limit", "64",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "-v", f"{db_path.resolve().parent.as_posix()}:/data:ro",
        settings.docker_image, "python", "-I", "/app/pandas_worker.py",
    ]  # fmt: skip


def run_pandas_docker(
    db_path: Path, code: str, tables: list[str] | None, settings: Settings
) -> ExecutionResult:
    """Run code in the sandbox container (stronger isolation than the subprocess runner)."""
    started = time.perf_counter()
    name = f"datasleuth-{uuid.uuid4().hex[:12]}"
    cmd = docker_command(db_path, settings, name)
    payload = _payload(f"/data/{db_path.name}", code, tables, settings, True)
    try:
        proc = subprocess.run(
            cmd, input=payload, capture_output=True, text=True, encoding="utf-8",
            timeout=settings.pandas_timeout_s + 5,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True, check=False)
        return ExecutionResult(
            ok=False, error_type="timeout",
            error_message=f"Pandas code exceeded {settings.pandas_timeout_s:.0f}s",
            runtime_ms=(time.perf_counter() - started) * 1000,
        )  # fmt: skip
    except FileNotFoundError:
        return ExecutionResult(
            ok=False, error_type="runtime_error",
            error_message="Docker is not installed or not on PATH.",
        )  # fmt: skip
    if proc.returncode == 137:
        return ExecutionResult(
            ok=False, error_type="runtime_error",
            error_message=f"Pandas code exceeded the {settings.pandas_memory_mb} MB memory limit",
            runtime_ms=(time.perf_counter() - started) * 1000,
        )  # fmt: skip
    return _parse(proc.stdout, proc.stderr, started)
