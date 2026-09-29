"""Opt-in tests: ``pytest -m live`` (needs a real API key) and ``pytest -m docker`` (needs Docker)."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from datasleuth.config import Settings
from datasleuth.db.source import DataSource
from datasleuth.graph import DataSleuthAgent
from datasleuth.llm.client import LLM
from datasleuth.sandbox.executor import Sandbox

IMAGE = "datasleuth-sandbox:test"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


@pytest.fixture(scope="module")
def sandbox_image() -> str:
    if not _docker_ready():
        pytest.skip("Docker engine is not running")
    build = subprocess.run(
        ["docker", "build", "-q", "-f", "Dockerfile.sandbox", "-t", IMAGE, "."],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    return IMAGE


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "d.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sales (region TEXT, amount REAL)")
        conn.executemany("INSERT INTO sales VALUES (?, ?)", [("n", 10.0), ("s", 20.0), ("n", 5.0)])
    return path


@pytest.mark.docker
def test_docker_executor_runs_pandas(sandbox_image: str, db: Path) -> None:
    s = Settings(
        llm_provider="fake", sandbox="docker", docker_image=sandbox_image, pandas_timeout_s=60
    )
    res = Sandbox(DataSource(dialect="sqlite", db_path=db), s).execute(
        "pandas",
        "result = sales.groupby('region', as_index=False).amount.sum().sort_values('region')",
    )
    assert res.ok, res.error_message
    assert res.rows == [["n", 15.0], ["s", 20.0]]


@pytest.mark.docker
def test_docker_executor_contains_escape_attempts(sandbox_image: str, db: Path) -> None:
    s = Settings(
        llm_provider="fake",
        sandbox="docker",
        docker_image=sandbox_image,
        pandas_timeout_s=30,
        pandas_memory_mb=200,
    )
    sb = Sandbox(DataSource(dialect="sqlite", db_path=db), s)
    assert not sb.execute("pandas", "x = []\nwhile True:\n    x.append('a' * 10**7)\nresult = 1").ok
    assert sb.execute("pandas", "while True:\n    pass\nresult = 1").error_type in {
        "timeout",
        "runtime_error",
    }


@pytest.mark.docker
def test_container_flags_really_isolate(sandbox_image: str, tmp_path: Path) -> None:
    """The exact flags used by the executor: no network, read-only FS, non-root."""
    base = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--entrypoint",
        "python",
        sandbox_image,
        "-c",
    ]

    def run(code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([*base, code], capture_output=True, text=True, timeout=60)

    assert run("import socket; socket.create_connection(('1.1.1.1', 53), 2)").returncode != 0
    assert run("open('/etc/evil', 'w')").returncode != 0
    assert run("import os; print(os.getuid())").stdout.strip() == "65534"


@pytest.mark.live
def test_live_gemini_end_to_end(tmp_path: Path) -> None:
    if not Settings().gemini_api_key:
        pytest.skip("GEMINI_API_KEY not set")
    db = tmp_path / "d.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sales (region TEXT, amount REAL)")
        conn.executemany(
            "INSERT INTO sales VALUES (?, ?)", [("north", 10.0), ("south", 20.0), ("north", 5.5)]
        )
    s = Settings(
        cache_dir=tmp_path / "c", log_dir=tmp_path / "l", use_chart=False, llm_daily_limit=30
    )
    report = DataSleuthAgent(DataSource(dialect="sqlite", db_path=db), s, llm=LLM(s)).run(
        "What is the total amount for the north region?"
    )
    assert report.status == "success" and report.rows == [[15.5]]
