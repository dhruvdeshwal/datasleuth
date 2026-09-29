"""Validator + sandbox behaviour, including adversarial inputs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from datasleuth.config import Settings
from datasleuth.db.source import DataSource
from datasleuth.sandbox.executor import Sandbox
from datasleuth.sandbox.sql_executor import run_sqlite
from datasleuth.sandbox.validator import validate_pandas, validate_sql


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "d.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount REAL)")
        conn.executemany(
            "INSERT INTO sales (region, amount) VALUES (?, ?)",
            [("north", 10.0), ("south", 20.0), ("north", 5.5), ("east", None)],
        )
    return path


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="fake", sql_timeout_s=1.0, pandas_timeout_s=4.0, pandas_memory_mb=300
    )


@pytest.fixture
def sandbox(db: Path, settings: Settings) -> Sandbox:
    return Sandbox(DataSource(dialect="sqlite", db_path=db), settings)


# --------------------------------------------------------------- SQL validator
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT region, SUM(amount) FROM sales GROUP BY region",
        "WITH t AS (SELECT * FROM sales) SELECT COUNT(*) FROM t;",
        "SELECT * FROM sales UNION SELECT * FROM sales",
        "SELECT region FROM sales WHERE amount > (SELECT AVG(amount) FROM sales) -- ; DROP TABLE x",
        "SELECT 'DROP TABLE sales' AS s",
    ],
)
def test_valid_selects_pass(sql: str) -> None:
    assert validate_sql(sql).ok, validate_sql(sql).message


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "SELECT 1; DELETE FROM sales",
        "DELETE FROM sales",
        "INSERT INTO sales (region) VALUES ('x')",
        "UPDATE sales SET amount = 0",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "PRAGMA writable_schema = 1",
        "PRAGMA query_only = OFF",
        "SELECT load_extension('evil')",
        "SELECT readfile('/etc/passwd')",
        "WITH x AS (SELECT 1) DELETE FROM sales",
        "CREATE TABLE t AS SELECT * FROM sales",
        "ALTER TABLE sales ADD COLUMN x",
        "SELECT * FROM sales INTO OUTFILE '/tmp/x'",
        "SELECT randomblob(1000000000)",
        "VACUUM",
        "",
        "   ;  ",
    ],
)
def test_dangerous_sql_blocked(sql: str) -> None:
    result = validate_sql(sql)
    assert not result.ok and result.message


def test_syntax_error_is_classified() -> None:
    result = validate_sql("SELEC region FROM sales WHERE")
    assert not result.ok and result.error_type == "syntax_error"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM read_csv('/etc/passwd')",
    ],
)
def test_postgres_specific_blocks(sql: str) -> None:
    assert not validate_sql(sql, "postgres").ok


# ------------------------------------------------------------ Pandas validator
@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nresult = sales.groupby('region')['amount'].sum().reset_index()",
        "import numpy as np\nimport math\nresult = pd.DataFrame({'a': [math.sqrt(4)]})",
        "from math import floor\nresult = pd.DataFrame({'a': [floor(2.5)]})",
        "result = sales.assign(r=sales['amount'].rolling(2).mean())",
    ],
)
def test_valid_pandas_passes(code: str) -> None:
    assert validate_pandas(code).ok, validate_pandas(code).message


@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = os.listdir('.')",
        "import subprocess\nresult = 1",
        "import sys\nresult = sys.modules",
        "from os import path\nresult = 1",
        "result = __import__('os').system('id')",
        "result = open('/etc/passwd').read()",
        "result = eval('1+1')",
        "exec('x=1')\nresult = 1",
        "result = pd.read_csv('/etc/passwd')",
        "result = pd.read_pickle('x')",
        "sales.to_csv('/tmp/x.csv')\nresult = 1",
        "result = ().__class__.__bases__[0].__subclasses__()",
        "result = getattr(pd, 'read_csv')('/etc/passwd')",
        "result = np.load('/tmp/x.npy')",
        "result = pd.io.common.os.system('id')",
        "result = sales.query('amount > 1')",
        "result = '{0.__class__}'.format(1)",
        "class A: pass\nresult = 1",
        "result = compile('1', 'x', 'eval')",
        "result = globals()",
        "import numpy.ctypeslib\nresult = 1",
    ],
)
def test_dangerous_pandas_blocked(code: str) -> None:
    assert not validate_pandas(code).ok


def test_pandas_requires_result_variable() -> None:
    assert "result" in (validate_pandas("x = 1").message or "")


# ------------------------------------------------------------------ SQL sandbox
def test_sql_executes_and_returns_columns(sandbox: Sandbox) -> None:
    res = sandbox.execute(
        "sql", "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY 1"
    )
    assert res.ok and res.columns == ["region", "total"]
    assert res.rows == [["east", None], ["north", 15.5], ["south", 20.0]]


@pytest.mark.parametrize(
    ("sql", "kind"),
    [
        ("SELECT nope FROM sales", "unknown_column"),
        ("SELECT * FROM nope", "unknown_table"),
        ("SELECT id FROM sales a JOIN sales b ON a.id = b.id", "ambiguous_column"),
    ],
)
def test_sql_error_taxonomy(sandbox: Sandbox, sql: str, kind: str) -> None:
    res = sandbox.execute("sql", sql)
    assert not res.ok and res.error_type == kind


def test_connection_is_read_only_even_if_validator_bypassed(db: Path, settings: Settings) -> None:
    res = run_sqlite(db, "DROP TABLE sales", settings)
    assert not res.ok
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sales").fetchone() == (4,)


def test_infinite_sql_times_out(sandbox: Sandbox) -> None:
    res = sandbox.execute(
        "sql",
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c",
    )
    assert not res.ok and res.error_type == "timeout" and res.runtime_ms < 5000


def test_sql_memory_bomb_contained(sandbox: Sandbox) -> None:
    res = sandbox.execute_validated("sql", "SELECT printf('%1000000000d', 1)")
    assert res.runtime_ms < 2000
    assert not res.ok or res.rows == [[None]]  # SQLite refuses oversize strings (returns NULL)


def test_row_cap_and_truncation(db: Path) -> None:
    s = Settings(llm_provider="fake", row_cap=3)
    res = run_sqlite(db, "SELECT * FROM sales", s)
    assert res.ok and res.row_count == 3 and res.truncated


def test_cell_cap(db: Path) -> None:
    s = Settings(llm_provider="fake", cell_cap_chars=10)
    res = run_sqlite(db, "SELECT replace(hex(zeroblob(50)), '0', 'a')", s)
    assert res.ok and len(res.rows[0][0]) == 10


# ---------------------------------------------------------------- Pandas sandbox
def test_pandas_runs_with_tables_preloaded(sandbox: Sandbox) -> None:
    code = "result = sales.groupby('region', as_index=False)['amount'].sum().sort_values('region')"
    res = sandbox.execute("pandas", code)
    assert res.ok, res.error_message
    assert res.columns == ["region", "amount"]
    assert res.rows == [["east", 0.0], ["north", 15.5], ["south", 20.0]]


def test_pandas_series_and_scalar_results(sandbox: Sandbox) -> None:
    assert sandbox.execute("pandas", "result = sales['amount'].sum()").rows == [[35.5]]
    res = sandbox.execute("pandas", "result = sales.groupby('region')['amount'].count()")
    assert res.ok and res.columns == ["region", "amount"]


def test_pandas_runtime_error_is_reported(sandbox: Sandbox) -> None:
    res = sandbox.execute("pandas", "result = sales['missing_col']")
    assert not res.ok and res.error_type == "unknown_column"
    assert "line 1" in (res.error_message or "")


def test_pandas_infinite_loop_times_out(sandbox: Sandbox) -> None:
    res = sandbox.execute("pandas", "while True:\n    pass\nresult = 1")
    assert not res.ok and res.error_type == "timeout"


def test_pandas_memory_bomb_killed(sandbox: Sandbox) -> None:
    code = "x = []\nwhile True:\n    x.append('a' * 10**7 + str(len(x)))\nresult = 1"
    res = sandbox.execute("pandas", code)
    assert not res.ok and (
        "memory" in (res.error_message or "").lower() or res.error_type == "timeout"
    )


def test_worker_contains_unvalidated_escape_attempts(sandbox: Sandbox) -> None:
    """Defence in depth: even if the validator were bypassed, the worker's builtins hold."""
    for code in (
        "result = open('/etc/passwd').read()",
        "result = __import__('os').listdir('.')",
        "import os\nresult = 1",
        "import socket\nresult = 1",
        "result = eval('1')",
    ):
        res = sandbox.execute_validated("pandas", code)
        assert not res.ok, code


def test_pandas_result_not_assigned_reported(sandbox: Sandbox) -> None:
    res = sandbox.execute_validated("pandas", "x = 1")
    assert not res.ok


def test_pandas_output_is_capped(db: Path) -> None:
    s = Settings(llm_provider="fake", row_cap=2)
    res = Sandbox(DataSource(dialect="sqlite", db_path=db), s).execute("pandas", "result = sales")
    assert res.ok and res.row_count == 2 and res.truncated


def test_secrets_not_visible_to_worker(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    res = sandbox.execute_validated("pandas", "result = pd.DataFrame({'k': [str(sorted(dfs))]})")
    assert res.ok and "super-secret" not in str(res.rows)


def test_docker_command_is_locked_down(db: Path, settings: Settings) -> None:
    from datasleuth.sandbox.pandas_runner import docker_command

    cmd = " ".join(docker_command(db, settings, "n"))
    for flag in (
        "--network none",
        "--read-only",
        "--user 65534:65534",
        "--cap-drop ALL",
        ":/data:ro",
    ):
        assert flag in cmd
