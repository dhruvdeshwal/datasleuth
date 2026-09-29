"""Read-only SQL execution with timeouts and row/cell caps."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from datasleuth.config import Settings
from datasleuth.db.connectors import open_sqlite_ro
from datasleuth.sandbox.results import classify_sql_error, jsonable
from datasleuth.state import Dialect, ExecutionResult

_PROGRESS_OPS = 10_000


def run_sqlite(db_path: Path, sql: str, settings: Settings) -> ExecutionResult:
    """Execute validated SQL on a read-only SQLite connection."""
    started = time.perf_counter()
    deadline = started + settings.sql_timeout_s
    conn: sqlite3.Connection | None = None
    try:
        conn = open_sqlite_ro(db_path)
        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, _PROGRESS_OPS)
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description or []]
        fetched = cur.fetchmany(settings.row_cap + 1)
        return _build(columns, fetched, started, settings)
    except sqlite3.Error as exc:
        return _error(str(exc), started)
    finally:
        if conn is not None:
            conn.close()


def run_engine(engine: Engine, dialect: Dialect, sql: str, settings: Settings) -> ExecutionResult:
    """Execute validated SQL on an external database via SQLAlchemy (read-only session)."""
    started = time.perf_counter()
    try:
        with engine.connect() as conn:
            ms = int(settings.sql_timeout_s * 1000)
            if dialect == "postgres":
                conn.exec_driver_sql(f"SET LOCAL statement_timeout = {ms}")
            elif dialect == "mysql":
                conn.exec_driver_sql(f"SET SESSION MAX_EXECUTION_TIME = {ms}")
            result = conn.execute(text(sql))
            columns = list(result.keys())
            fetched = result.fetchmany(settings.row_cap + 1)
            return _build(columns, [tuple(r) for r in fetched], started, settings)
    except DBAPIError as exc:
        return _error(str(exc.orig or exc), started)


def _build(
    columns: list[str], fetched: list[tuple[object, ...]], started: float, settings: Settings
) -> ExecutionResult:
    truncated = len(fetched) > settings.row_cap
    rows = [
        [jsonable(v, settings.cell_cap_chars) for v in row] for row in fetched[: settings.row_cap]
    ]
    return ExecutionResult(
        ok=True,
        rows=rows,
        columns=columns,
        row_count=len(rows),
        truncated=truncated,
        runtime_ms=(time.perf_counter() - started) * 1000,
    )


def _error(message: str, started: float) -> ExecutionResult:
    message = message.strip().splitlines()[0][:500] if message.strip() else "unknown error"
    return ExecutionResult(
        ok=False,
        error_type=classify_sql_error(message),
        error_message=message,
        runtime_ms=(time.perf_counter() - started) * 1000,
    )
