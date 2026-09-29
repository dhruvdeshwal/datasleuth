"""Database connections: read-only SQLite plus external SQLAlchemy databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import Engine, create_engine, event

from datasleuth.state import Dialect

_ALLOWED_DRIVERS = {
    "sqlite": "sqlite",
    "postgresql": "postgres",
    "postgres": "postgres",
    "mysql": "mysql",
}


class ConnectionRejected(ValueError):
    """The requested connection is unsupported or unsafe."""


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def open_sqlite_ro(path: Path | str) -> sqlite3.Connection:
    """Open a SQLite file strictly read-only (``mode=ro`` + ``query_only``)."""
    uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.text_factory = _decode
    conn.execute("PRAGMA query_only = ON")
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 10_000_000)  # blunt string/blob memory bombs
    return conn


def sqlite_engine(path: Path | str) -> Engine:
    """SQLAlchemy engine over a read-only SQLite connection."""
    return create_engine("sqlite://", creator=lambda: open_sqlite_ro(path))


def dialect_of_url(url: str) -> Dialect:
    """Map a SQLAlchemy URL to a supported dialect or raise ConnectionRejected."""
    scheme = urlparse(url).scheme.split("+")[0].lower()
    if scheme not in _ALLOWED_DRIVERS:
        raise ConnectionRejected(
            f"Unsupported database '{scheme}'. Supported: SQLite, PostgreSQL, MySQL."
        )
    return _ALLOWED_DRIVERS[scheme]  # type: ignore[return-value]


def external_engine(url: str) -> tuple[Engine, Dialect]:
    """Create an engine for an external database and force read-only sessions.

    This is defence in depth: the credentials you provide should *also* be
    read-only at the database level, and the SQL validator only allows SELECTs.
    """
    dialect = dialect_of_url(url)
    if dialect == "sqlite":
        path = urlparse(url).path.lstrip("/") if not urlparse(url).netloc else urlparse(url).path
        if not Path(path).exists():
            raise ConnectionRejected("SQLite database file not found.")
        return sqlite_engine(path), dialect
    engine = create_engine(url, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _read_only(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
        cur = dbapi_conn.cursor()
        try:
            if dialect == "postgres":
                cur.execute("SET default_transaction_read_only = on")
            else:
                cur.execute("SET SESSION TRANSACTION READ ONLY")
        finally:
            cur.close()

    return engine, dialect
