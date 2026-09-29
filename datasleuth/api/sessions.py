"""Session management: per-session isolated data and TTL cleanup.

A session is a directory ``<data_dir>/sessions/<id>/`` holding its own SQLite database,
profiled schema and traces. Session ids are 128-bit random hex and are validated before
touching the filesystem, so they cannot be used for path traversal.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine

from datasleuth.config import Settings
from datasleuth.db.connectors import external_engine
from datasleuth.db.ingest import ingest_files
from datasleuth.db.profile import profile_schema
from datasleuth.db.source import DataSource
from datasleuth.state import Dialect, SchemaInfo

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{16,32}$")


class SessionNotFound(KeyError):
    """Unknown, malformed or expired session id."""


@dataclass
class Session:
    """A live session's non-secret handle."""

    id: str
    directory: Path
    created_at: float
    last_access: float

    @property
    def db_path(self) -> Path:
        return self.directory / "data.sqlite"

    @property
    def traces_dir(self) -> Path:
        return self.directory / "traces"


class SessionStore:
    """Creates, looks up, expires and deletes sessions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.data_dir / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # External connections live only in memory (URL may contain credentials).
        self._engines: dict[str, tuple[Engine, Dialect, SchemaInfo]] = {}

    # ---------------------------------------------------------------- lookup
    def _dir(self, session_id: str) -> Path:
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise SessionNotFound(session_id)
        return self.root / session_id

    def _meta_path(self, session_id: str) -> Path:
        return self._dir(session_id) / "meta.json"

    def create(self) -> Session:
        """Create an empty session."""
        sid = uuid.uuid4().hex
        directory = self._dir(sid)
        directory.mkdir(parents=True)
        now = time.time()
        self._write_meta(sid, now, now)
        return Session(sid, directory, now, now)

    def _write_meta(self, sid: str, created: float, last: float) -> None:
        self._meta_path(sid).write_text(
            json.dumps({"created_at": created, "last_access": last}), encoding="utf-8"
        )

    def get(self, session_id: str, *, touch: bool = True) -> Session:
        """Return a live session or raise SessionNotFound (expired sessions are deleted)."""
        meta_path = self._meta_path(session_id)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SessionNotFound(session_id) from None
        if time.time() - meta["last_access"] > self.settings.session_ttl_min * 60:
            self.delete(session_id)
            raise SessionNotFound(session_id)
        now = time.time()
        if touch:
            self._write_meta(session_id, meta["created_at"], now)
        return Session(session_id, self._dir(session_id), meta["created_at"], now)

    def delete(self, session_id: str) -> None:
        """Remove a session and all of its data (idempotent)."""
        directory = self._dir(session_id)
        with self._lock:
            conn = self._engines.pop(session_id, None)
        if conn:
            conn[0].dispose()
        shutil.rmtree(directory, ignore_errors=True)

    def cleanup_expired(self) -> int:
        """Delete every session idle for longer than the TTL; returns how many."""
        removed = 0
        for directory in list(self.root.iterdir()):
            if not directory.is_dir() or not _SESSION_ID_RE.fullmatch(directory.name):
                continue
            try:
                self.get(directory.name, touch=False)
            except SessionNotFound:
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
        return removed

    # ------------------------------------------------------------------ data
    def _schema_path(self, session: Session) -> Path:
        return session.directory / "schema.json"

    def schema(self, session: Session) -> SchemaInfo | None:
        """The session's profiled schema, if any data has been added."""
        if session.id in self._engines:
            return self._engines[session.id][2]
        path = self._schema_path(session)
        if not path.exists():
            return None
        return SchemaInfo.model_validate_json(path.read_text(encoding="utf-8"))

    def add_files(self, session: Session, files: list[tuple[Path, str]]) -> SchemaInfo:
        """Ingest uploaded files into the session DB; the DB is restored if ingestion fails."""
        backup = session.directory / "data.sqlite.bak"
        had_db = session.db_path.exists()
        if had_db:
            shutil.copyfile(session.db_path, backup)
        try:
            schema = ingest_files(files, session.db_path, self.settings)
        except Exception:
            if had_db:
                shutil.move(backup, session.db_path)
            else:
                session.db_path.unlink(missing_ok=True)
            raise
        backup.unlink(missing_ok=True)
        self._schema_path(session).write_text(schema.model_dump_json(), encoding="utf-8")
        return schema

    def connect_database(self, session: Session, url: str) -> SchemaInfo:
        """Attach an external read-only database (only when explicitly enabled)."""
        from urllib.parse import urlparse

        engine, dialect = external_engine(url)
        allowed = self.settings.db_connect_allowed_hosts
        host = urlparse(url).hostname or ""
        if allowed and host not in allowed:
            engine.dispose()
            raise PermissionError("That database host is not on the allowlist.")
        schema = profile_schema(engine, dialect, sample_rows=self.settings.profile_sample_rows)
        self._engines[session.id] = (engine, dialect, schema)
        return schema

    def source(self, session: Session) -> DataSource:
        """The DataSource agents should query for this session."""
        if session.id in self._engines:
            engine, dialect, _ = self._engines[session.id]
            return DataSource(dialect=dialect, engine=engine)
        return DataSource(dialect="sqlite", db_path=session.db_path)

    def find_trace(self, trace_id: str) -> Path | None:
        """Locate a trace directory by id (ids are unguessable capabilities)."""
        if not _TRACE_ID_RE.fullmatch(trace_id):
            return None
        for directory in self.root.glob(f"*/traces/{trace_id}"):
            if (directory / "trace.json").exists():
                return directory
        return None
