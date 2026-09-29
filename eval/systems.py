"""Systems under evaluation: a uniform ``predict(case) -> Prediction`` interface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from datasleuth.config import Settings
from datasleuth.db.connectors import sqlite_engine
from datasleuth.db.profile import profile_schema
from datasleuth.db.source import DataSource
from datasleuth.llm.base import LLMOutputError
from datasleuth.llm.client import LLM, CallTracker
from datasleuth.nodes.query_writer import WriterOptions, write_query
from datasleuth.sandbox.executor import Sandbox
from datasleuth.state import ExecutionResult, Language, SchemaInfo


@dataclass(frozen=True)
class Case:
    """One evaluation question against one SQLite database."""

    qid: str
    db_path: Path
    question: str
    gold_sql: str
    difficulty: str
    group: str = ""
    lenient: bool = False


@dataclass
class AttemptLog:
    """Compact per-attempt record stored in results."""

    language: str
    code: str
    ok: bool
    error_type: str | None
    rows: list[list[Any]] = field(default_factory=list)


@dataclass
class Prediction:
    """What a system produced for a case."""

    final_language: str = "sql"
    final_code: str | None = None
    result: ExecutionResult | None = None
    attempts: list[AttemptLog] = field(default_factory=list)
    answer: str | None = None
    status: str = "success"
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)


class System(Protocol):
    """A text-to-answer system."""

    def predict(self, case: Case) -> Prediction: ...


_SCHEMAS: dict[Path, SchemaInfo] = {}


def schema_for(db_path: Path, settings: Settings) -> SchemaInfo:
    """Profile a SQLite database once (memory + disk cache keyed on file size/mtime)."""
    if db_path in _SCHEMAS:
        return _SCHEMAS[db_path]
    sidecar = db_path.with_suffix(".schema.json")
    if sidecar.exists():  # written at ingestion: keeps original names, semantics and warnings
        schema = SchemaInfo.model_validate_json(sidecar.read_text(encoding="utf-8"))
        _SCHEMAS[db_path] = schema
        return schema
    stat = db_path.stat()
    key = hashlib.sha256(
        f"{db_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    ).hexdigest()[:20]
    cache = settings.cache_dir.parent / "schemas" / f"{key}.json"
    if cache.exists():
        schema = SchemaInfo.model_validate_json(cache.read_text(encoding="utf-8"))
    else:
        engine = sqlite_engine(db_path)
        try:
            schema = profile_schema(
                engine, "sqlite", sample_rows=settings.profile_sample_rows, infer_keys=False
            )
        finally:
            engine.dispose()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(schema.model_dump()), encoding="utf-8")
    _SCHEMAS[db_path] = schema
    return schema


class SingleShotSystem:
    """Baselines (1) and (2): one LLM call, no schema linking, no retries."""

    def __init__(self, llm: LLM, settings: Settings, opts: WriterOptions) -> None:
        self.llm = llm
        self.settings = settings
        self.opts = opts

    def predict(self, case: Case) -> Prediction:
        """Write one SQL query for ``case`` and execute it once."""
        schema = schema_for(case.db_path, self.settings)
        tracker = CallTracker(self.settings.max_llm_calls)
        sandbox = Sandbox(DataSource(dialect="sqlite", db_path=case.db_path), self.settings)
        pred = Prediction()
        language: Language = "sql"
        try:
            out = write_query(self.llm, tracker, case.question, schema, self.opts)
        except LLMOutputError as exc:
            pred.status = "failed"
            pred.result = ExecutionResult(
                ok=False, error_type="syntax_error", error_message=str(exc)
            )
            pred.attempts.append(AttemptLog(language, "", False, "syntax_error"))
        else:
            res = sandbox.execute("sql", out.code)
            pred.final_code, pred.result = out.code, res
            pred.status = "success" if res.ok else "failed"
            pred.attempts.append(AttemptLog(language, out.code, res.ok, res.error_type, res.rows))
        pred.llm_calls = tracker.calls
        pred.prompt_tokens, pred.completion_tokens = (
            tracker.prompt_tokens,
            tracker.completion_tokens,
        )
        pred.cost_usd, pred.latency_ms = tracker.cost_usd, tracker.latency_ms
        return pred
