"""Ingestor node: makes sure the agent has a profiled SchemaInfo.

Files are cleaned and loaded at upload time (see ``db/ingest.py``); this node picks up
the already-profiled schema, or profiles the data source on demand (e.g. a SQLite file
or an external database passed straight to the CLI).
"""

from __future__ import annotations

from datasleuth.db.connectors import sqlite_engine
from datasleuth.db.profile import profile_schema
from datasleuth.db.source import DataSource
from datasleuth.nodes.context import RunContext
from datasleuth.state import AgentState, SchemaInfo


def profile_source(source: DataSource, sample_rows: int) -> SchemaInfo:
    """Reflect + profile whatever the source points at."""
    if source.db_path is not None:
        engine = sqlite_engine(source.db_path)
        try:
            return profile_schema(engine, "sqlite", sample_rows=sample_rows)
        finally:
            engine.dispose()
    assert source.engine is not None
    return profile_schema(source.engine, source.dialect, sample_rows=sample_rows)


def ingest(state: AgentState, ctx: RunContext) -> AgentState:
    """Provide ``schema_info`` (and the unpruned ``pruned_schema`` default)."""
    schema = state.get("schema_info") or ctx.schema
    if schema is None:
        schema = profile_source(ctx.source, ctx.settings.profile_sample_rows)
        ctx.schema = schema
    return {
        "schema_info": schema,
        "pruned_schema": schema,
        "attempts": [],
        "retry_count": 0,
        "caveats": [],
        "widened": False,
        "status": "running",
    }
