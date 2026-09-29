"""Pydantic models and the LangGraph state shared by every node."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

Dialect = Literal["sqlite", "postgres", "mysql"]
Language = Literal["sql", "pandas"]
ErrorType = Literal[
    "syntax_error",
    "unknown_column",
    "unknown_table",
    "type_mismatch",
    "ambiguous_column",
    "join_error",
    "empty_result",
    "timeout",
    "semantic_mismatch",
    "validation_error",
    "runtime_error",
]
DType = Literal["integer", "real", "text", "date", "boolean"]


class _Base(BaseModel):
    """Base for API-visible models: fields with defaults are always sent, so mark them required
    in the *serialization* schema (accurate OpenAPI/TypeScript types). LLM-facing validation
    schemas are unaffected."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")


def sanitize_for_prompt(value: object, limit: int = 60) -> str:
    """Flatten and truncate untrusted text (cell values, names) before prompt insertion."""
    text = _CONTROL_RE.sub(" ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- schema
class ColumnInfo(_Base):
    """One column with its inferred type and profile."""

    name: str
    original_name: str | None = None
    dtype: str
    semantic: str | None = Field(
        default=None, description="e.g. 'currency' or 'percent (12 means 12%)'."
    )
    null_count: int = 0
    distinct_count: int | None = None
    min: str | None = None
    max: str | None = None
    samples: list[str] = Field(default_factory=list)
    primary_key: bool = False


class ForeignKey(_Base):
    """A foreign-key relationship from this table to another."""

    columns: list[str]
    ref_table: str
    ref_columns: list[str]


class TableInfo(_Base):
    """A table (or a file/sheet ingested as one)."""

    name: str
    source: str | None = Field(default=None, description="Display name of the origin file/sheet.")
    row_count: int = 0
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def column(self, name: str) -> ColumnInfo | None:
        """Case-insensitive column lookup."""
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


class SchemaInfo(_Base):
    """Everything the agent knows about the available data."""

    dialect: Dialect = "sqlite"
    tables: list[TableInfo]

    def table(self, name: str) -> TableInfo | None:
        """Case-insensitive table lookup."""
        lowered = name.lower()
        return next((t for t in self.tables if t.name.lower() == lowered), None)

    def prune(self, selection: dict[str, list[str]]) -> SchemaInfo:
        """Keep only the selected tables/columns (PK/FK columns are always retained)."""
        tables: list[TableInfo] = []
        for tbl in self.tables:
            match = next((k for k in selection if k.lower() == tbl.name.lower()), None)
            if match is None:
                continue
            wanted = {c.lower() for c in selection[match]}
            fk_cols = {c.lower() for fk in tbl.foreign_keys for c in fk.columns}
            cols = [
                c
                for c in tbl.columns
                if not wanted
                or c.name.lower() in wanted
                or c.primary_key
                or c.name.lower() in fk_cols
            ]
            tables.append(tbl.model_copy(update={"columns": cols}))
        return SchemaInfo(dialect=self.dialect, tables=tables or self.tables)

    def render(self, *, include_samples: bool = True, minimal: bool = False) -> str:
        """Render as compact text for prompts. Content is untrusted DATA, not instructions.

        ``minimal`` emits only names, types and keys (the zero-shot baseline schema).
        """
        lines: list[str] = []
        for tbl in self.tables:
            if minimal:
                cols = ", ".join(
                    f'"{sanitize_for_prompt(c.name, 80)}" {c.dtype}{" PK" if c.primary_key else ""}'
                    for c in tbl.columns
                )
                lines.append(f'TABLE "{sanitize_for_prompt(tbl.name, 80)}" ({cols})')
                lines.extend(
                    f"  FK ({', '.join(fk.columns)}) -> {fk.ref_table}({', '.join(fk.ref_columns)})"
                    for fk in tbl.foreign_keys
                )
                continue
            lines.append(f'TABLE "{sanitize_for_prompt(tbl.name, 80)}" ({tbl.row_count} rows)')
            for col in tbl.columns:
                parts = [f'  - "{sanitize_for_prompt(col.name, 80)}" {col.dtype}']
                flags = []
                if col.primary_key:
                    flags.append("PK")
                if col.semantic:
                    flags.append(sanitize_for_prompt(col.semantic))
                if col.null_count:
                    flags.append(f"{col.null_count} nulls")
                if col.distinct_count is not None:
                    flags.append(f"{col.distinct_count} distinct")
                if flags:
                    parts.append(f"[{', '.join(flags)}]")
                if col.min is not None and col.max is not None:
                    parts.append(
                        f"range {sanitize_for_prompt(col.min, 30)}..{sanitize_for_prompt(col.max, 30)}"
                    )
                if include_samples and col.samples:
                    shown = ", ".join(repr(sanitize_for_prompt(s, 40)) for s in col.samples)
                    parts.append(f"e.g. {shown}")
                lines.append(" ".join(parts))
            for fk in tbl.foreign_keys:
                lines.append(
                    f"  FK ({', '.join(fk.columns)}) -> {fk.ref_table}({', '.join(fk.ref_columns)})"
                )
            for warning in tbl.warnings:
                lines.append(f"  NOTE: {sanitize_for_prompt(warning, 160)}")
        return "\n".join(lines)


# ------------------------------------------------------------------------ planning
class Plan(_Base):
    """Router/Planner output."""

    status: Literal["answerable", "clarify", "refuse"] = "answerable"
    question_types: list[
        Literal["aggregation", "filter", "join", "ranking", "time_series", "comparison", "other"]
    ] = Field(default_factory=list)
    needs_chart: bool = False
    expected_columns: int | None = Field(
        default=None,
        description="Informational only: the planner's guess of the answer's column count. "
        "Not used to trigger retries (measured to be unreliable).",
    )
    expects_many_rows: bool | None = Field(
        default=None, description="True for lists/breakdowns, False for a single value."
    )
    preferred_language: Language = "sql"
    steps: str = Field(default="", description="Short plan (1-4 sentences).")
    message: str | None = Field(
        default=None, description="Clarifying question or refusal reason when not answerable."
    )


# ---------------------------------------------------------------------- execution
class ExecutionResult(_Base):
    """Outcome of running validated code."""

    ok: bool
    rows: list[list[Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error_type: ErrorType | None = None
    error_message: str | None = None
    runtime_ms: float = 0.0


class Diagnosis(_Base):
    """Critic verdict on a failed or suspicious attempt."""

    error_type: ErrorType
    explanation: str
    repair_instruction: str
    escalate: Literal["none", "widen_schema", "switch_to_pandas"] = "none"


class QueryOutput(_Base):
    """Structured Query Writer output."""

    language: Language = "sql"
    code: str
    rationale: str = ""


class Attempt(_Base):
    """One write→validate→execute cycle, kept in the trace."""

    index: int
    language: Language
    code: str
    rationale: str = ""
    result: ExecutionResult | None = None
    error: str | None = None
    diagnosis: Diagnosis | None = None


# ------------------------------------------------------------------------- charts
class ChartSpec(_Base):
    """LLM-emitted chart description; rendering is deterministic code."""

    chart_type: Literal["bar", "line", "scatter", "pie", "histogram", "box", "area"]
    x: str
    y: str | None = None
    color: str | None = None
    aggregation: Literal["none", "sum", "mean", "count", "min", "max"] = "none"
    title: str = ""
    x_label: str | None = None
    y_label: str | None = None
    sort: Literal["none", "asc", "desc"] = "none"
    top_n: int | None = Field(default=None, ge=1, le=50)


class Metrics(_Base):
    """Per-question accounting."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    attempts: int = 0
    retries: int = 0


Status = Literal["running", "success", "failed", "clarify", "refused"]


class Report(_Base):
    """Everything returned to the user for one question."""

    trace_id: str
    question: str
    status: Status
    answer: str | None = None
    language: Language | None = None
    code: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    chart_spec: ChartSpec | None = None
    chart_json: str | None = None
    has_chart_png: bool = False
    plan: Plan | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    metrics: Metrics = Field(default_factory=Metrics)


# --------------------------------------------------------------------------- state
class AgentState(TypedDict, total=False):
    """LangGraph state. Node functions return partial updates."""

    question: str
    schema_info: SchemaInfo
    pruned_schema: SchemaInfo
    plan: Plan
    attempts: list[Attempt]
    retry_count: int
    repair_instruction: str | None
    escalation: Literal["widen_schema", "switch_to_pandas"] | None
    widened: bool
    force_language: Language | None
    final_result: ExecutionResult | None
    caveats: list[str]
    status: Status
    failure_reason: str | None
    answer: str | None
    chart_spec: ChartSpec | None
    chart_path: str | None
    chart_json: str | None
    metrics: Metrics
    trace_id: str
    report: Report
