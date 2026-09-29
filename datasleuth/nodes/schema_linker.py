"""Schema Linker node: prune the schema to what the question needs (toggleable)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from datasleuth.fewshot import tokenize
from datasleuth.llm.base import LLMOutputError
from datasleuth.nodes.context import RunContext
from datasleuth.prompts import data_block, render
from datasleuth.state import AgentState, SchemaInfo

_SMALL_SCHEMA_COLUMNS = 40
_LEXICAL_MAX_TABLES = 4


class LinkedTable(BaseModel):
    """A table and the columns selected from it."""

    name: str
    columns: list[str] = Field(default_factory=list)


class LinkerOutput(BaseModel):
    """Schema Linker LLM output."""

    tables: list[LinkedTable]


def _should_skip(schema: SchemaInfo, ctx: RunContext, state: AgentState) -> bool:
    if state.get("escalation") == "widen_schema" or not ctx.settings.use_schema_linker:
        return True
    total_cols = sum(len(t.columns) for t in schema.tables)
    return (
        len(schema.tables) < ctx.settings.schema_linker_min_tables
        and total_cols <= _SMALL_SCHEMA_COLUMNS
    )


def _add_bridge_tables(schema: SchemaInfo, selection: dict[str, list[str]]) -> dict[str, list[str]]:
    """Add tables that connect two selected tables through foreign keys."""
    chosen = {n.lower() for n in selection}
    for tbl in schema.tables:
        if tbl.name.lower() in chosen:
            continue
        neighbours = {fk.ref_table.lower() for fk in tbl.foreign_keys}
        neighbours |= {
            other.name.lower()
            for other in schema.tables
            if any(fk.ref_table.lower() == tbl.name.lower() for fk in other.foreign_keys)
        }
        if len(neighbours & chosen) >= 2:
            selection[tbl.name] = []
            chosen.add(tbl.name.lower())
    return selection


def lexical_link(schema: SchemaInfo, question: str) -> dict[str, list[str]]:
    """Fallback linker: token overlap between the question and names/sample values."""
    q_tokens = set(tokenize(question))
    scored: list[tuple[float, str, list[str]]] = []
    for tbl in schema.tables:
        t_score = float(len(q_tokens & set(tokenize(tbl.name.replace("_", " ")))))
        cols: list[str] = []
        for col in tbl.columns:
            c_tokens = set(tokenize(col.name.replace("_", " ")))
            v_tokens = {t for s in col.samples for t in tokenize(s)}
            c_score = len(q_tokens & c_tokens) + 0.5 * len(q_tokens & v_tokens)
            if c_score > 0 or col.primary_key:
                cols.append(col.name)
            t_score += c_score
        scored.append((t_score, tbl.name, cols))
    scored.sort(key=lambda s: (-s[0], s[1]))
    picked = [s for s in scored if s[0] > 0][:_LEXICAL_MAX_TABLES] or scored[:1]
    return {name: cols for _, name, cols in picked}


def _validate_selection(schema: SchemaInfo, out: LinkerOutput) -> dict[str, list[str]]:
    """Drop hallucinated table/column names from the LLM's selection."""
    selection: dict[str, list[str]] = {}
    for item in out.tables:
        tbl = schema.table(item.name)
        if tbl is None:
            continue
        selection[tbl.name] = [c.name for n in item.columns if (c := tbl.column(n))]
    return selection


def link_schema(state: AgentState, ctx: RunContext) -> AgentState:
    """Select relevant tables/columns; on escalation, restore the full schema."""
    schema = state["schema_info"]
    if state.get("escalation") == "widen_schema":
        return {"pruned_schema": schema, "widened": True, "escalation": None}
    if _should_skip(schema, ctx, state):
        return {"pruned_schema": schema}
    user = (
        data_block("schema_data", schema.render(include_samples=True))
        + f"\n\nQuestion: {state['question']}"
    )
    selection: dict[str, list[str]] = {}
    try:
        out = ctx.llm.generate_structured(
            LinkerOutput,
            render("schema_linker"),
            user,
            purpose="schema_linker",
            tracker=ctx.tracker,
        )
        selection = _validate_selection(schema, out)
    except LLMOutputError:
        selection = {}
    if not selection:
        selection = lexical_link(schema, state["question"])
    selection = _add_bridge_tables(schema, selection)
    return {"pruned_schema": schema.prune(selection)}
