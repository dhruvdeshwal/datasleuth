"""Query Writer: turns question + (pruned) schema into SQL or Pandas."""

from __future__ import annotations

from dataclasses import dataclass, field

from datasleuth.llm.base import LLMOutputError
from datasleuth.llm.client import LLM, CallTracker
from datasleuth.nodes.context import RunContext
from datasleuth.prompts import data_block, render
from datasleuth.state import (
    AgentState,
    Attempt,
    Dialect,
    ExecutionResult,
    Language,
    Plan,
    QueryOutput,
    SchemaInfo,
    sanitize_for_prompt,
)


@dataclass
class WriterOptions:
    """How much context the writer gets (toggled by ablations)."""

    include_samples: bool = True
    minimal_schema: bool = False
    allow_pandas: bool = True
    force_language: Language | None = None
    examples: list[str] = field(default_factory=list)


def _system_prompt(dialect: Dialect, opts: WriterOptions) -> str:
    pandas_ok = opts.allow_pandas and opts.force_language != "sql"
    return render(
        "query_writer",
        languages="SQL or Pandas" if pandas_ok else "SQL",
        dialect=dialect,
        pandas_rules=render("pandas_rules") if pandas_ok else "",
    )


def _attempts_block(attempts: list[Attempt]) -> str:
    lines: list[str] = []
    for a in attempts[-3:]:
        outcome = a.error or (a.result.error_message if a.result and not a.result.ok else None)
        if outcome is None and a.result is not None:
            outcome = f"returned {a.result.row_count} row(s)"
        lines.append(
            f"Attempt {a.index} ({a.language}):\n{a.code}\n-> {sanitize_for_prompt(outcome or 'n/a', 300)}"
        )
        if a.diagnosis:
            lines.append(f"Diagnosis: {a.diagnosis.error_type}: {a.diagnosis.explanation}")
    return "\n\n".join(lines)


def build_user_prompt(
    question: str,
    schema: SchemaInfo,
    opts: WriterOptions,
    plan: Plan | None = None,
    attempts: list[Attempt] | None = None,
    repair_instruction: str | None = None,
) -> str:
    """Assemble the user message; untrusted content goes in delimited data blocks."""
    parts = [
        data_block(
            "schema_data",
            schema.render(include_samples=opts.include_samples, minimal=opts.minimal_schema),
        )
    ]
    if opts.examples:
        parts.append(data_block("examples", "\n\n".join(opts.examples)))
    if plan and plan.steps:
        parts.append(f"Plan: {plan.steps}")
    if attempts:
        parts.append(data_block("previous_attempts", _attempts_block(attempts)))
    if repair_instruction:
        parts.append(f"REPAIR INSTRUCTION: {repair_instruction}")
    if opts.force_language:
        parts.append(f'You must use language "{opts.force_language}" for this attempt.')
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def write_query(
    llm: LLM,
    tracker: CallTracker,
    question: str,
    schema: SchemaInfo,
    opts: WriterOptions,
    *,
    plan: Plan | None = None,
    attempts: list[Attempt] | None = None,
    repair_instruction: str | None = None,
    purpose: str = "query_writer",
) -> QueryOutput:
    """Call the LLM for a structured ``QueryOutput``."""
    out = llm.generate_structured(
        QueryOutput,
        _system_prompt(schema.dialect, opts),
        build_user_prompt(question, schema, opts, plan, attempts, repair_instruction),
        purpose=purpose,
        tracker=tracker,
    )
    if opts.force_language:
        out.language = opts.force_language
    elif not opts.allow_pandas:
        out.language = "sql"
    out.code = _strip_code_fences(out.code)
    return out


def write_node(state: AgentState, ctx: RunContext) -> AgentState:
    """Graph node: write the next attempt (initial query or a targeted repair)."""
    plan = state.get("plan") or Plan()
    attempts = list(state.get("attempts", []))
    force = state.get("force_language")
    if force is None and plan.preferred_language == "pandas" and ctx.pandas_allowed:
        force = "pandas"
    opts = WriterOptions(
        include_samples=ctx.settings.sample_values_in_prompt,
        allow_pandas=ctx.pandas_allowed,
        force_language=force,
        examples=(
            ctx.few_shot.retrieve(state["question"], ctx.settings.few_shot_k)
            if ctx.settings.use_few_shot and ctx.few_shot
            else []
        ),
    )
    index = len(attempts) + 1
    try:
        out = write_query(
            ctx.llm,
            ctx.tracker,
            state["question"],
            state.get("pruned_schema") or state["schema_info"],
            opts,
            plan=plan,
            attempts=attempts,
            repair_instruction=state.get("repair_instruction"),
            purpose="query_writer" if index == 1 else "query_repair",
        )
        attempt = Attempt(
            index=index, language=out.language, code=out.code, rationale=out.rationale
        )
    except LLMOutputError as exc:
        attempt = Attempt(
            index=index,
            language=force or "sql",
            code="",
            error=f"The model did not return a valid query: {exc}",
            result=ExecutionResult(
                ok=False, error_type="syntax_error", error_message="model returned no valid query"
            ),
        )
    return {"attempts": [*attempts, attempt]}


def _strip_code_fences(code: str) -> str:
    text = code.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
