"""Critic/Corrector node.

*Errors* are diagnosed deterministically (free, fast, and good for an LLM on a rate-limited
tier): the DB message is mapped onto the error taxonomy and turned into a targeted repair
instruction that names the closest valid tables/columns. *Suspicious-but-successful*
results (empty, all NULL, wrong shape, Cartesian blow-up) are judged by one LLM call when
``use_critic_semantic`` is on. Repeated identical failures escalate: widen the pruned
schema first, then switch SQL to Pandas.
"""

from __future__ import annotations

import difflib
import re
from typing import Literal

from pydantic import BaseModel

from datasleuth.llm.base import LLMBudgetExceeded, LLMOutputError
from datasleuth.nodes.context import RunContext
from datasleuth.prompts import data_block, render
from datasleuth.state import (
    AgentState,
    Attempt,
    Diagnosis,
    ErrorType,
    ExecutionResult,
    SchemaInfo,
    sanitize_for_prompt,
)

_NAME_PATTERNS = [
    r"no such column: ([\w\.\"`]+)",
    r"has no column named ([\w\"`]+)",
    r'column "?([\w\.]+)"? does not exist',
    r"Unknown column '([^']+)'",
    r"KeyError: '([^']+)'",
]
_TABLE_PATTERNS = [
    r"no such table: ([\w\.\"`]+)",
    r'relation "?([\w\.]+)"? does not exist',
    r"Table '([^']+)' doesn't exist",
    r"NameError: name '([^']+)' is not defined",
]
_AMBIGUOUS_PATTERN = r"ambiguous column name: ([\w\.\"`]+)"


class CriticVerdict(BaseModel):
    """LLM verdict on a suspicious result."""

    accept: bool
    error_type: ErrorType = "semantic_mismatch"
    explanation: str = ""
    repair_instruction: str = ""


def _find(patterns: list[str], message: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1).strip("\"`'")
    return None


def _closest(name: str, candidates: list[str], n: int = 4) -> list[str]:
    lowered = {c.lower(): c for c in candidates}
    hits = difflib.get_close_matches(name.lower(), list(lowered), n=n, cutoff=0.5)
    return [lowered[h] for h in hits]


def diagnose_error(attempt: Attempt, schema: SchemaInfo) -> Diagnosis:
    """Map a failed attempt onto the taxonomy and build a targeted repair instruction."""
    res = attempt.result
    kind: ErrorType = (res.error_type if res and res.error_type else None) or "runtime_error"
    message = (res.error_message if res else None) or attempt.error or "unknown error"
    lang = "Pandas" if attempt.language == "pandas" else f"{schema.dialect} SQL"
    hint = ""
    if kind == "unknown_column":
        name = (_find(_NAME_PATTERNS, message) or "").split(".")[-1]
        owners = [t.name for t in schema.tables if name and t.column(name)]
        all_cols = sorted({c.name for t in schema.tables for c in t.columns})
        similar = [
            f"{t.name}.{c.name}"
            for t in schema.tables
            for c in t.columns
            if c.name in _closest(name, all_cols)
        ]
        hint = f'Column "{name}" does not exist in the referenced table(s). '
        if owners:
            hint += f"A column with that name exists in: {', '.join(owners)}. "
        if similar:
            hint += f"Similar columns: {', '.join(similar[:6])}. "
        hint += "Use only columns from the schema, qualified by their table."
    elif kind == "unknown_table":
        name = (_find(_TABLE_PATTERNS, message) or "").split(".")[-1]
        tables = [t.name for t in schema.tables]
        hint = f'Table "{name}" does not exist. '
        hint += f"Available tables: {', '.join(tables)}."
        if attempt.language == "pandas":
            hint += " In Pandas each table is a DataFrame variable named exactly like the table."
    elif kind == "ambiguous_column":
        name = (_find([_AMBIGUOUS_PATTERN], message) or "").split(".")[-1]
        owners = [t.name for t in schema.tables if name and t.column(name)]
        hint = f'Column "{name}" is ambiguous. Qualify it with a table alias' + (
            f" (present in: {', '.join(owners)})." if owners else "."
        )
    elif kind == "syntax_error":
        hint = f"Fix the syntax error and write valid {lang}."
    elif kind == "type_mismatch":
        hint = (
            "Fix the type problem: dates are ISO text (compare/cast accordingly), and "
            "currency/percent columns are already numeric. Cast explicitly when mixing types."
        )
    elif kind == "timeout":
        hint = (
            "The query was too slow. Avoid cross joins and missing join conditions, filter "
            "before joining, aggregate early, and avoid unbounded recursion."
        )
    elif kind == "validation_error":
        hint = (
            "The code was blocked by the sandbox policy. Write a single read-only SELECT"
            if attempt.language == "sql"
            else "The code was blocked by the sandbox policy. Use only pandas/numpy/math, "
            "no file access, no dunder attributes"
        ) + " and follow the rules exactly."
    elif kind == "join_error":
        hint = "Check the join keys and conditions; use the documented FK relationships."
    else:
        hint = "Fix the runtime error."
    return Diagnosis(
        error_type=kind,
        explanation=message,
        repair_instruction=f"The previous attempt failed with: {sanitize_for_prompt(message, 300)}. {hint}",
    )


def suspicious_reasons(attempt: Attempt, state: AgentState, ctx: RunContext) -> list[str]:
    """Cheap, deterministic red flags on a *successful* result."""
    res = attempt.result
    assert res is not None and res.ok
    plan = state.get("plan")
    reasons: list[str] = []
    if res.row_count == 0:
        reasons.append("the query returned no rows")
    elif res.rows and all(v is None for row in res.rows for v in row):
        reasons.append("every returned value is NULL")
    # Note: the planner's `expected_columns` guess is deliberately NOT a red flag. Measured on
    # Spider it was often wrong, and acting on it turned correct queries into wrong ones.
    if plan and plan.expects_many_rows and res.row_count == 1 and not res.truncated:
        reasons.append("only one row came back although the question asks for several")
    joined = " join " in attempt.code.lower() or ".merge(" in attempt.code
    if joined and (res.truncated or res.row_count >= ctx.settings.row_cap):
        reasons.append(
            "a huge join result suggests a Cartesian product or a missing join condition"
        )
    return reasons


def _result_preview(res: ExecutionResult, rows: int = 10) -> str:
    header = " | ".join(sanitize_for_prompt(c, 40) for c in res.columns)
    body = "\n".join(" | ".join(sanitize_for_prompt(v, 40) for v in row) for row in res.rows[:rows])
    return f"{res.row_count} row(s){' (truncated)' if res.truncated else ''}\n{header}\n{body}"


def _semantic_verdict(
    attempt: Attempt, reasons: list[str], state: AgentState, ctx: RunContext
) -> CriticVerdict:
    schema = state.get("pruned_schema") or state["schema_info"]
    user = "\n\n".join(
        [
            data_block("schema_data", schema.render(include_samples=True)),
            f"Question: {state['question']}",
            f"Query ({attempt.language}):\n{attempt.code}",
            data_block("result_data", _result_preview(attempt.result)),  # type: ignore[arg-type]
            "Red flags: " + "; ".join(reasons),
        ]
    )
    try:
        return ctx.llm.generate_structured(
            CriticVerdict, render("critic"), user, purpose="critic", tracker=ctx.tracker
        )
    except (LLMOutputError, LLMBudgetExceeded):
        return CriticVerdict(accept=True)


def _signature(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip().lower())


def _is_repeat(attempts: list[Attempt]) -> bool:
    """True if the latest attempt repeats an earlier one (same code, or same error twice)."""
    last = attempts[-1]
    earlier = attempts[:-1]
    if any(_signature(a.code) == _signature(last.code) for a in earlier):
        return True
    err = last.error or ""
    return bool(err) and sum(1 for a in earlier if (a.error or "") == err) >= 1


def _escalation(
    state: AgentState, ctx: RunContext, attempts: list[Attempt]
) -> Literal["widen_schema", "switch_to_pandas"] | None:
    if not _is_repeat(attempts):
        return None
    full, pruned = state["schema_info"], state.get("pruned_schema")
    narrowed = pruned is not None and (
        len(pruned.tables) < len(full.tables)
        or sum(len(t.columns) for t in pruned.tables) < sum(len(t.columns) for t in full.tables)
    )
    if narrowed and not state.get("widened"):
        return "widen_schema"
    if (
        attempts[-1].language == "sql"
        and ctx.pandas_allowed
        and state.get("force_language") != "pandas"
    ):
        return "switch_to_pandas"
    return None


def critic(state: AgentState, ctx: RunContext) -> AgentState:
    """Accept the result, or diagnose it and route a targeted repair back to the writer."""
    attempts = list(state["attempts"])
    attempt = attempts[-1]
    retries = state.get("retry_count", 0)
    res = attempt.result
    caveats = list(state.get("caveats", []))
    max_retries = ctx.settings.max_retries

    if res is not None and res.ok:
        reasons = suspicious_reasons(attempt, state, ctx)
        if not reasons or not ctx.settings.use_critic_semantic:
            return {"final_result": res, "status": "success"}
        if retries >= max_retries:
            caveats.append("The result may be incomplete or wrong: " + "; ".join(reasons) + ".")
            return {"final_result": res, "status": "success", "caveats": caveats}
        if any(
            _signature(a.code) == _signature(attempt.code) and a.result is not None and a.result.ok
            for a in attempts[:-1]
        ):
            # Challenged once, the writer re-derived the identical (already successful) query:
            # treat that as confirmation instead of looping or escalating to Pandas.
            caveats.append("Accepted as is after a re-check: " + "; ".join(reasons) + ".")
            return {"final_result": res, "status": "success", "caveats": caveats}
        verdict = _semantic_verdict(attempt, reasons, state, ctx)
        if verdict.accept:
            return {"final_result": res, "status": "success"}
        diagnosis = Diagnosis(
            error_type=verdict.error_type,
            explanation=verdict.explanation or "; ".join(reasons),
            repair_instruction=verdict.repair_instruction
            or "The result looks wrong (" + "; ".join(reasons) + "). Reconsider the query.",
        )
    else:
        diagnosis = diagnose_error(attempt, state["schema_info"])
        if retries >= max_retries:
            return {
                "final_result": res,
                "status": "failed",
                "failure_reason": (
                    f"No working query after {len(attempts)} attempt(s). "
                    f"Last error: {diagnosis.explanation}"
                ),
            }
    attempts[-1] = attempt.model_copy(update={"diagnosis": diagnosis})
    escalation = _escalation(state, ctx, attempts)
    if escalation:
        diagnosis = diagnosis.model_copy(update={"escalate": escalation})
        attempts[-1] = attempts[-1].model_copy(update={"diagnosis": diagnosis})
    update: AgentState = {
        "attempts": attempts,
        "retry_count": retries + 1,
        "repair_instruction": diagnosis.repair_instruction,
        "escalation": escalation,
    }
    if escalation == "switch_to_pandas":
        update["force_language"] = "pandas"
    return update
