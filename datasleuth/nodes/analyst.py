"""Analyst node: a concise answer grounded only in the returned data."""

from __future__ import annotations

import re

from pydantic import BaseModel

from datasleuth.llm.base import LLMBudgetExceeded, LLMOutputError
from datasleuth.nodes.context import RunContext
from datasleuth.prompts import data_block, render
from datasleuth.state import AgentState, ExecutionResult, sanitize_for_prompt

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PREVIEW_ROWS = 30
_REL_TOL = 0.006


class AnalystOutput(BaseModel):
    """Analyst LLM output."""

    answer: str


def _numbers(text: str) -> list[float]:
    out: list[float] = []
    for raw in _NUM_RE.findall(text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def ungrounded_numbers(answer: str, res: ExecutionResult, question: str) -> list[float]:
    """Numbers in ``answer`` that appear neither in the data, the question nor as counts."""
    known: list[float] = [float(res.row_count), float(len(res.columns))]
    known += _numbers(question)
    for row in res.rows:
        for cell in row:
            if isinstance(cell, bool):
                continue
            if isinstance(cell, int | float):
                known.append(float(cell))
            elif isinstance(cell, str):
                known += _numbers(cell)
    bad = []
    for n in _numbers(answer):
        if n == int(n) and n <= 10:  # "top 3", "two of them": too noisy to police
            continue
        if not any(abs(n - k) <= _REL_TOL * max(1.0, abs(k)) for k in known):
            bad.append(n)
    return bad


def fallback_answer(res: ExecutionResult) -> str:
    """Deterministic summary used when no LLM answer is available."""
    if res.row_count == 0:
        return "No matching rows were found."
    if res.row_count == 1 and len(res.columns) == 1:
        return f"{res.columns[0]}: {res.rows[0][0]}"
    suffix = " (truncated)" if res.truncated else ""
    return f"{res.row_count} row(s){suffix} with columns {', '.join(res.columns)}; see the table below."


def _result_block(res: ExecutionResult) -> str:
    lines = [" | ".join(sanitize_for_prompt(c, 40) for c in res.columns)]
    lines += [
        " | ".join(sanitize_for_prompt(v, 60) for v in row) for row in res.rows[:_PREVIEW_ROWS]
    ]
    shown = min(res.row_count, _PREVIEW_ROWS)
    head = f"{res.row_count} row(s) total, first {shown} shown" + (
        "; RESULT WAS TRUNCATED at the row cap" if res.truncated else ""
    )
    return data_block("result_data", head + "\n" + "\n".join(lines))


def analyst(state: AgentState, ctx: RunContext) -> AgentState:
    """Write the natural-language answer; verify every number against the data."""
    res = state.get("final_result")
    if res is None or not res.ok:
        return {}
    caveats = list(state.get("caveats", []))
    if not ctx.settings.use_analyst:
        return {"answer": fallback_answer(res)}
    schema = state.get("pruned_schema") or state["schema_info"]
    notes = [
        w for t in schema.tables for w in t.warnings if "could not be parsed" in w or "assum" in w
    ]
    parts = [f"Question: {state['question']}", _result_block(res)]
    if notes or caveats:
        parts.append(data_block("notes", "\n".join(notes + caveats)))
    user = "\n\n".join(parts)
    try:
        answer = ctx.llm.generate_structured(
            AnalystOutput, render("analyst"), user, purpose="analyst", tracker=ctx.tracker
        ).answer.strip()
        bad = ungrounded_numbers(answer, res, state["question"])
        if bad:
            shown = ", ".join(f"{n:g}" for n in bad[:5])
            retry = (
                f"{user}\n\nYour previous answer contained figures that are not in the result: "
                f"{shown}. Rewrite it using only figures from the result."
            )
            answer = ctx.llm.generate_structured(
                AnalystOutput,
                render("analyst"),
                retry,
                purpose="analyst:grounding",
                tracker=ctx.tracker,
            ).answer.strip()
            if ungrounded_numbers(answer, res, state["question"]):
                caveats.append(
                    "Some figures in the summary could not be verified against the data."
                )
    except (LLMOutputError, LLMBudgetExceeded):
        answer = fallback_answer(res)
    return {"answer": answer, "caveats": caveats}
