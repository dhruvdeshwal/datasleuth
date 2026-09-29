"""Reporter node: assemble the final Report and persist the full trace."""

from __future__ import annotations

import time

from datasleuth.nodes.context import RunContext
from datasleuth.state import AgentState, Metrics, Report


def _failure_message(state: AgentState) -> str:
    reason = state.get("failure_reason") or "the agent could not produce a working query"
    return (
        f"I couldn't answer this question: {reason}. "
        "The full attempt trace is available; try rephrasing the question or naming the columns to use."
    )


def build_metrics(state: AgentState, ctx: RunContext) -> Metrics:
    """Aggregate tracker usage and wall-clock latency."""
    attempts = state.get("attempts", [])
    t = ctx.tracker
    return Metrics(
        llm_calls=t.calls,
        prompt_tokens=t.prompt_tokens,
        completion_tokens=t.completion_tokens,
        cost_usd=t.cost_usd,
        latency_ms=(time.perf_counter() - ctx.started) * 1000,
        llm_latency_ms=t.latency_ms,
        attempts=len(attempts),
        retries=max(0, len(attempts) - 1),
    )


def reporter(state: AgentState, ctx: RunContext) -> AgentState:
    """Build the :class:`Report`; write ``trace.json`` when an artifacts dir is set."""
    status = state.get("status", "failed")
    if status == "running":
        status = "failed"
    attempts = state.get("attempts", [])
    last = attempts[-1] if attempts else None
    res = state.get("final_result")
    answer = state.get("answer")
    if status == "failed" and not answer:
        answer = _failure_message(state)
    report = Report(
        trace_id=state.get("trace_id", ""),
        question=state["question"],
        status=status,
        answer=answer,
        language=last.language if last else None,
        code=last.code if last else None,
        columns=res.columns if res and res.ok else [],
        rows=res.rows if res and res.ok else [],
        row_count=res.row_count if res and res.ok else 0,
        truncated=bool(res and res.truncated),
        chart_spec=state.get("chart_spec"),
        chart_json=state.get("chart_json"),
        has_chart_png=bool(state.get("chart_path")),
        plan=state.get("plan"),
        attempts=attempts,
        caveats=state.get("caveats", []),
        failure_reason=state.get("failure_reason"),
        metrics=build_metrics(state, ctx),
    )
    if ctx.artifacts_dir is not None:
        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifacts_dir / "trace.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
    return {"report": report, "status": status, "answer": answer, "metrics": report.metrics}
