"""Chart Agent node: LLM proposes a ChartSpec; validated, then rendered by fixed code."""

from __future__ import annotations

from pydantic import BaseModel

from datasleuth.charts.render import export_png, render_chart
from datasleuth.charts.spec import canonical_columns, validate_chart_spec
from datasleuth.llm.base import LLMBudgetExceeded, LLMOutputError
from datasleuth.nodes.context import RunContext
from datasleuth.prompts import data_block, render
from datasleuth.state import AgentState, ChartSpec, ExecutionResult, sanitize_for_prompt

_MAX_PLOT_ROWS = 500
_CHART_TYPES = {"time_series", "comparison", "ranking", "aggregation"}


class ChartDecision(BaseModel):
    """Chart Agent LLM output."""

    use_chart: bool
    spec: ChartSpec | None = None
    reason: str = ""


def _has_numeric(res: ExecutionResult) -> bool:
    return any(
        all(isinstance(r[i], int | float) or r[i] is None for r in res.rows)
        and any(r[i] is not None for r in res.rows)
        for i in range(len(res.columns))
    )


def worth_charting(res: ExecutionResult, state: AgentState) -> bool:
    """Cheap gate so we only spend an LLM call where a chart could plausibly help."""
    if res.row_count < 2 or res.row_count > _MAX_PLOT_ROWS or not _has_numeric(res):
        return False
    plan = state.get("plan")
    if plan and plan.needs_chart:
        return True
    return bool(plan and res.row_count >= 3 and _CHART_TYPES & set(plan.question_types))


def _prompt(state: AgentState, res: ExecutionResult, problems: list[str]) -> str:
    head = " | ".join(sanitize_for_prompt(c, 40) for c in res.columns)
    body = "\n".join(" | ".join(sanitize_for_prompt(v, 30) for v in r) for r in res.rows[:15])
    text = f"Question: {state['question']}\n\n" + data_block(
        "result_data", f"{res.row_count} rows; first 15:\n{head}\n{body}"
    )
    if problems:
        text += (
            "\n\nYour previous spec was invalid: "
            + "; ".join(problems)
            + ". Fix it or set use_chart=false."
        )
    return text


def chart(state: AgentState, ctx: RunContext) -> AgentState:
    """Decide on, validate and render a chart; skip gracefully on any problem."""
    res = state.get("final_result")
    if not ctx.settings.use_chart or res is None or not res.ok or not worth_charting(res, state):
        return {}
    problems: list[str] = []
    for round_ in range(2):  # initial spec + one repair
        try:
            decision = ctx.llm.generate_structured(
                ChartDecision,
                render("chart_agent"),
                _prompt(state, res, problems),
                purpose="chart_agent" if round_ == 0 else "chart_agent:repair",
                tracker=ctx.tracker,
            )
        except (LLMOutputError, LLMBudgetExceeded):
            return {}
        if not decision.use_chart or decision.spec is None:
            return {}
        problems = validate_chart_spec(decision.spec, res.columns, res.rows)
        if not problems:
            spec = canonical_columns(decision.spec, res.columns)
            fig = render_chart(spec, res.columns, res.rows)
            update: AgentState = {"chart_spec": spec, "chart_json": fig.to_json()}
            if ctx.artifacts_dir is not None:
                png = ctx.artifacts_dir / "chart.png"
                if export_png(fig, png):
                    update["chart_path"] = str(png)
            return update
    return {}
