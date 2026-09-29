"""Router/Planner node: classify the question, plan, or ask/refuse instead of guessing."""

from __future__ import annotations

from datasleuth.nodes.context import RunContext
from datasleuth.prompts import data_block, render
from datasleuth.state import AgentState, Plan


def plan_question(state: AgentState, ctx: RunContext) -> AgentState:
    """Produce a :class:`Plan`; non-answerable questions end the run early."""
    if not ctx.settings.use_router:
        return {"plan": Plan()}
    schema = state["schema_info"]
    user = (
        data_block("schema_data", schema.render(include_samples=True))
        + f"\n\nQuestion: {state['question']}"
    )
    plan = ctx.llm.generate_structured(
        Plan, render("planner"), user, purpose="planner", tracker=ctx.tracker
    )
    if plan.preferred_language == "pandas" and not ctx.pandas_allowed:
        plan.preferred_language = "sql"
    update: AgentState = {"plan": plan}
    if plan.status == "clarify":
        update["status"] = "clarify"
        update["answer"] = plan.message or "Could you clarify your question?"
    elif plan.status == "refuse":
        update["status"] = "refused"
        update["answer"] = (
            plan.message or "This question cannot be answered from the uploaded data."
        )
    return update
