"""Sandbox Executor node: run the validated attempt and record the result."""

from __future__ import annotations

from datasleuth.nodes.context import RunContext
from datasleuth.state import AgentState


def execute(state: AgentState, ctx: RunContext) -> AgentState:
    """Execute the latest (already validated) attempt inside the sandbox."""
    attempts = list(state["attempts"])
    attempt = attempts[-1]
    pruned = state.get("pruned_schema")
    tables = [t.name for t in pruned.tables] if pruned else None
    result = ctx.sandbox.execute_validated(attempt.language, attempt.code, tables)
    attempts[-1] = attempt.model_copy(
        update={"result": result, "error": None if result.ok else result.error_message}
    )
    return {"attempts": attempts}
