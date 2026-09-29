"""Validator node: static, LLM-free policy check of the latest attempt."""

from __future__ import annotations

from datasleuth.nodes.context import RunContext
from datasleuth.state import AgentState, ExecutionResult


def validate(state: AgentState, ctx: RunContext) -> AgentState:
    """Mark the latest attempt as failed if it violates the sandbox policy."""
    attempts = list(state["attempts"])
    attempt = attempts[-1]
    verdict = ctx.sandbox.validate(attempt.language, attempt.code)
    if verdict.ok:
        return {}
    attempts[-1] = attempt.model_copy(
        update={
            "error": verdict.message,
            "result": ExecutionResult(
                ok=False, error_type=verdict.error_type, error_message=verdict.message
            ),
        }
    )
    return {"attempts": attempts}
