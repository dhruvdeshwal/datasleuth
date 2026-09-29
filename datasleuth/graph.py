"""The LangGraph StateGraph and the public ``DataSleuthAgent`` entry point.

Flow::

    ingest → plan ─┬─ (clarify/refuse) ───────────────────────────────┐
                   └→ link → write → validate → execute → critic ─┬─ retry → write (or link when widening)
                              ↑          └─(blocked)──→ critic     ├─ accept → analyst → chart → report
                              └──────────────────────────────────  └─ give up ─────────────────→ report
"""

from __future__ import annotations

import inspect
import queue
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

import datasleuth.state as state_module
from datasleuth.config import Settings
from datasleuth.db.source import DataSource
from datasleuth.fewshot import FewShotBank
from datasleuth.llm.base import LLMBudgetExceeded
from datasleuth.llm.client import LLM, CallTracker
from datasleuth.nodes.analyst import analyst
from datasleuth.nodes.chart_agent import chart
from datasleuth.nodes.context import Event, RunContext
from datasleuth.nodes.critic import critic
from datasleuth.nodes.executor import execute
from datasleuth.nodes.ingestor import ingest
from datasleuth.nodes.planner import plan_question
from datasleuth.nodes.query_writer import write_node
from datasleuth.nodes.reporter import reporter
from datasleuth.nodes.schema_linker import link_schema
from datasleuth.nodes.validator import validate
from datasleuth.sandbox.executor import Sandbox
from datasleuth.state import AgentState, Report, SchemaInfo
from datasleuth.telemetry import span

NODE_LABELS = {
    "ingest": "Reading schema",
    "plan": "Planning",
    "link": "Selecting relevant tables and columns",
    "write": "Writing query",
    "validate": "Validating query",
    "execute": "Executing in sandbox",
    "critic": "Reviewing result",
    "analyst": "Writing answer",
    "chart": "Choosing a chart",
    "report": "Assembling report",
}
_RECURSION_LIMIT = 80
_SENTINEL = object()


def _msgpack_allowlist() -> list[tuple[str, str]]:
    return [
        (state_module.__name__, name)
        for name, cls in inspect.getmembers(state_module, inspect.isclass)
        if issubclass(cls, BaseModel) and cls.__module__ == state_module.__name__
    ]


def _detail(name: str, state: AgentState, update: AgentState) -> dict[str, Any]:
    """Small, UI-friendly summary of what a node just did."""
    merged: dict[str, Any] = {**state, **update}
    attempts = merged.get("attempts", [])
    last = attempts[-1] if attempts else None
    if name == "plan" and (plan := update.get("plan")):
        return {"status": plan.status, "types": plan.question_types, "steps": plan.steps}
    if name == "link" and (pruned := update.get("pruned_schema")):
        return {"tables": [t.name for t in pruned.tables]}
    if name in {"write", "validate", "execute"} and last:
        detail: dict[str, Any] = {
            "attempt": last.index,
            "language": last.language,
            "code": last.code,
        }
        if last.result is not None:
            detail.update(ok=last.result.ok, error_type=last.result.error_type,
                          error=last.result.error_message, row_count=last.result.row_count)  # fmt: skip
        return detail
    if name == "critic":
        diag = last.diagnosis if last else None
        return {
            "status": merged.get("status"),
            "retry_count": merged.get("retry_count", 0),
            "error_type": diag.error_type if diag else None,
            "repair_instruction": diag.repair_instruction if diag else None,
            "escalation": merged.get("escalation"),
        }
    if name == "analyst":
        return {"answer": update.get("answer")}
    if name == "chart":
        spec = update.get("chart_spec")
        return {"chart_type": spec.chart_type if spec else None}
    return {}


def _instrument(
    name: str, fn: Callable[[AgentState, RunContext], AgentState], ctx: RunContext
) -> Callable[[AgentState], AgentState]:
    def node(state: AgentState) -> AgentState:
        ctx.emit({"type": "node_start", "node": name, "label": NODE_LABELS[name]})
        try:
            with span(f"datasleuth.node.{name}", trace_id=state.get("trace_id", "")):
                update = fn(state, ctx)
        except LLMBudgetExceeded as exc:
            attempts = state.get("attempts", [])
            last_err = attempts[-1].error if attempts else None
            reason = f"{exc}" + (f" Last error: {last_err}" if last_err else "")
            update = {"status": "failed", "failure_reason": reason}
        ctx.emit({
            "type": "node_end", "node": name, "label": NODE_LABELS[name],
            "detail": _detail(name, state, update),
            "llm_calls": ctx.tracker.calls,
        })  # fmt: skip
        return update

    return node


def _failed_or(target: str) -> Callable[[AgentState], str]:
    return lambda state: "report" if state.get("status") == "failed" else target


def _route_after_plan(state: AgentState) -> str:
    return "report" if state.get("status") in {"clarify", "refused", "failed"} else "link"


def _route_after_write(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "report"
    return "critic" if state["attempts"][-1].error else "validate"


def _route_after_validate(state: AgentState) -> str:
    result = state["attempts"][-1].result
    return "critic" if result is not None and not result.ok else "execute"


def _route_after_critic(state: AgentState) -> str:
    status = state.get("status")
    if status == "success":
        return "analyst"
    if status == "failed":
        return "report"
    return "link" if state.get("escalation") == "widen_schema" else "write"


def build_graph(ctx: RunContext, checkpointer: Any = None) -> Any:
    """Compile the agent graph bound to one question's :class:`RunContext`."""
    g = StateGraph(AgentState)
    nodes = {
        "ingest": ingest, "plan": plan_question, "link": link_schema, "write": write_node,
        "validate": validate, "execute": execute, "critic": critic, "analyst": analyst,
        "chart": chart, "report": reporter,
    }  # fmt: skip
    for name, fn in nodes.items():
        # LangGraph's overloads don't model nodes that return partial TypedDict updates.
        g.add_node(name, _instrument(name, fn, ctx))  # type: ignore[call-overload]
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "plan")
    g.add_conditional_edges("plan", _route_after_plan, {"link": "link", "report": "report"})
    g.add_conditional_edges("link", _failed_or("write"), {"write": "write", "report": "report"})
    g.add_conditional_edges(
        "write",
        _route_after_write,
        {"validate": "validate", "critic": "critic", "report": "report"},
    )
    g.add_conditional_edges(
        "validate", _route_after_validate, {"execute": "execute", "critic": "critic"}
    )
    g.add_edge("execute", "critic")
    g.add_conditional_edges(
        "critic", _route_after_critic,
        {"analyst": "analyst", "report": "report", "link": "link", "write": "write"},
    )  # fmt: skip
    g.add_edge("analyst", "chart")
    g.add_edge("chart", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


class DataSleuthAgent:
    """Ask natural-language questions of one data source.

    ``schema`` may be passed when already profiled (uploads). ``checkpoint_path`` enables
    the SQLite checkpointer so every step of a run can be replayed via :meth:`replay`.
    """

    def __init__(
        self,
        source: DataSource,
        settings: Settings,
        *,
        llm: LLM | None = None,
        schema: SchemaInfo | None = None,
        few_shot: FewShotBank | None = None,
        checkpoint_path: Path | None = None,
        artifacts_root: Path | None = None,
    ) -> None:
        self.source = source
        self.settings = settings
        self.llm = llm or LLM(settings)
        self.schema = schema
        self.few_shot = few_shot
        self.artifacts_root = artifacts_root
        self._saver: SqliteSaver | None = None
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
            serde = JsonPlusSerializer(allowed_msgpack_modules=_msgpack_allowlist())
            self._saver = SqliteSaver(conn, serde=serde)

    def _context(self, trace_id: str, emit: Callable[[Event], None]) -> RunContext:
        return RunContext(
            llm=self.llm,
            settings=self.settings,
            source=self.source,
            sandbox=Sandbox(self.source, self.settings),
            tracker=CallTracker(self.settings.max_llm_calls),
            schema=self.schema,
            few_shot=self.few_shot,
            artifacts_dir=self.artifacts_root / trace_id if self.artifacts_root else None,
            emit=emit,
        )

    def run(
        self,
        question: str,
        *,
        trace_id: str | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> Report:
        """Answer ``question`` and return the full :class:`Report`."""
        tid = trace_id or uuid.uuid4().hex[:16]
        ctx = self._context(tid, on_event or (lambda _e: None))
        graph = build_graph(ctx, self._saver)
        final = graph.invoke(
            {"question": question, "trace_id": tid},
            config={"configurable": {"thread_id": tid}, "recursion_limit": _RECURSION_LIMIT},
        )
        report: Report = final["report"]
        return report

    def stream(self, question: str, *, trace_id: str | None = None) -> Iterator[Event]:
        """Yield progress events, ending with ``{"type": "result", "report": ...}``.

        Runs the agent in a worker thread; an exception surfaces as an ``error`` event.
        """
        events: queue.Queue[Any] = queue.Queue()
        tid = trace_id or uuid.uuid4().hex[:16]

        def work() -> None:
            try:
                report = self.run(question, trace_id=tid, on_event=events.put)
                events.put({"type": "result", "report": report})
            except Exception as exc:
                events.put(
                    {"type": "error", "message": str(exc), "error_class": type(exc).__name__}
                )
            finally:
                events.put(_SENTINEL)

        threading.Thread(target=work, daemon=True).start()
        while (item := events.get()) is not _SENTINEL:
            yield item

    def replay(self, trace_id: str) -> list[AgentState]:
        """Return the checkpointed state after every step of a past run (oldest first)."""
        if self._saver is None:
            raise RuntimeError("Replay needs the agent to be created with checkpoint_path.")
        graph = build_graph(self._context(trace_id, lambda _e: None), self._saver)
        history = graph.get_state_history({"configurable": {"thread_id": trace_id}})
        return [snap.values for snap in reversed(list(history))]
