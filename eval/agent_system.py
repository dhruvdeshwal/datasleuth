"""Adapter: run the full DataSleuth agent as an evaluated ``System``."""

from __future__ import annotations

from datasleuth.config import Settings
from datasleuth.db.source import DataSource
from datasleuth.fewshot import FewShotBank
from datasleuth.graph import DataSleuthAgent
from datasleuth.llm.client import LLM
from eval.systems import AttemptLog, Case, Prediction, schema_for


class AgentSystem:
    """The LangGraph agent, configured entirely through ``Settings`` toggles."""

    def __init__(self, llm: LLM, settings: Settings, few_shot: FewShotBank | None = None) -> None:
        self.llm = llm
        self.settings = settings
        self.few_shot = few_shot

    def predict(self, case: Case) -> Prediction:
        """Answer one case and convert the Report into a flat Prediction."""
        agent = DataSleuthAgent(
            DataSource(dialect="sqlite", db_path=case.db_path),
            self.settings,
            llm=self.llm,
            schema=schema_for(case.db_path, self.settings),
            few_shot=self.few_shot,
        )
        report = agent.run(case.question)
        attempts = [
            AttemptLog(
                language=a.language,
                code=a.code,
                ok=bool(a.result and a.result.ok),
                error_type=a.result.error_type if a.result and not a.result.ok else None,
                rows=a.result.rows if a.result and a.result.ok else [],
            )
            for a in report.attempts
        ]
        last = report.attempts[-1] if report.attempts else None
        return Prediction(
            final_language=last.language if last else "sql",
            final_code=last.code if last else None,
            result=last.result if last and report.status == "success" else None,
            attempts=attempts,
            answer=report.answer,
            status=report.status,
            llm_calls=report.metrics.llm_calls,
            prompt_tokens=report.metrics.prompt_tokens,
            completion_tokens=report.metrics.completion_tokens,
            cost_usd=report.metrics.cost_usd,
            latency_ms=report.metrics.latency_ms,
        )
