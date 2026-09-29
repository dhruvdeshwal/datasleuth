"""Evaluation runner: resumable, quota-aware, deterministic.

Each run writes ``results/<run_id>/{config.json, per_question.jsonl, summary.json}``.
Re-running the same command resumes from ``per_question.jsonl`` and, thanks to the LLM
disk cache, replays already-answered prompts for free (important on free API tiers).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from datasleuth.config import Settings
from datasleuth.llm.base import LLMError, LLMQuotaExceeded
from datasleuth.llm.client import LLM
from eval.metrics import has_order_by, rows_match, summarize
from eval.systems import Case, Prediction, System

log = logging.getLogger(__name__)
RESULTS_DIR = Path("results")
_MAX_CONSECUTIVE_LLM_ERRORS = 3


class EvalConfig(BaseModel):
    """Config-driven experiment definition (YAML under ``configs/``)."""

    name: str
    description: str = ""
    system: Literal["single_shot", "agent"] = "agent"
    suite: Literal["spider", "custom", "business"] = "spider"
    minimal_schema: bool = False
    include_samples: bool = True
    limit: int | None = None
    seed: int = 0
    ts_eval: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> EvalConfig:
        """Read an EvalConfig from YAML."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def make_settings(self, **overrides: Any) -> Settings:
        """Settings for this run: eval-friendly defaults < config file < explicit overrides."""
        base: dict[str, Any] = {"row_cap": 10_000, "cell_cap_chars": 2000, "sql_timeout_s": 30.0}
        return Settings(**{**base, **self.settings, **overrides})


def run_id_for(config: EvalConfig, settings: Settings, n: int) -> str:
    """Deterministic run id so identical invocations resume."""
    model = settings.llm_model.replace("/", "-").replace(":", "-")
    return f"{config.name}__{model}__n{n}_s{config.seed}"


def gold_result(db_path: Path, sql: str, settings: Settings) -> list[list[Any]]:
    """Execute gold SQL (trusted) and return normalised rows."""
    from datasleuth.sandbox.sql_executor import run_sqlite

    res = run_sqlite(db_path, sql, settings)
    if not res.ok:
        raise RuntimeError(f"gold query failed: {res.error_message}")
    return res.rows


def _attempt_matches(
    rows: list[list[Any]] | None, ok: bool, gold_rows: Any, order: bool, lenient: bool = False
) -> bool:
    return bool(ok and rows is not None and rows_match(rows, gold_rows, order, lenient))


def score(case: Case, pred: Prediction, settings: Settings) -> dict[str, Any]:
    """Turn a Prediction into the flat record stored in ``per_question.jsonl``."""
    final = pred.result
    first = pred.attempts[0] if pred.attempts else None
    if not case.gold_sql:  # unanswerable: correct behaviour is to decline or ask, not to answer
        ex = pred.status in {"refused", "clarify"}
        first_ex = ex
    else:
        order = has_order_by(case.gold_sql)
        gold_rows = gold_result(case.db_path, case.gold_sql, settings)
        ex = _attempt_matches(
            final.rows if final else None, bool(final and final.ok), gold_rows, order, case.lenient
        )
        first_ex = bool(
            first and _attempt_matches(first.rows, first.ok, gold_rows, order, case.lenient)
        )
    return {
        "qid": case.qid,
        "group": case.group,
        "difficulty": case.difficulty,
        "question": case.question,
        "gold_sql": case.gold_sql,
        "pred_code": pred.final_code,
        "pred_language": pred.final_language,
        "status": pred.status,
        "ex": ex,
        "first_attempt_ex": first_ex,
        "first_attempt_valid": bool(first and first.ok),
        "retries": max(0, len(pred.attempts) - 1),
        "attempt_errors": [a.error_type for a in pred.attempts if a.error_type],
        "attempts": [
            {"language": a.language, "code": a.code, "ok": a.ok, "error_type": a.error_type}
            for a in pred.attempts
        ],
        "final_error": final.error_message if final and not final.ok else None,
        "final_rows": final.rows[:200] if final and final.ok else None,
        "answer": pred.answer,
        "llm_calls": pred.llm_calls,
        "prompt_tokens": pred.prompt_tokens,
        "completion_tokens": pred.completion_tokens,
        "cost_usd": pred.cost_usd,
        "latency_ms": pred.latency_ms,
    }


def _read_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done[rec["qid"]] = rec
    return done


def run_eval(
    config: EvalConfig,
    cases: list[Case],
    make_system: Callable[[LLM, Settings], System],
    *,
    settings: Settings | None = None,
    llm: LLM | None = None,
    results_dir: Path = RESULTS_DIR,
    ts_scorer: Callable[[Case, Prediction], bool | None] | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run ``cases`` through the configured system and write results; returns the summary."""
    settings = settings or config.make_settings()
    run_dir = results_dir / run_id_for(config, settings, len(cases))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {**config.model_dump(), "n_cases": len(cases), "llm_model": settings.llm_model,
             "llm_provider": settings.llm_provider, "temperature": settings.llm_temperature},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )  # fmt: skip
    per_q = run_dir / "per_question.jsonl"
    done = _read_done(per_q)
    system = make_system(llm or LLM(settings), settings)
    todo = [c for c in cases if c.qid not in done]
    out(f"[{config.name}] {len(done)} cached, {len(todo)} to run (run dir: {run_dir})")
    complete, llm_errors_in_row = True, 0
    with per_q.open("a", encoding="utf-8") as fh:
        for i, case in enumerate(todo, start=1):
            try:
                pred = system.predict(case)
                llm_errors_in_row = 0
            except LLMQuotaExceeded as exc:
                out(
                    f"\nStopped: {exc}\nRe-run the same command to resume; finished answers are cached."
                )
                complete = False
                break
            except LLMError as exc:
                llm_errors_in_row += 1
                out(f"  ! {case.qid}: LLM error: {exc}")
                if llm_errors_in_row >= _MAX_CONSECUTIVE_LLM_ERRORS:
                    out("Aborting after repeated LLM errors (check your API key / quota).")
                    complete = False
                    break
                continue
            rec = score(case, pred, settings)
            if ts_scorer is not None:
                rec["ts_ex"] = ts_scorer(case, pred)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            done[case.qid] = rec
            mark = "ok " if rec["ex"] else "BAD"
            out(
                f"  [{i}/{len(todo)}] {mark} {case.qid} ({case.difficulty}) calls={rec['llm_calls']}"
            )
    records = [done[c.qid] for c in cases if c.qid in done]
    summary = summarize(records, seed=config.seed)
    summary.update(
        {
            "complete": complete and len(records) == len(cases),
            "n_requested": len(cases),
            "run_id": run_dir.name,
        }
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if records:
        from eval.plots import plot_run

        plot_run(summary, run_dir)
    return summary


def make_ts_scorer(spider_root: Path | None = None) -> Callable[[Case, Prediction], bool | None]:
    """Official test-suite accuracy via taoyds ``exec_eval`` (needs the test-suite DB download)."""
    from eval.spider_loader import SpiderData

    data = SpiderData(spider_root)
    ts_dir = data.test_suite_dir()
    if ts_dir is None:
        raise FileNotFoundError(
            "Test-suite databases missing; run scripts/download_spider.sh --testsuite"
        )
    if str(data.official) not in sys.path:
        sys.path.append(str(data.official))
    import exec_eval

    def scorer(case: Case, pred: Prediction) -> bool | None:
        if not pred.final_code:
            return False
        db_id = case.db_path.parent.name
        target = ts_dir / db_id / f"{db_id}.sqlite"
        if not target.exists():
            return None
        try:
            return bool(
                exec_eval.eval_exec_match(
                    str(target), pred.final_code, case.gold_sql,
                    plug_value=False, keep_distinct=True, progress_bar_for_each_datapoint=False,
                )
            )  # fmt: skip
        except (sqlite3.Error, AssertionError, RuntimeError) as exc:
            log.warning("test-suite eval failed for %s: %s", case.qid, exc)
            return None

    return scorer
