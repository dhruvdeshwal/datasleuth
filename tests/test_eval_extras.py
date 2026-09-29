"""Ablation ladder, paired stats, error analysis, results generation, plots and an oracle
run over the whole custom suite."""

from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
import yaml

from datasleuth.config import Settings
from datasleuth.llm.client import LLM
from datasleuth.llm.providers import FakeProvider
from eval import ablations, error_analysis
from eval.custom_suite import build
from eval.metrics import summarize
from eval.plots import plot_all, plot_run
from eval.runner import EvalConfig, run_eval
from eval.systems import AttemptLog, Case, Prediction, SingleShotSystem


# ------------------------------------------------------------------- ladder
def test_ladder_has_the_seven_steps_in_order() -> None:
    names = [c.name for c in ablations.ladder("gemini-2.5-flash", 20)]
    assert names == ["zero_shot", "full_schema", "linker", "retry_r1", "retry_r2", "retry_r3",
                     "few_shot", "critic", "full"]  # fmt: skip
    cfgs = {c.name: c for c in ablations.ladder("m", 20)}
    assert cfgs["zero_shot"].minimal_schema and not cfgs["full_schema"].minimal_schema
    assert [
        cfgs[n].settings["max_retries"] for n in ("linker", "retry_r1", "retry_r2", "retry_r3")
    ] == [0, 1, 2, 3]
    assert (
        not cfgs["retry_r3"].settings["use_few_shot"] and cfgs["few_shot"].settings["use_few_shot"]
    )
    assert (
        not cfgs["few_shot"].settings["use_critic_semantic"]
        and cfgs["critic"].settings["use_critic_semantic"]
    )
    assert not cfgs["critic"].settings["use_router"] and cfgs["full"].settings["use_router"]
    assert all(c.settings["llm_model"] == "m" and c.limit == 20 for c in cfgs.values())


def test_each_rung_only_adds_one_thing() -> None:
    cfgs = {c.name: c.settings for c in ablations.ladder("m", 5)}
    diffs = {}
    order = ["linker", "retry_r1", "retry_r2", "retry_r3", "few_shot", "critic", "full"]
    for a, b in pairwise(order):
        diffs[b] = {k for k in cfgs[b] if cfgs[a].get(k) != cfgs[b].get(k)}
    assert diffs["retry_r1"] == {"max_retries"} and diffs["few_shot"] == {"use_few_shot"}
    assert diffs["critic"] == {"use_critic_semantic"} and diffs["full"] == {
        "use_router",
        "allow_pandas",
    }


def test_written_configs_load_back(tmp_path: Path) -> None:
    paths = ablations.write_configs(tmp_path, "gemini-2.5-flash-lite", 12)
    assert len(paths) == 9
    loaded = EvalConfig.load(paths[-1])
    assert (
        loaded.name == "full"
        and loaded.limit == 12
        and loaded.settings["llm_model"] == "gemini-2.5-flash-lite"
    )
    assert yaml.safe_load(paths[0].read_text())["system"] == "single_shot"


def test_committed_configs_are_valid() -> None:
    for path in Path("configs").rglob("*.yaml"):
        cfg = EvalConfig.load(path)
        Settings(**cfg.settings)  # every knob must be a real Settings field


def test_call_estimate_scales_with_n() -> None:
    assert sum(ablations.estimate_calls(50).values()) > sum(ablations.estimate_calls(10).values())


# ------------------------------------------------------------------ pairing
def _write_run(d: Path, flags: dict[str, bool]) -> Path:
    d.mkdir(parents=True)
    (d / "per_question.jsonl").write_text(
        "\n".join(json.dumps({"qid": q, "ex": ok}) for q, ok in flags.items()), encoding="utf-8"
    )
    return d


def test_paired_tests_use_common_questions(tmp_path: Path) -> None:
    a = _write_run(tmp_path / "zero_shot", {f"q{i}": i % 4 == 0 for i in range(40)})
    b = _write_run(
        tmp_path / "full_schema", {f"q{i}": i % 2 == 0 for i in range(30)}
    )  # subset overlap only
    rows = ablations.paired_tests({"zero_shot": a, "full_schema": b})
    assert rows[0]["n"] == 30 and rows[0]["b_only"] > rows[0]["a_only"]


# ----------------------------------------------------------- error analysis
def rec(qid: str, gold: str, pred: str | None, rows: Any = None, **kw: Any) -> dict[str, Any]:
    return {"qid": qid, "question": f"question {qid}", "gold_sql": gold, "pred_code": pred, "ex": False,
            "status": "success", "final_rows": rows, "difficulty": "medium", **kw}  # fmt: skip


def test_failure_clusters() -> None:
    cases = [
        (
            rec(
                "e1",
                "SELECT a FROM t",
                "SELECT nope FROM t",
                final_error="no such column",
                status="failed",
            ),
            None,
            "execution_error",
        ),
        (rec("n1", "SELECT a FROM t", None, status="refused"), None, "no_answer"),
        (
            rec("j1", "SELECT a FROM t JOIN u ON t.id=u.id", "SELECT a FROM t", [[1]]),
            [[2]],
            "wrong_join",
        ),
        (
            rec("a1", "SELECT count(*) FROM t", "SELECT sum(x) FROM t", [[3]]),
            [[4]],
            "wrong_aggregation",
        ),
        (
            rec("d1", "SELECT a FROM t", "SELECT a FROM t", [[1], [1], [2]]),
            [[1], [2]],
            "missing_distinct",
        ),
        (
            rec(
                "v1",
                "SELECT a FROM t WHERE city = 'Paris'",
                "SELECT a FROM t WHERE city = 'paris'",
                [],
            ),
            [[1]],
            "value_format_mismatch",
        ),
        (
            rec("o1", "SELECT a FROM t ORDER BY a", "SELECT a FROM t ORDER BY a DESC", [[2], [1]]),
            [[1], [2]],
            "ordering_or_limit",
        ),
        (rec("c1", "SELECT a FROM t", "SELECT a, b FROM t", [[1, 2]]), [[9]], "wrong_columns"),
    ]
    for record, gold_rows, expected in cases:
        assert error_analysis.classify(record, gold_rows) == expected, record["qid"]


def test_annotations_override_and_report(tmp_path: Path) -> None:
    records = [
        rec(
            f"q{i}",
            "SELECT a FROM t",
            "SELECT nope FROM t",
            final_error="no such column",
            status="failed",
        )
        for i in range(12)
    ]
    records.append({**rec("ok", "SELECT 1", "SELECT 1"), "ex": True})
    clusters = error_analysis.analyse(
        records,
        annotations={"q3": {"cluster": "ambiguous_or_wrong_gold", "note": "gold ignores ties"}},
    )
    assert len(clusters["ambiguous_or_wrong_gold"]) == 1 and len(clusters["execution_error"]) == 11
    out = error_analysis.write_report(clusters, tmp_path / "r.md", "unit", n_total=13)
    text = out.read_text()
    assert "12 failures out of 13" in text and "gold ignores ties" in text
    assert text.count("**q") >= 10  # at least ten concrete examples


# ------------------------------------------------------------------- plots
def _summary(ex: float) -> dict[str, Any]:
    return {"n": 20, "ex": ex, "ex_ci95": [ex - 0.1, ex + 0.1], "cost_usd_per_question": 0.001, "run_id": "x",
            "ex_by_difficulty": {d: {"n": 5, "ex": ex} for d in ("easy", "medium", "hard", "extra")}}  # fmt: skip


def test_plots_render(tmp_path: Path) -> None:
    summaries = {
        n: _summary(0.5 + i / 20)
        for i, n in enumerate(
            [
                "zero_shot",
                "full_schema",
                "linker",
                "retry_r1",
                "retry_r2",
                "retry_r3",
                "few_shot",
                "critic",
                "full",
            ]
        )
    }
    for p in plot_all(summaries, tmp_path):
        assert p.exists() and p.stat().st_size > 5000
    assert plot_run(summaries["full"], tmp_path) is not None
    assert plot_run({"n": 1}, tmp_path) is None


def test_generate_results_from_summaries_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(Path("scripts").resolve()))
    import generate_results as gr

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gr, "RESULTS", tmp_path / "results")
    empty = gr.build("gemini-2.5-flash", 5)
    assert (
        "No Spider runs found" in empty and "No custom-suite runs found" in empty
    )  # never invents numbers
    recs = [
        {
            "ex": True,
            "first_attempt_ex": True,
            "first_attempt_valid": True,
            "retries": 0,
            "attempt_errors": [],
            "difficulty": "easy",
            "latency_ms": 5.0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cost_usd": 0.0,
            "llm_calls": 1,
        }
    ] * 4
    for name in ("zero_shot", "full_schema"):
        d = tmp_path / "results" / f"{name}__gemini-2.5-flash__n5_s0"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({**summarize(recs), "complete": True}))
        (d / "per_question.jsonl").write_text(
            "\n".join(json.dumps({"qid": f"q{i}", "ex": True}) for i in range(4))
        )
    body = gr.build("gemini-2.5-flash", 5)
    assert (
        "| 1 | zero-shot | 4 | 100.0% [100, 100]" in body
        and "McNemar" in body
        and "cost_vs_accuracy.png" in body
    )


# --------------------------------------------- oracle over the entire custom suite
def test_oracle_agent_scores_perfectly_on_custom_suite(tmp_path: Path) -> None:
    from eval.agent_system import AgentSystem

    settings_kw = {"cache_dir": str(tmp_path / "c"), "log_dir": str(tmp_path / "l"), "llm_provider": "fake",
                   "llm_rpm": 0, "use_cache": False, "use_analyst": False, "use_chart": False,
                   "use_schema_linker": False, "use_critic_semantic": False}  # fmt: skip
    cfg = EvalConfig(name="custom_oracle", system="agent", suite="custom", settings=settings_kw)
    settings = cfg.make_settings()
    cases = build.load_cases(tmp_path / "suite", settings)
    gold = {c.question: c.gold_sql for c in cases}

    def oracle(system: str, user: str) -> str:
        question = user.rsplit("Question: ", 1)[1].strip()
        if "planning stage" in system:
            if not gold[question]:
                return json.dumps(
                    {"status": "refuse", "message": "That information is not in the data."}
                )
            return json.dumps({"status": "answerable", "question_types": ["other"], "steps": "s"})
        return json.dumps({"language": "sql", "code": gold[question], "rationale": ""})

    llm = LLM(settings, FakeProvider(oracle), sleep=lambda _: None)
    summary = run_eval(cfg, cases, lambda l, s: AgentSystem(l, s), settings=settings, llm=llm,
                       results_dir=tmp_path / "res", out=lambda _: None)  # fmt: skip
    assert summary["n"] == len(cases) >= 80 and summary["ex"] == 1.0, summary["ex_by_group"]
    assert set(summary["ex_by_group"]) >= {"sales_orders", "hr_employees", "shop"}
    assert (tmp_path / "res" / summary["run_id"] / "ex_breakdown.png").exists()


def test_unanswerable_scoring_and_wrong_answers_fail(tmp_path: Path) -> None:
    from eval.runner import score

    settings = Settings(llm_provider="fake")
    case = Case("u1", tmp_path / "x.sqlite", "avg income?", "", "unanswerable")
    assert score(case, Prediction(status="refused"), settings)["ex"] is True
    assert score(case, Prediction(status="success"), settings)["ex"] is False
    _ = (AttemptLog, SingleShotSystem)


def test_grouping_key_mistake_is_its_own_cluster() -> None:
    gold = "SELECT c.name, SUM(o.t) FROM o JOIN c ON o.cid = c.id GROUP BY c.id"
    pred = "SELECT c.name, SUM(o.t) FROM o JOIN c ON o.cid = c.id GROUP BY c.name"
    assert (
        error_analysis.classify(rec("g1", gold, pred, [["IT", 9.0]]), [["A", 5.0]])
        == "wrong_grouping"
    )
