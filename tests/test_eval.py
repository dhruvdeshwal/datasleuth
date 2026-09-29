"""Eval harness: comparator, sampling, runner (resume/quota) and real-Spider oracle checks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from datasleuth.config import Settings
from datasleuth.llm.base import LLMRateLimitError
from datasleuth.llm.client import LLM
from datasleuth.llm.providers import FakeProvider
from datasleuth.nodes.query_writer import WriterOptions
from datasleuth.sandbox.validator import validate_sql
from eval.metrics import bootstrap_ci, has_order_by, mcnemar, rows_match, summarize
from eval.runner import EvalConfig, run_eval
from eval.spider_loader import SpiderData, SpiderExample, SpiderNotFound, stratified_sample
from eval.systems import Case, SingleShotSystem

# ---------------------------------------------------------------- comparator


def test_rows_match_is_order_insensitive_by_default() -> None:
    assert rows_match([[1, "a"], [2, "b"]], [[2, "b"], [1, "a"]], order_matters=False)
    assert not rows_match([[1, "a"], [2, "b"]], [[2, "b"], [1, "a"]], order_matters=True)


def test_rows_match_tolerates_column_order_and_floats() -> None:
    assert rows_match([["a", 1.0000000001]], [[1, "a"]], order_matters=False)
    assert not rows_match([["a", 1.001]], [[1, "a"]], order_matters=False)


def test_rows_match_is_multiset_not_set() -> None:
    assert not rows_match([[1], [1], [2]], [[1], [2], [2]], order_matters=False)
    assert not rows_match([[1], [1]], [[1]], order_matters=False)


def test_rows_match_nulls_and_empty() -> None:
    assert rows_match([[None]], [[None]], False)
    assert rows_match([], [], False)
    assert not rows_match([], [[1]], False)
    assert not rows_match([[1, 2]], [[1]], False)
    assert not rows_match([[1]], [["1"]], False)


def test_has_order_by_top_level_only() -> None:
    assert has_order_by("SELECT a FROM t ORDER BY a")
    assert not has_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a) x")
    assert not has_order_by("SELECT a FROM t")


def test_bootstrap_and_mcnemar() -> None:
    lo, hi = bootstrap_ci([1.0] * 70 + [0.0] * 30, seed=1)
    assert lo < 0.7 < hi and hi - lo < 0.25
    res = mcnemar([True] * 20 + [False] * 5, [False] * 20 + [True] * 5)
    assert res["a_only"] == 20 and res["b_only"] == 5 and res["p_value"] < 0.05
    assert mcnemar([True, False], [True, False])["p_value"] == 1.0


def _rec(ex: bool, first: bool, retries: int = 0, diff: str = "easy") -> dict:
    return {
        "ex": ex, "first_attempt_ex": first, "first_attempt_valid": True, "retries": retries,
        "attempt_errors": ["unknown_column"] if retries else [], "difficulty": diff,
        "latency_ms": 100.0, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.01,
        "llm_calls": 1 + retries,
    }  # fmt: skip


def test_summarize_retry_metrics() -> None:
    recs = [
        _rec(True, True),
        _rec(True, False, 1),
        _rec(False, False, 2),
        _rec(False, True, 1, "hard"),
    ]
    s = summarize(recs)
    assert s["ex"] == 0.5 and s["first_attempt_ex"] == 0.5
    assert s["retry_success_rate"] == 0.5  # 1 of 2 first-attempt failures fixed
    assert s["retry_harm_rate"] == 1.0  # the one first-attempt success was broken
    assert s["ex_by_difficulty"]["hard"]["n"] == 1
    assert s["error_type_distribution"] == {"unknown_column": 3}


# ------------------------------------------------------------------ sampling


def test_stratified_sample_is_proportional_and_seeded() -> None:
    exs = [
        SpiderExample(
            f"q{i}", "d", "?", "s", ("easy", "medium", "hard", "extra")[i % 4 if i < 40 else 0]
        )
        for i in range(100)
    ]
    a = stratified_sample(exs, 20, seed=3)
    assert a == stratified_sample(exs, 20, seed=3)
    assert len(a) == 20
    assert {e.difficulty for e in a} == {"easy", "medium", "hard", "extra"}
    assert stratified_sample(exs, 500, seed=0) == exs


# -------------------------------------------------------------------- runner


@pytest.fixture
def toy(tmp_path: Path) -> tuple[Path, list[Case]]:
    db = tmp_path / "toy.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)")
        conn.executemany(
            "INSERT INTO t (name, score) VALUES (?, ?)", [("a", 1), ("b", 2), ("c", 3)]
        )
    cases = [
        Case("q1", db, "How many rows?", "SELECT COUNT(*) FROM t", "easy"),
        Case(
            "q2",
            db,
            "Highest score name?",
            "SELECT name FROM t ORDER BY score DESC LIMIT 1",
            "medium",
        ),
    ]
    return db, cases


def _llm(tmp_path: Path, handler) -> tuple[LLM, Settings]:
    s = Settings(
        llm_provider="fake",
        cache_dir=tmp_path / "c" / "llm",
        log_dir=tmp_path / "l",
        llm_rpm=0,
        row_cap=1000,
    )
    return LLM(s, FakeProvider(handler), sleep=lambda _: None), s


def _reply(sql: str) -> str:
    return json.dumps({"language": "sql", "code": sql, "rationale": "x"})


def test_runner_scores_resumes_and_writes_summary(tmp_path: Path, toy) -> None:
    _, cases = toy
    answers = {
        "How many rows?": "SELECT COUNT(*) FROM t",
        "Highest score name?": "SELECT name FROM t",
    }

    def handler(_s: str, user: str) -> str:
        return _reply(next(v for k, v in answers.items() if k in user))

    llm, settings = _llm(tmp_path, handler)
    cfg = EvalConfig(name="unit", system="single_shot")
    factory = lambda l, s: SingleShotSystem(
        l, s, WriterOptions(allow_pandas=False, force_language="sql")
    )
    summary = run_eval(
        cfg,
        cases,
        factory,
        settings=settings,
        llm=llm,
        results_dir=tmp_path / "res",
        out=lambda _: None,
    )
    assert summary["n"] == 2 and summary["ex"] == 0.5 and summary["complete"]
    run_dir = tmp_path / "res" / summary["run_id"]
    assert len((run_dir / "per_question.jsonl").read_text().splitlines()) == 2
    provider = llm.provider
    assert isinstance(provider, FakeProvider)
    calls_before = len(provider.calls)
    run_eval(
        cfg,
        cases,
        factory,
        settings=settings,
        llm=llm,
        results_dir=tmp_path / "res",
        out=lambda _: None,
    )
    assert len(provider.calls) == calls_before  # fully resumed, no new LLM calls


def test_runner_stops_cleanly_on_quota(tmp_path: Path, toy) -> None:
    _, cases = toy
    llm, settings = _llm(tmp_path, lambda s, u: _reply("SELECT COUNT(*) FROM t"))
    settings.llm_daily_limit = 1
    llm = LLM(settings, llm.provider, sleep=lambda _: None)
    cfg = EvalConfig(name="quota", system="single_shot")
    factory = lambda l, s: SingleShotSystem(
        l, s, WriterOptions(allow_pandas=False, force_language="sql")
    )
    messages: list[str] = []
    # Distinct prompts so the second call is uncached and hits the daily limit.
    summary = run_eval(
        cfg,
        cases,
        factory,
        settings=settings,
        llm=llm,
        results_dir=tmp_path / "res",
        out=messages.append,
    )
    assert summary["n"] == 1 and summary["complete"] is False
    assert any("Re-run" in m for m in messages)


def test_runner_aborts_after_repeated_llm_errors(tmp_path: Path, toy) -> None:
    _, cases = toy
    llm, settings = _llm(tmp_path, lambda s, u: "x")

    def boom(*_a, **_k):
        raise LLMRateLimitError("429")

    llm.provider.complete = boom  # type: ignore[method-assign]
    cfg = EvalConfig(name="err", system="single_shot")
    factory = lambda l, s: SingleShotSystem(
        l, s, WriterOptions(allow_pandas=False, force_language="sql")
    )
    summary = run_eval(
        cfg,
        cases * 3,
        factory,
        settings=settings,
        llm=llm,
        results_dir=tmp_path / "res",
        out=lambda _: None,
    )
    assert summary["complete"] is False and summary["n"] == 0


# ------------------------------------------------------- real Spider (optional)


@pytest.fixture(scope="module")
def spider() -> SpiderData:
    try:
        return SpiderData()
    except SpiderNotFound:
        pytest.skip("Spider not downloaded")


def test_validator_accepts_every_spider_gold_query(spider: SpiderData) -> None:
    rejected = [
        (e.qid, validate_sql(e.gold_sql).message)
        for e in spider.dev()
        if not validate_sql(e.gold_sql).ok
    ]
    assert not rejected, rejected[:5]


def test_spider_difficulty_distribution_matches_official(spider: SpiderData) -> None:
    from collections import Counter

    counts = Counter(e.difficulty for e in spider.dev())
    assert sum(counts.values()) == 1034
    assert counts == {"easy": 248, "medium": 446, "hard": 174, "extra": 166}


def test_oracle_llm_scores_perfectly_on_real_spider(tmp_path: Path, spider: SpiderData) -> None:
    from eval.run_config import spider_cases

    cases = spider_cases(40, seed=1, root=spider.root)
    gold = {c.question: c.gold_sql for c in cases}

    def oracle(_s: str, user: str) -> str:
        question = user.rsplit("Question: ", 1)[1].strip()
        return _reply(gold[question])

    llm, settings = _llm(tmp_path, oracle)
    settings.row_cap, settings.sql_timeout_s = 10_000, 30.0
    cfg = EvalConfig(name="oracle", system="single_shot")
    factory = lambda l, s: SingleShotSystem(
        l, s, WriterOptions(allow_pandas=False, force_language="sql")
    )
    summary = run_eval(
        cfg,
        cases,
        factory,
        settings=settings,
        llm=llm,
        results_dir=tmp_path / "res",
        out=lambda _: None,
    )
    assert summary["ex"] == 1.0, summary


def test_valid_sql_rate_ignores_refusals() -> None:
    answered = _rec(True, True)
    refused = {**_rec(True, True), "attempts": [], "first_attempt_valid": False}
    assert summarize([answered, refused])["valid_sql_rate"] == 1.0
