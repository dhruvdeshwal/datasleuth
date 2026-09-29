"""Business case study (Northwind): gold SQL must match independently computed pandas answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasleuth.config import Settings
from datasleuth.sandbox.sql_executor import run_sqlite
from datasleuth.sandbox.validator import validate_sql
from eval.business_suite import build
from eval.metrics import has_order_by, rows_match

pytestmark = pytest.mark.skipif(
    not build.DB.exists(), reason="Northwind not downloaded (python -m eval.business_suite.build)"
)


@pytest.fixture(scope="module")
def qa(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    root = tmp_path_factory.mktemp("biz")
    return build.build(root, build.load_frames(build.DB))


def test_suite_shape(qa: list[dict]) -> None:
    assert len(qa) >= 20 and len({q["qid"] for q in qa}) == len(qa)
    assert {q["difficulty"] for q in qa} >= {"easy", "medium", "hard", "unanswerable"}
    assert sum(not q["answerable"] for q in qa) >= 2


def test_gold_sql_matches_independent_pandas_answers(qa: list[dict]) -> None:
    settings = Settings(llm_provider="fake", row_cap=10_000, sql_timeout_s=120)
    for q in filter(lambda x: x["answerable"], qa):
        assert validate_sql(q["gold_sql"]).ok, q["qid"]
        res = run_sqlite(build.DB, q["gold_sql"], settings)
        assert res.ok, (q["qid"], res.error_message)
        order = q["order_matters"] or has_order_by(q["gold_sql"])
        assert rows_match(res.rows, q["expected"], order, lenient=True), (
            q["qid"],
            res.rows[:2],
            q["expected"][:2],
        )


def test_answers_are_not_trivial(qa: list[dict]) -> None:
    for q in filter(lambda x: x["answerable"], qa):
        assert json.dumps(q["expected"]) not in ("[]", "[[null]]", "[[0]]"), q["qid"]


def test_load_cases_points_at_the_database(tmp_path: Path) -> None:
    cases = build.load_cases()
    assert all(c.db_path == build.DB and c.lenient for c in cases)
    assert sum(1 for c in cases if not c.gold_sql) >= 2
