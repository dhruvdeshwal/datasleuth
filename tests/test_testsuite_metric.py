"""Test-suite accuracy vs plain EX: why plain execution accuracy gives false positives.

Uses a synthetic mini "test suite" (two database instances) with the *official* taoyds
``exec_eval`` code, so it needs no large download.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from eval import download
from eval.metrics import rows_match
from eval.runner import make_ts_scorer
from eval.systems import Case, Prediction

OFFICIAL = Path("eval/spider/test-suite-sql-eval")


def _db(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE singer (name TEXT, age INTEGER)")
        c.executemany("INSERT INTO singer VALUES (?, ?)", rows)


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if not OFFICIAL.exists():
        pytest.skip("official evaluator not downloaded")
    ts = tmp_path / "testsuite_databases" / "music"
    # Original instance: the youngest singer is also the first row / the max-length name.
    _db(ts / "music.sqlite", [("Ann", 20), ("Bo", 35), ("Cy", 41)])
    # Distilled variant that breaks the coincidence.
    _db(ts / "musicv1.sqlite", [("Ann", 50), ("Bo", 35), ("Cy", 21)])
    orig = tmp_path / "spider_data" / "database" / "music" / "music.sqlite"
    _db(orig, [("Ann", 20), ("Bo", 35), ("Cy", 41)])

    class FakeData:
        official = OFFICIAL.resolve()

        def __init__(self, _root: object = None) -> None:
            pass

        def test_suite_dir(self) -> Path:
            return tmp_path / "testsuite_databases"

    monkeypatch.setattr("eval.spider_loader.SpiderData", FakeData)
    return orig, make_ts_scorer(None)


def test_plain_ex_accepts_a_lucky_wrong_query_but_test_suite_rejects_it(suite) -> None:
    orig, scorer = suite
    gold = "SELECT name FROM singer ORDER BY age LIMIT 1"  # the youngest singer
    lucky = "SELECT name FROM singer LIMIT 1"  # just the first row: right only by coincidence
    case = Case("q1", orig, "Who is the youngest singer?", gold, "easy")

    with sqlite3.connect(orig) as c:
        gold_rows = [list(r) for r in c.execute(gold)]
        lucky_rows = [list(r) for r in c.execute(lucky)]
    assert rows_match(lucky_rows, gold_rows, order_matters=True)  # plain EX: false positive!
    assert scorer(case, Prediction(final_code=lucky)) is False  # test suite: caught
    assert scorer(case, Prediction(final_code=gold)) is True  # a truly correct query passes


def test_scorer_handles_missing_prediction_and_missing_db(suite, tmp_path: Path) -> None:
    orig, scorer = suite
    case = Case("q2", orig, "q", "SELECT name FROM singer", "easy")
    assert scorer(case, Prediction(final_code=None)) is False
    other = Case(
        "q3",
        tmp_path / "spider_data" / "database" / "nodb" / "nodb.sqlite",
        "q",
        "SELECT 1",
        "easy",
    )
    assert (
        scorer(other, Prediction(final_code="SELECT 1")) is None
    )  # no suite for this db: not scored


def test_testsuite_extraction_is_bounded_and_keeps_originals() -> None:
    members = [("database/a/a.sqlite", 10), ("database/a/av1.sqlite", 10), ("database/a/av2.sqlite", 10),
               ("database/a/avbig.sqlite", 500 * 1024 * 1024), ("database/b/bv1.sqlite", 5), ("database/b/b.sqlite", 5),
               ("database/zzz/zzz.sqlite", 5), ("__MACOSX/database/a/._a.sqlite", 1)]  # fmt: skip
    chosen = download.plan_testsuite(members, {"a", "b"}, variants=1, max_file_mb=100)
    assert chosen == [
        "database/a/a.sqlite",
        "database/a/av1.sqlite",
        "database/b/b.sqlite",
        "database/b/bv1.sqlite",
    ]
