"""The custom messy-CSV suite: gold SQL on the *ingested messy files* must reproduce the
expected answers computed independently with pandas from the clean data."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from datasleuth.config import Settings
from datasleuth.sandbox.sql_executor import run_sqlite
from datasleuth.sandbox.validator import validate_sql
from eval.custom_suite import build
from eval.metrics import has_order_by, rows_match


@pytest.fixture(scope="module")
def suite(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict], dict[str, Path]]:
    root = tmp_path_factory.mktemp("custom")
    build.build_all(root)
    settings = Settings(llm_provider="fake", data_dir=root)
    qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
    dbs = {name: build.ingest_dataset(name, settings, root) for name in {q["dataset"] for q in qa}}
    return root, qa, dbs


def test_suite_size_and_coverage(suite) -> None:
    _, qa, _ = suite
    per_dataset: dict[str, int] = {}
    for q in qa:
        per_dataset[q["dataset"]] = per_dataset.get(q["dataset"], 0) + 1
    assert len(per_dataset) >= 5 and all(n >= 10 for n in per_dataset.values())
    assert len(qa) >= 70 and len({q["qid"] for q in qa}) == len(qa)
    tags = {t for q in qa for t in q["tags"]}
    for needed in (
        "currency",
        "percent",
        "dates",
        "nulls",
        "duplicate_headers",
        "title_row",
        "leading_zeros",
        "join",
        "mixed_types",
        "unanswerable",
    ):
        assert needed in tags, needed
    assert sum(not q["answerable"] for q in qa) >= 3


def test_gold_sql_reproduces_pandas_expected_answers(suite) -> None:
    _, qa, dbs = suite
    settings = Settings(llm_provider="fake", row_cap=10_000, sql_timeout_s=30)
    failures = []
    for q in qa:
        if not q["answerable"]:
            assert q["gold_sql"] == "" and q["expected"] is None
            continue
        assert validate_sql(q["gold_sql"]).ok, q["qid"]
        res = run_sqlite(dbs[q["dataset"]], q["gold_sql"], settings)
        assert res.ok, (q["qid"], res.error_message)
        order = q["order_matters"] or has_order_by(q["gold_sql"])
        if not rows_match(res.rows, q["expected"], order, lenient=True):
            failures.append((q["qid"], res.rows[:3], q["expected"][:3]))
    assert not failures, failures


def test_expected_answers_are_not_trivial(suite) -> None:
    _, qa, _ = suite
    answers = [json.dumps(q["expected"]) for q in qa if q["answerable"]]
    assert all(a not in ("[]", "[[null]]", "[[0]]") for a in answers)
    assert len(set(answers)) > len(answers) * 0.9


def test_messy_features_survived_ingestion(suite) -> None:
    _, _, dbs = suite
    with closing(sqlite3.connect(dbs["hr_employees"])) as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(hr_employees)")]
        assert cols[:3] == ["employee_id", "full_name", "department"]  # title row skipped
        assert "notes" in cols and "notes_2" in cols  # duplicate headers de-duplicated
    with closing(sqlite3.connect(dbs["inventory"])) as c:
        assert c.execute("SELECT supplier_zip FROM inventory WHERE sku = 'A-007'").fetchone()[
            0
        ] in {"02134", "10001", "60601", "94105", "00501", "33101"}
        assert (
            c.execute("SELECT COUNT(*) FROM inventory WHERE supplier_zip = '02134'").fetchone()[0]
            > 0
        )
        assert (
            c.execute("SELECT typeof(qty) FROM inventory WHERE qty IS NOT NULL LIMIT 1").fetchone()[
                0
            ]
            == "integer"
        )
    with closing(sqlite3.connect(dbs["sales_orders"])) as c:
        assert (
            c.execute(
                "SELECT COUNT(*) FROM sales_orders WHERE order_date NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'"
            ).fetchone()[0]
            == 0
        )
        assert (
            c.execute("SELECT COUNT(DISTINCT region) FROM sales_orders").fetchone()[0] == 4
        )  # whitespace trimmed
    with closing(sqlite3.connect(dbs["survey_responses"])) as c:
        assert (
            c.execute(
                "SELECT COUNT(*) FROM survey_responses WHERE comment LIKE '%Ignore previous instructions%'"
            ).fetchone()[0]
            == 1
        )
        assert (
            c.execute("SELECT COUNT(*) FROM survey_responses").fetchone()[0] == 100
        )  # nothing dropped by quoting/newlines


def test_load_cases_builds_scoreable_cases(suite, tmp_path: Path) -> None:
    root, _, _ = suite
    cases = build.load_cases(root)
    assert len(cases) >= 70
    assert {c.difficulty for c in cases} == {"answerable", "unanswerable"}
    assert all(c.lenient for c in cases)
    assert (root / ".db" / "shop.schema.json").exists()
