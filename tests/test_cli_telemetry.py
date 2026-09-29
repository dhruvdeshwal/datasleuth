"""CLI behaviour (scripted fake LLM) and the optional tracing hook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from datasleuth import cli, telemetry
from datasleuth.config import Settings
from datasleuth.llm.client import LLM
from datasleuth.llm.providers import FakeProvider

runner = CliRunner()
CSV = "region,amount\nnorth,10\nsouth,20\nnorth,5\n"


def handler_for(sql: str, first_bad: str | None = None) -> Any:
    def handler(system: str, user: str) -> str:
        if "planning stage" in system:
            return json.dumps(
                {"status": "answerable", "question_types": ["aggregation"], "steps": "s"}
            )
        if "careful data analyst" in system:
            bad = first_bad is not None and "REPAIR INSTRUCTION" not in user
            return json.dumps(
                {"language": "sql", "code": first_bad if bad else sql, "rationale": "r"}
            )
        return json.dumps({"answer": "The total is 35."})

    return handler


@pytest.fixture
def patched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    csv = tmp_path / "sales.csv"
    csv.write_text(CSV)

    def fake(settings: Settings) -> LLM:
        s = settings.model_copy(
            update={
                "llm_provider": "fake",
                "use_cache": False,
                "llm_rpm": 0,
                "cache_dir": tmp_path / "c",
                "log_dir": tmp_path / "l",
            }
        )
        return LLM(
            s,
            FakeProvider(handler_for("SELECT SUM(amount) AS total FROM sales")),
            sleep=lambda _s: None,
        )

    monkeypatch.setattr(cli, "make_llm", fake)
    return csv


def test_ask_prints_answer_code_and_table(patched: Path) -> None:
    res = runner.invoke(cli.app, ["ask", "--csv", str(patched), "What is the total?"])
    assert res.exit_code == 0, res.output
    assert (
        "The total is 35." in res.output and "SUM(amount)" in res.output and "total" in res.output
    )
    assert "35" in res.output


def test_ask_json_output_is_a_full_report(patched: Path) -> None:
    res = runner.invoke(cli.app, ["ask", "--csv", str(patched), "--json", "What is the total?"])
    data = json.loads(res.stdout)
    assert (
        data["status"] == "success"
        and data["rows"] == [[35.0]]
        and data["attempts"][0]["result"]["ok"]
    )


def test_ask_shows_self_correction_with_trace(
    patched: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake(settings: Settings) -> LLM:
        s = settings.model_copy(
            update={
                "llm_provider": "fake",
                "use_cache": False,
                "llm_rpm": 0,
                "cache_dir": tmp_path / "c2",
                "log_dir": tmp_path / "l2",
            }
        )
        return LLM(
            s,
            FakeProvider(
                handler_for("SELECT SUM(amount) FROM sales", "SELECT SUM(amout) FROM sales")
            ),
            sleep=lambda _s: None,
        )

    monkeypatch.setattr(cli, "make_llm", fake)
    res = runner.invoke(cli.app, ["ask", "--csv", str(patched), "--trace", "total?"])
    assert res.exit_code == 0
    assert "retry 1" in res.output and "unknown_column" in res.output and "Attempt 2" in res.output


def test_ask_exit_codes_and_errors(
    patched: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def clarify(system: str, user: str) -> str:
        return json.dumps({"status": "clarify", "message": "Which metric?"})

    monkeypatch.setattr(
        cli,
        "make_llm",
        lambda s: LLM(
            s.model_copy(
                update={
                    "llm_provider": "fake",
                    "use_cache": False,
                    "llm_rpm": 0,
                    "cache_dir": tmp_path / "c3",
                    "log_dir": tmp_path / "l3",
                }
            ),
            FakeProvider(clarify),
        ),
    )
    res = runner.invoke(cli.app, ["ask", "--csv", str(patched), "best region?"])
    assert res.exit_code == 2 and "Which metric?" in res.output
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    res = runner.invoke(cli.app, ["ask", "--csv", str(bad), "q"])
    assert res.exit_code == 1 and "Unsupported" in res.output
    assert runner.invoke(cli.app, ["ask", "q"]).exit_code != 0  # no data source given
    assert (
        runner.invoke(cli.app, ["ask", "--sqlite", str(tmp_path / "missing.db"), "q"]).exit_code
        != 0
    )


def test_export_openapi(tmp_path: Path) -> None:
    out = tmp_path / "o.json"
    assert runner.invoke(cli.app, ["export-openapi", str(out)]).exit_code == 0
    assert "/sessions/{session_id}/ask" in json.loads(out.read_text())["paths"]


def test_eval_command_reports_missing_spider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPIDER_DIR", str(tmp_path / "nowhere"))
    res = runner.invoke(cli.app, ["eval", "--config", "configs/zero_shot.yaml", "--limit", "2"])
    assert res.exit_code == 1 and "download_spider" in res.output


def test_telemetry_is_noop_by_default_and_uses_injected_tracer() -> None:
    telemetry.set_tracer(None)
    with telemetry.span("x", a=1) as s:
        assert s is None

    seen: list[tuple[str, dict[str, Any]]] = []

    class FakeSpan:
        def set_attribute(self, k: str, v: Any) -> None:
            seen[-1][1][k] = v

    class FakeTracer:
        def start_as_current_span(self, name: str) -> Any:
            import contextlib

            @contextlib.contextmanager
            def cm() -> Any:
                seen.append((name, {}))
                yield FakeSpan()

            return cm()

    telemetry.set_tracer(FakeTracer())
    try:
        with telemetry.span("datasleuth.node.plan", trace_id="t"):
            pass
        assert seen == [("datasleuth.node.plan", {"trace_id": "t"})]
    finally:
        telemetry.set_tracer(None)


def test_graph_emits_node_and_llm_spans(tmp_path: Path) -> None:
    import sqlite3

    from datasleuth.db.source import DataSource
    from datasleuth.graph import DataSleuthAgent

    names: list[str] = []

    class T:
        def start_as_current_span(self, name: str) -> Any:
            import contextlib

            names.append(name)
            return contextlib.nullcontext(type("S", (), {"set_attribute": lambda *_a: None})())

    db = tmp_path / "d.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t (a INTEGER)")
        c.execute("INSERT INTO t VALUES (1)")
    s = Settings(llm_provider="fake", cache_dir=tmp_path / "c", log_dir=tmp_path / "l", llm_rpm=0, use_cache=False,
                 use_router=False, use_analyst=False, use_chart=False)  # fmt: skip
    llm = LLM(s, FakeProvider(handler_for("SELECT COUNT(*) FROM t")), sleep=lambda _s: None)
    telemetry.set_tracer(T())
    try:
        DataSleuthAgent(DataSource(dialect="sqlite", db_path=db), s, llm=llm).run("count?")
    finally:
        telemetry.set_tracer(None)
    assert "datasleuth.node.execute" in names and "datasleuth.llm" in names
