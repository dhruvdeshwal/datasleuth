"""Command line interface: ``datasleuth ask | eval | serve | export-openapi``."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from datasleuth.config import Settings, get_settings
from datasleuth.db.connectors import ConnectionRejected, external_engine
from datasleuth.db.ingest import UploadError, ingest_files
from datasleuth.db.source import DataSource
from datasleuth.graph import DataSleuthAgent
from datasleuth.llm.base import LLMError
from datasleuth.llm.client import LLM
from datasleuth.state import Report, SchemaInfo

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="DataSleuth: ask questions of your data."
)
out = Console()
err = Console(stderr=True)


def make_llm(settings: Settings) -> LLM:
    """Build the LLM (a seam that tests replace with a scripted fake)."""
    return LLM(settings)


def _settings(model: str | None, provider: str | None) -> Settings:
    overrides: dict[str, Any] = {}
    if model:
        overrides["llm_model"] = model
    if provider:
        overrides["llm_provider"] = provider
    return get_settings(**overrides)


def _source(
    files: list[Path], sqlite: Path | None, db_url: str | None, settings: Settings, workdir: Path
) -> tuple[DataSource, SchemaInfo | None]:
    if db_url:
        engine, dialect = external_engine(db_url)
        return DataSource(dialect=dialect, engine=engine), None
    if sqlite:
        if not sqlite.exists():
            raise typer.BadParameter(f"SQLite file not found: {sqlite}")
        return DataSource(dialect="sqlite", db_path=sqlite), None
    if not files:
        raise typer.BadParameter("Provide --csv/--xlsx files, --sqlite FILE or --db URL.")
    db = workdir / "data.sqlite"
    schema = ingest_files([(p, p.name) for p in files], db, settings)
    return DataSource(dialect="sqlite", db_path=db), schema


def _progress(event: dict[str, Any]) -> None:
    if event["type"] != "node_end":
        return
    d = event.get("detail", {})
    note = ""
    if event["node"] == "execute" and not d.get("ok", True):
        note = f" [red]failed ({d.get('error_type')})[/red]"
    elif event["node"] == "critic" and d.get("status") == "running":
        note = f" [yellow]retry {d.get('retry_count')}: {d.get('error_type')}[/yellow]"
    elif event["node"] == "write":
        note = f" attempt {d.get('attempt')} ({d.get('language')})"
    err.print(f"[dim]▸ {event['label']}[/dim]{note}")


def _render(report: Report, show_trace: bool, max_rows: int = 20) -> None:
    if report.answer:
        out.print(
            Panel(
                report.answer,
                title="Answer",
                border_style="green" if report.status == "success" else "yellow",
            )
        )
    if report.code and report.language:
        out.print(
            Syntax(
                report.code,
                "sql" if report.language == "sql" else "python",
                theme="ansi_dark",
                word_wrap=True,
            )
        )
    if report.status == "success" and report.rows:
        table = Table(show_lines=False)
        for c in report.columns:
            table.add_column(c)
        for row in report.rows[:max_rows]:
            table.add_row(*["NULL" if v is None else str(v) for v in row])
        out.print(table)
        if report.row_count > max_rows or report.truncated:
            out.print(
                f"[dim]{report.row_count} row(s){' (truncated)' if report.truncated else ''}; showing {min(max_rows, report.row_count)}[/dim]"
            )
    for caveat in report.caveats:
        out.print(f"[yellow]⚠ {caveat}[/yellow]")
    if show_trace:
        for a in report.attempts:
            status = (
                "ok"
                if a.result and a.result.ok
                else f"error: {a.result.error_type if a.result else 'n/a'}"
            )
            out.print(f"\n[bold]Attempt {a.index}[/bold] ({a.language}) → {status}")
            out.print(Syntax(a.code, "sql" if a.language == "sql" else "python", theme="ansi_dark"))
            if a.diagnosis:
                out.print(
                    f"[yellow]diagnosis {a.diagnosis.error_type}: {a.diagnosis.repair_instruction}[/yellow]"
                )
    m = report.metrics
    out.print(
        f"[dim]{m.llm_calls} LLM calls · {m.prompt_tokens + m.completion_tokens} tokens · ~${m.cost_usd:.4f} (list price) · {m.latency_ms / 1000:.1f}s · {m.retries} retries[/dim]"
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question in plain English.")],
    csv: Annotated[list[Path], typer.Option("--csv", help="CSV/TSV file (repeatable).")] = [],  # noqa: B006
    xlsx: Annotated[list[Path], typer.Option("--xlsx", help="Excel file (repeatable).")] = [],  # noqa: B006
    sqlite: Annotated[
        Path | None, typer.Option(help="Query an existing SQLite database (read-only).")
    ] = None,
    db: Annotated[
        str | None,
        typer.Option(help="External DB URL (postgresql://, mysql://). Use read-only credentials."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Override DATASLEUTH_LLM_MODEL.")] = None,
    provider: Annotated[
        str | None, typer.Option(help="gemini | anthropic | openai | ollama")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full report as JSON.")] = False,
    trace: Annotated[bool, typer.Option(help="Show every attempt, error and diagnosis.")] = False,
    chart: Annotated[
        Path | None, typer.Option(help="Save the chart (if any) as a PNG here.")
    ] = None,
) -> None:
    """Ask a question of CSV/XLSX files or a database. Example:

    datasleuth ask --csv sales.csv "Which region had the highest revenue growth in Q3?"
    """
    settings = _settings(model, provider)
    workdir = Path(tempfile.mkdtemp(prefix="datasleuth_"))
    try:
        source, schema = _source([*csv, *xlsx], sqlite, db, settings, workdir)
        agent = DataSleuthAgent(
            source,
            settings,
            llm=make_llm(settings),
            schema=schema,
            artifacts_root=workdir / "traces",
        )
        report = agent.run(question, on_event=None if as_json else _progress)
        if chart and report.has_chart_png:
            png = workdir / "traces" / report.trace_id / "chart.png"
            shutil.copyfile(png, chart)
            err.print(f"[dim]chart saved to {chart}[/dim]")
        if as_json:
            typer.echo(report.model_dump_json(indent=2))
        else:
            _render(report, trace)
    except (UploadError, ConnectionRejected) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except LLMError as exc:
        err.print(f"[red]LLM problem: {exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    raise typer.Exit(
        0 if report.status == "success" else 2 if report.status in {"clarify", "refused"} else 1
    )


@app.command(name="eval")
def eval_(
    config: Annotated[Path, typer.Option(help="YAML config, e.g. configs/full.yaml")],
    limit: Annotated[
        int | None, typer.Option(help="Stratified sample size (default: the config's).")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Sampling seed.")] = None,
    ts: Annotated[
        bool, typer.Option("--ts", help="Also compute official test-suite accuracy.")
    ] = False,
) -> None:
    """Run an evaluation config on Spider or the custom messy-CSV suite (resumable, cached)."""
    try:
        from eval.run_config import run_config
    except ImportError as exc:  # pragma: no cover - dev extras missing
        err.print(
            f"[red]Evaluation extras are missing ({exc.name}). Install: pip install -e '.[eval]'[/red]"
        )
        raise typer.Exit(1) from exc
    try:
        summary = run_config(
            config, limit=limit, seed=seed, ts=True if ts else None, out=lambda s: err.print(s)
        )
    except FileNotFoundError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    table = Table(title=summary.get("run_id", "run"))
    table.add_column("metric")
    table.add_column("value")
    lo, hi = summary.get("ex_ci95", [0, 0])
    rows = [("questions", f"{summary['n']} of {summary['n_requested']}{'' if summary['complete'] else ' (partial: re-run to resume)'}"),
            ("execution accuracy", f"{100 * summary['ex']:.1f}%  [95% CI {100 * lo:.0f}-{100 * hi:.0f}]"),
            ("first-attempt EX", f"{100 * summary['first_attempt_ex']:.1f}%"), ("valid SQL rate", f"{100 * summary['valid_sql_rate']:.1f}%"),
            ("avg retries", f"{summary['avg_retries']:.2f}"), ("LLM calls / question", f"{summary['llm_calls_mean']:.1f}"),
            ("est. $ / question", f"{summary['cost_usd_per_question']:.4f}")]  # fmt: skip
    for k, v in rows:
        table.add_row(k, v)
    out.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload for development.")] = False,
) -> None:
    """Start the API (and the built frontend, if frontend/dist exists)."""
    import uvicorn

    if reload:
        uvicorn.run("datasleuth.api.factory:app", host=host, port=port, reload=True, factory=False)
    else:
        from datasleuth.api.app import create_app

        uvicorn.run(create_app(), host=host, port=port)


@app.command(name="export-openapi")
def export_openapi(path: Annotated[Path, typer.Argument()] = Path("docs/openapi.json")) -> None:
    """Write the OpenAPI schema used to generate the frontend client."""
    from datasleuth.api.app import create_app

    with tempfile.TemporaryDirectory() as tmp:
        schema = create_app(
            Settings(data_dir=Path(tmp), frontend_dist=Path(tmp) / "none")
        ).openapi()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    out.print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
