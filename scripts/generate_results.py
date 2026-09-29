"""Generate every results table/plot from ``results/*/summary.json`` (nothing hand-typed).

Writes ``docs/RESULTS.md`` and replaces the block between the RESULTS markers in README.md.
Run: ``python scripts/generate_results.py [--model gemini-3.1-flash-lite] [--n 25]``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from eval.ablations import paired_tests
from eval.plots import LABELS, LADDER_ORDER, plot_all

RESULTS = Path("results")
README = Path("README.md")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _load(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ci(s: dict[str, Any], key: str = "ex", ci: str = "ex_ci95") -> str:
    lo, hi = s[ci]
    return f"{_pct(s[key])} [{100 * lo:.0f}, {100 * hi:.0f}]"


def find_ladder(model: str, n: int) -> dict[str, Path]:
    """Run directories for the ladder rungs of one model/subset size."""
    slug = model.replace("/", "-").replace(":", "-")
    found: dict[str, Path] = {}
    for name in LADDER_ORDER:
        d = RESULTS / f"{name}__{slug}__n{n}_s0"
        if (d / "summary.json").exists():
            found[name] = d
    return found


def spider_section(model: str, n: int) -> tuple[str, dict[str, dict[str, Any]]]:
    """Markdown for the Spider ablation ladder."""
    dirs = find_ladder(model, n)
    summaries = {k: s for k, d in dirs.items() if (s := _load(d)) and s.get("n")}
    if not summaries:
        return (
            f"_No Spider runs found for `{model}` at n={n}._ Run:\n\n"
            f"```bash\nbash scripts/download_spider.sh\npython -m eval.ablations --model {model} --limit {n}\n"
            "python scripts/generate_results.py\n```\n",
            {},
        )
    rows = ["| # | Configuration | n | EX [95% CI] | Valid SQL | 1st-attempt EX | Avg retries | Retry success | LLM calls/q | $/q* | Latency (s) |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for name in LADDER_ORDER:
        if name not in summaries:
            continue
        s = summaries[name]
        partial = "" if s.get("complete") else " ⚠ partial"
        rows.append(
            f"| {LABELS[name].split(' ', 1)[0]} | {LABELS[name].split(' ', 1)[1]}{partial} | {s['n']} | {_ci(s)} | {_pct(s['valid_sql_rate'])} "
            f"| {_pct(s['first_attempt_ex'])} | {s['avg_retries']:.2f} | {_pct(s.get('retry_success_rate'))} "
            f"| {s['llm_calls_mean']:.1f} | {s['cost_usd_per_question']:.4f} | {s['latency_ms_mean'] / 1000:.1f} |"
        )
    out = [f"**Model:** `{model}` · Spider 1.0 dev, stratified sample of **{n}** of 1,034 questions (seed 0), temperature 0. "
           "\\*$/q uses paid-tier list prices (actual free-tier cost is $0).", "", *rows, ""]  # fmt: skip
    tests = paired_tests({k: dirs[k] for k in dirs if k in summaries})
    if tests:
        out += ["**Paired significance (exact McNemar on per-question correctness):**", "",
                "| A | B | n | only A correct | only B correct | p-value |", "|---|---|---:|---:|---:|---:|"]  # fmt: skip
        out += [
            f"| {t['a']} | {t['b']} | {t['n']} | {t['a_only']} | {t['b_only']} | {t['p_value']:.3f} |"
            for t in tests
        ]
        out += [
            "",
            "With samples this small most adjacent steps are *not* statistically distinguishable; see Limitations.",
            "",
        ]
    diff = [
        "| Config | " + " | ".join(["easy", "medium", "hard", "extra"]) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("zero_shot", "full_schema", "retry_r3", "full"):
        if name in summaries:
            by = summaries[name].get("ex_by_difficulty", {})
            diff.append(
                f"| {LABELS[name]} | "
                + " | ".join(
                    f"{_pct(by[d]['ex'])} (n={by[d]['n']})" if d in by else "-"
                    for d in ("easy", "medium", "hard", "extra")
                )
                + " |"
            )
    out += ["**EX by difficulty:**", "", *diff, ""]
    err = summaries.get("full", {}).get("error_type_distribution") or summaries.get(
        "retry_r3", {}
    ).get("error_type_distribution")
    if err:
        out += [
            "**Error types seen across attempts (best agent run):** "
            + ", ".join(f"`{k}`: {v}" for k, v in err.items()),
            "",
        ]
    if "ts_ex" in summaries.get("full", {}):
        s = summaries["full"]
        out += [
            f"**Test-suite accuracy (official taoyds evaluator), full system:** {_ci(s, 'ts_ex', 'ts_ex_ci95')} on {s['ts_n']} questions.",
            "",
        ]
    plots = plot_all(summaries, RESULTS / f"ablation__{model.replace('/', '-')}__n{n}")
    out += [f"![ablation]({plots[0].as_posix()})", "", f"![by difficulty]({plots[1].as_posix()}) ![retries]({plots[2].as_posix()})", "",
            f"![cost]({plots[3].as_posix()})", ""]  # fmt: skip
    return "\n".join(out), summaries


def custom_section(model: str) -> str:
    """Markdown for the custom messy-CSV suite (any agent/baseline run found)."""
    slug = model.replace("/", "-").replace(":", "-")
    runs = sorted(RESULTS.glob(f"custom_*__{slug}__n*_s0"))
    summaries = [(d, s) for d in runs if (s := _load(d))]
    if not summaries:
        return ("_No custom-suite runs found._ Run:\n\n```bash\ndatasleuth eval --config configs/custom_full.yaml\n"
                "python scripts/generate_results.py\n```\n")  # fmt: skip
    out: list[str] = []
    for d, s in summaries:
        out += [f"**{d.name}** — EX {_ci(s)} on {s['n']} questions · first-attempt EX {_pct(s['first_attempt_ex'])} · "
                f"avg retries {s['avg_retries']:.2f} · retry success {_pct(s.get('retry_success_rate'))}", "",
                "| Dataset | n | EX |", "|---|---:|---:|"]  # fmt: skip
        out += [
            f"| {g} | {v['n']} | {_pct(v['ex'])} |" for g, v in s.get("ex_by_group", {}).items()
        ]
        out += [""]
        report = d / "error_analysis.md"
        if report.exists():
            out += [f"Failure notes: see [{report.as_posix()}]({report.as_posix()}).", ""]
    return "\n".join(out)


def business_section(model: str) -> str:
    """Markdown for the Northwind business case study."""
    slug = model.replace("/", "-").replace(":", "-")
    runs = [(d, s) for d in sorted(RESULTS.glob(f"business_*__{slug}__n*_s0")) if (s := _load(d))]
    if not runs:
        return (
            "_No business runs found._ Run: `datasleuth eval --config configs/business_full.yaml`\n"
        )
    out: list[str] = []
    for d, s in runs:
        status = (
            ""
            if s.get("complete")
            else f" ⚠ partial: {s['n']} of {s['n_requested']} answered so far"
        )
        out += [f"**{d.name}**{status}: EX {_ci(s)} on {s['n']} questions · first-attempt EX {_pct(s['first_attempt_ex'])} · "
                f"valid SQL {_pct(s['valid_sql_rate'])} · {s['llm_calls_mean']:.1f} LLM calls/question", "",
                "| Difficulty | n | EX |", "|---|---:|---:|"]  # fmt: skip
        out += [f"| {k} | {v['n']} | {_pct(v['ex'])} |" for k, v in s["ex_by_difficulty"].items()]
        out += [""]
        if (d / "error_analysis.md").exists():
            out += [
                f"Failure notes: [{(d / 'error_analysis.md').as_posix()}]({(d / 'error_analysis.md').as_posix()}).",
                "",
            ]
    return "\n".join(out)


def build(model: str, n: int) -> str:
    """Full RESULTS.md body."""
    spider, _ = spider_section(model, n)
    return "\n".join(
        [
            "## Business case study: Northwind trading database",
            "",
            business_section(model),
            "## Spider 1.0 dev: ablation ladder",
            "",
            spider,
            "## Custom messy-CSV suite",
            "",
            custom_section(model),
        ]
    )


def headline(model: str) -> str:
    """Compact table for the README: one row per finished benchmark, named by dataset."""
    slug = model.replace("/", "-").replace(":", "-")
    rows: list[tuple[str, dict[str, Any]]] = []
    spider = sorted(
        (d for d in RESULTS.glob(f"full__{slug}__n*_s0") if _load(d)),
        key=lambda d: int(d.name.split("__n")[1].split("_")[0]),
    )
    if spider and (s := _load(spider[-1])) and s.get("n"):
        rows.append(("Spider 1.0 dev (stratified by difficulty)", s))
    business = sorted(RESULTS.glob(f"business_full__{slug}__n*_s0"))
    if business and (s := _load(business[-1])) and s.get("n"):
        rows.append(("Northwind business database", s))
    if not rows:
        return "_Run the evaluations below to populate this table._"
    out = [
        "| Dataset | Questions | Execution accuracy (95% CI) | Valid SQL |",
        "|---|---:|---|---:|",
    ]
    for name, s in rows:
        out.append(f"| {name} | {s['n']} | {_ci(s)} | {_pct(s['valid_sql_rate'])} |")
    for name, s in rows:
        by = " · ".join(f"{k} {_pct(v['ex'])}" for k, v in s["ex_by_difficulty"].items())
        out.append(f"\n{name}: {by}")
    out.append(
        "\nFull tables (ablation, error analysis, per-dataset results): [docs/RESULTS.md](docs/RESULTS.md)"
    )
    return "\n".join(out)


def main() -> None:
    """Regenerate docs/RESULTS.md and the README block."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    body = build(args.model, args.n)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/RESULTS.md").write_text("# Results (generated)\n\n" + body, encoding="utf-8")
    if README.exists():
        text = README.read_text(encoding="utf-8")
        block = f"{START}\n{headline(args.model)}\n{END}"
        text = (
            re.sub(f"{re.escape(START)}.*?{re.escape(END)}", lambda _m: block, text, flags=re.S)
            if START in text
            else text
        )
        README.write_text(text, encoding="utf-8")
    print("wrote docs/RESULTS.md")


if __name__ == "__main__":
    main()
