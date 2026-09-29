"""Cluster failures from a run and write a markdown error-analysis report.

Clusters are assigned by comparing the predicted and gold SQL/results (heuristics, applied
in priority order). Anything a heuristic cannot explain is left as ``other`` for manual
review; a human can override any question via ``eval/error_annotations.yaml`` (for example
to mark an ambiguous question or a wrong gold query) and the report merges those in.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sqlglot
import yaml
from sqlglot import exp

from eval.metrics import rows_match

CLUSTERS = {
    "execution_error": "Prediction failed to execute (syntax, unknown column/table, timeout...)",
    "no_answer": "The agent produced no query (refused, asked to clarify, or gave up)",
    "wrong_join": "Different tables or join structure than the gold query",
    "wrong_aggregation": "Different aggregate function or GROUP BY than the gold query",
    "wrong_grouping": "Grouped by a different column (e.g. a non-unique name instead of the key), merging or splitting entities",
    "missing_distinct": "Right rows but duplicated (or extra DISTINCT dropped rows)",
    "value_format_mismatch": "Filter literals differ from the gold ones (case, spelling, format), often giving empty/fewer rows",
    "wrong_columns": "Right rows but different projection (extra, missing or reordered-with-different-columns)",
    "ordering_or_limit": "Same rows but wrong order / LIMIT",
    "ambiguous_or_wrong_gold": "Question is ambiguous or the gold query is questionable (manual annotation)",
    "other": "Needs manual review",
}
_AGGS = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)


def _parse(sql: str | None) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="sqlite") if sql else None
    except sqlglot.errors.SqlglotError:
        return None


def _tables(tree: exp.Expression | None) -> set[str]:
    return {t.name.lower() for t in tree.find_all(exp.Table)} if tree else set()


def _aggs(tree: exp.Expression | None) -> Counter[str]:
    return Counter(type(n).__name__ for n in tree.find_all(*_AGGS)) if tree else Counter()


def _group_columns(tree: exp.Expression | None) -> set[str]:
    group = tree.args.get("group") if tree else None
    if not group:
        return set()
    return {c.name.lower() for g in group.expressions for c in g.find_all(exp.Column)}


def _literals(tree: exp.Expression | None) -> set[str]:
    return {str(n.this) for n in tree.find_all(exp.Literal) if n.is_string} if tree else set()


def _norm_lit(s: str) -> str:
    return re.sub(r"[\s_%-]+", "", s.lower())


def classify(rec: dict[str, Any], gold_rows: list[list[Any]] | None = None) -> str:
    """Assign one failure cluster to a failed record."""
    gold, pred = _parse(rec.get("gold_sql")), _parse(rec.get("pred_code"))
    if rec.get("status") in {"refused", "clarify"} or not rec.get("pred_code"):
        return "no_answer"
    if rec.get("final_error") or rec.get("status") == "failed":
        return "execution_error"
    rows = rec.get("final_rows")
    if gold is not None and pred is not None and gold_rows is not None and rows is not None:
        if (
            len(rows) == len(gold_rows)
            and rows_match(rows, gold_rows, False)
            and not rows_match(rows, gold_rows, True)
        ):
            return "ordering_or_limit"
        as_set = {json.dumps(r, default=str) for r in rows} == {
            json.dumps(r, default=str) for r in gold_rows
        }
        if as_set and len(rows) != len(gold_rows):
            return "missing_distinct"
    if gold is not None and pred is not None:
        if _tables(gold) != _tables(pred) or len(list(gold.find_all(exp.Join))) != len(
            list(pred.find_all(exp.Join))
        ):
            return "wrong_join"
        g_lits, p_lits = _literals(gold), _literals(pred)
        if g_lits != p_lits and {_norm_lit(x) for x in g_lits} == {_norm_lit(x) for x in p_lits}:
            return "value_format_mismatch"
        if g_lits - p_lits and p_lits - g_lits and (not rows):
            return "value_format_mismatch"
        if _aggs(gold) != _aggs(pred) or bool(gold.args.get("group")) != bool(
            pred.args.get("group")
        ):
            return "wrong_aggregation"
        if _group_columns(gold) != _group_columns(pred):
            return "wrong_grouping"
        g_dist, p_dist = bool(gold.args.get("distinct")), bool(pred.args.get("distinct"))
        if g_dist != p_dist or (gold.find(exp.Distinct) is None) != (
            pred.find(exp.Distinct) is None
        ):
            return "missing_distinct"
        if len(gold.expressions) != len(pred.expressions):
            return "wrong_columns"
        if bool(gold.args.get("order")) != bool(pred.args.get("order")) or (
            gold.args.get("limit") is None
        ) != (pred.args.get("limit") is None):
            return "ordering_or_limit"
    return "other"


def load_annotations(path: Path = Path("eval/error_annotations.yaml")) -> dict[str, dict[str, str]]:
    """Manual overrides: ``qid: {cluster: ..., note: ...}``."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def analyse(
    records: list[dict[str, Any]],
    gold_rows: dict[str, list[list[Any]]] | None = None,
    annotations: dict[str, dict[str, str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Cluster every failed record; returns ``cluster -> [record + note]``."""
    annotations = annotations or {}
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("ex"):
            continue
        note = annotations.get(rec["qid"], {})
        cluster = note.get("cluster") or classify(rec, (gold_rows or {}).get(rec["qid"]))
        clusters[cluster].append({**rec, "note": note.get("note", "")})
    return clusters


def _example(rec: dict[str, Any]) -> str:
    lines = [f"**{rec['qid']}** ({rec.get('difficulty', '?')}) — {rec['question']}", ""]
    lines += ["- Gold: `" + " ".join(str(rec.get("gold_sql", "")).split()) + "`"]
    lines += ["- Predicted: `" + " ".join(str(rec.get("pred_code") or "(none)").split()) + "`"]
    if rec.get("final_error"):
        lines += [f"- Error: `{rec['final_error']}`"]
    elif rec.get("final_rows") is not None:
        lines += [f"- Predicted rows (first 3): `{json.dumps(rec['final_rows'][:3], default=str)}`"]
    if rec.get("attempt_errors"):
        lines += [f"- Errors seen while retrying: {', '.join(rec['attempt_errors'])}"]
    if rec.get("note"):
        lines += [f"- Note: {rec['note']}"]
    return "\n".join(lines)


def write_report(
    clusters: dict[str, list[dict[str, Any]]],
    out: Path,
    title: str,
    n_total: int,
    per_cluster: int = 3,
    min_examples: int = 10,
) -> Path:
    """Write the markdown report (counts table + concrete failing examples)."""
    failed = sum(len(v) for v in clusters.values())
    lines = [f"# Error analysis: {title}", "", f"{failed} failures out of {n_total} questions.", ""]
    lines += ["| Cluster | Count | Meaning |", "|---|---:|---|"]
    for name, items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {name} | {len(items)} | {CLUSTERS.get(name, '')} |")
    lines.append("")
    shown = 0
    for name, items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        lines += [f"## {name} ({len(items)})", "", CLUSTERS.get(name, ""), ""]
        for rec in items[:per_cluster]:
            lines += [_example(rec), ""]
            shown += 1
    if shown < min_examples:
        remaining = [r for items in clusters.values() for r in items[per_cluster:]]
        lines += ["## More failing examples", ""]
        for rec in remaining[: min_examples - shown]:
            lines += [_example(rec), ""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    """CLI: ``python -m eval.error_analysis results/<run_id>``."""
    from datasleuth.config import Settings
    from eval.runner import gold_result

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--annotations", type=Path, default=Path("eval/error_annotations.yaml"))
    args = ap.parse_args()
    records = [
        json.loads(x)
        for x in (args.run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    settings = Settings(llm_provider="fake", row_cap=10_000, sql_timeout_s=120)
    from eval.business_suite.build import load_cases as business_cases
    from eval.run_config import custom_cases, spider_cases

    cfg = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    loaders = {
        "spider": lambda: spider_cases(cfg.get("limit"), cfg.get("seed", 0)),
        "business": business_cases,
        "custom": lambda: custom_cases(None),
    }
    cases = {c.qid: c for c in loaders[cfg["suite"]]()}
    gold = {
        q: gold_result(cases[q].db_path, cases[q].gold_sql, settings)
        for q in (r["qid"] for r in records)
        if q in cases and cases[q].gold_sql
    }
    clusters = analyse(records, gold, load_annotations(args.annotations))
    out = write_report(
        clusters, args.run_dir / "error_analysis.md", args.run_dir.name, len(records)
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
