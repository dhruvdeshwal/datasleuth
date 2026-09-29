"""Execution-accuracy comparison, summary statistics, bootstrap CIs and McNemar's test."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import sqlglot
from scipy import stats

_FLOAT_DECIMALS = 6  # tolerance 1e-6
_MAX_PERMUTATIONS = 5000
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def has_order_by(gold_sql: str) -> bool:
    """True if the gold query's top-level statement has an ORDER BY (row order then matters)."""
    try:
        return sqlglot.parse_one(gold_sql, read="sqlite").args.get("order") is not None
    except sqlglot.errors.SqlglotError:
        return bool(_ORDER_BY_RE.search(gold_sql))


def _norm(value: Any, lenient: bool = False) -> tuple[str, Any]:
    if value is None:
        return ("z", 0)
    if isinstance(value, bool | int | float):
        number = float(value)
        return ("z", 0) if math.isnan(number) else ("n", round(number, _FLOAT_DECIMALS))
    text = str(value)
    if lenient and not (len(text) > 1 and text.startswith("0") and text[1] != "."):
        # numeric strings equal numbers ("2021" == 2021) but never lose a leading zero ("02134")
        try:
            return ("n", round(float(text), _FLOAT_DECIMALS))
        except ValueError:
            pass
    return ("s", text)


def rows_match(
    pred: list[list[Any]], gold: list[list[Any]], order_matters: bool, lenient: bool = False
) -> bool:
    """Compare two result sets the way Execution Accuracy defines equality.

    Multiset of rows (order-insensitive unless ``order_matters``), tolerant to column
    order, floats compared to 1e-6, ``3 == 3.0``, and NULLs equal.
    """
    if not pred and not gold:
        return True
    if len(pred) != len(gold) or len(pred[0]) != len(gold[0]):
        return False
    ncols = len(gold[0])
    p_rows = [tuple(_norm(v, lenient) for v in row) for row in pred]
    g_rows = [tuple(_norm(v, lenient) for v in row) for row in gold]
    g_sig = [sorted(r[i] for r in g_rows) for i in range(ncols)]
    p_sig = [sorted(r[j] for r in p_rows) for j in range(ncols)]
    candidates = [[j for j in range(ncols) if p_sig[j] == g_sig[i]] for i in range(ncols)]
    if any(not c for c in candidates):
        return False
    g_counter = Counter(g_rows)
    for tried, perm in enumerate(product(*candidates)):
        if tried >= _MAX_PERMUTATIONS:
            break
        if len(set(perm)) != ncols:
            continue
        permuted = [tuple(r[j] for j in perm) for r in p_rows]
        if (permuted == g_rows) if order_matters else (Counter(permuted) == g_counter):
            return True
    return False


# ------------------------------------------------------------------- statistics
def bootstrap_ci(
    values: list[float], *, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values`` (fixed seed)."""
    if not values:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict[str, float]:
    """Exact McNemar test on paired per-question correctness (system A vs system B)."""
    if len(a_correct) != len(b_correct):
        raise ValueError("paired samples must have equal length")
    only_a = sum(a and not b for a, b in zip(a_correct, b_correct, strict=True))
    only_b = sum(b and not a for a, b in zip(a_correct, b_correct, strict=True))
    discordant = only_a + only_b
    p = 1.0 if discordant == 0 else float(stats.binomtest(only_a, discordant, 0.5).pvalue)
    return {"a_only": only_a, "b_only": only_b, "p_value": p}


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def _valid_rate(records: list[dict[str, Any]]) -> float:
    """Share of *attempted* queries that executed; refusals/clarifications write no query."""
    attempted = [r for r in records if r.get("attempts", True)]
    return (
        sum(bool(r["first_attempt_valid"]) for r in attempted) / len(attempted)
        if attempted
        else 0.0
    )


def summarize(records: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Aggregate per-question records into the summary reported in the README.

    Each record needs: ``ex``, ``first_attempt_ex``, ``first_attempt_valid``,
    ``retries``, ``attempt_errors``, ``difficulty``, ``latency_ms``, token/cost fields.
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    ex = [1.0 if r["ex"] else 0.0 for r in records]
    lo, hi = bootstrap_ci(ex, seed=seed)
    by_diff: dict[str, dict[str, float]] = {}
    for diff in (
        "easy",
        "medium",
        "hard",
        "extra",
        "answerable",
        "unanswerable",
        "custom",
        "unknown",
    ):
        sub = [r for r in records if r.get("difficulty") == diff]
        if sub:
            by_diff[diff] = {"n": len(sub), "ex": sum(bool(r["ex"]) for r in sub) / len(sub)}
    retried = [r for r in records if r["retries"] > 0]
    fixed = [r for r in retried if not r["first_attempt_ex"]]
    broken = [r for r in retried if r["first_attempt_ex"]]
    errors = Counter(e for r in records for e in r.get("attempt_errors", []) if e)
    lat = sorted(r["latency_ms"] for r in records)
    summary: dict[str, Any] = {
        "n": n,
        "ex": sum(ex) / n,
        "ex_ci95": [lo, hi],
        "ex_by_difficulty": by_diff,
        "valid_sql_rate": _valid_rate(records),
        "first_attempt_ex": sum(bool(r["first_attempt_ex"]) for r in records) / n,
        "final_ex": sum(ex) / n,
        "avg_retries": statistics.fmean(r["retries"] for r in records),
        "retry_success_rate": (sum(bool(r["ex"]) for r in fixed) / len(fixed)) if fixed else None,
        "retry_harm_rate": (sum(not r["ex"] for r in broken) / len(broken)) if broken else None,
        "error_type_distribution": dict(errors.most_common()),
        "latency_ms_mean": statistics.fmean(lat),
        "latency_ms_p50": _pct(lat, 0.5),
        "latency_ms_p95": _pct(lat, 0.95),
        "prompt_tokens_mean": statistics.fmean(r["prompt_tokens"] for r in records),
        "completion_tokens_mean": statistics.fmean(r["completion_tokens"] for r in records),
        "cost_usd_per_question": statistics.fmean(r["cost_usd"] for r in records),
        "llm_calls_mean": statistics.fmean(r["llm_calls"] for r in records),
    }
    groups = sorted({r["group"] for r in records if r.get("group")})
    if groups:
        summary["ex_by_group"] = {
            g: {"n": len(sub_), "ex": sum(bool(r["ex"]) for r in sub_) / len(sub_)}
            for g in groups
            if (sub_ := [r for r in records if r.get("group") == g])
        }
    ts = [r["ts_ex"] for r in records if r.get("ts_ex") is not None]
    if ts:
        t_lo, t_hi = bootstrap_ci([1.0 if v else 0.0 for v in ts], seed=seed)
        summary["ts_ex"] = sum(bool(v) for v in ts) / len(ts)
        summary["ts_ex_ci95"] = [t_lo, t_hi]
        summary["ts_n"] = len(ts)
    return summary
