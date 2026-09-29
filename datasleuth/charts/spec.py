"""Validation of LLM-emitted ChartSpecs against the actual result columns/values."""

from __future__ import annotations

from typing import Any

from datasleuth.state import ChartSpec

MAX_SERIES = 8  # the categorical palette has eight validated slots
_NEEDS_Y = {"bar", "line", "scatter", "area", "box", "pie"}


def _is_numeric_column(rows: list[list[Any]], idx: int) -> bool:
    values = [r[idx] for r in rows if r[idx] is not None]
    if not values:
        return False
    numeric = sum(isinstance(v, int | float) and not isinstance(v, bool) for v in values)
    return numeric / len(values) >= 0.8


def canonical_columns(spec: ChartSpec, columns: list[str]) -> ChartSpec:
    """Return the spec with column references snapped to the exact result column names."""
    lookup = {c.lower(): c for c in columns}

    def fix(name: str | None) -> str | None:
        return lookup.get(name.lower(), name) if name else name

    return spec.model_copy(update={"x": fix(spec.x), "y": fix(spec.y), "color": fix(spec.color)})


def validate_chart_spec(spec: ChartSpec, columns: list[str], rows: list[list[Any]]) -> list[str]:
    """Return human-readable problems (empty list means the spec is renderable)."""
    problems: list[str] = []
    lookup = {c.lower(): i for i, c in enumerate(columns)}

    def idx(name: str | None) -> int | None:
        return lookup.get(name.lower()) if name else None

    for role, name in (("x", spec.x), ("y", spec.y), ("color", spec.color)):
        if name and idx(name) is None:
            problems.append(
                f"{role} column '{name}' is not in the result (columns: {', '.join(columns)})"
            )
    if problems:
        return problems
    if not rows:
        return ["the result has no rows to plot"]
    counting = spec.aggregation == "count"
    if spec.chart_type in _NEEDS_Y and not spec.y and not counting:
        problems.append(f"chart type '{spec.chart_type}' needs a y column (or aggregation 'count')")
    y_idx = idx(spec.y)
    if y_idx is not None and not counting and not _is_numeric_column(rows, y_idx):
        problems.append(f"y column '{spec.y}' is not numeric")
    x_idx = idx(spec.x)
    assert x_idx is not None
    if spec.chart_type == "histogram" and not _is_numeric_column(rows, x_idx):
        problems.append(f"histogram x column '{spec.x}' must be numeric")
    if spec.chart_type == "scatter" and not _is_numeric_column(rows, x_idx):
        problems.append(f"scatter x column '{spec.x}' must be numeric")
    if spec.chart_type in {"line", "area"} and len(rows) < 2:
        problems.append("a line/area chart needs at least 2 rows")
    c_idx = idx(spec.color)
    if c_idx is not None and len({r[c_idx] for r in rows}) > MAX_SERIES:
        problems.append(f"color column '{spec.color}' has more than {MAX_SERIES} distinct values")
    if spec.chart_type == "pie" and len({r[x_idx] for r in rows}) > MAX_SERIES:
        problems.append(f"a pie chart supports at most {MAX_SERIES} categories")
    if spec.chart_type in {"bar", "box"} and not spec.top_n and len({r[x_idx] for r in rows}) > 60:
        problems.append("too many categories for a bar/box chart; set top_n")
    return problems
