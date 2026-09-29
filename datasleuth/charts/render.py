"""Deterministic Plotly rendering of a validated ChartSpec. The LLM never runs plot code."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datasleuth.charts.spec import canonical_columns
from datasleuth.state import ChartSpec

log = logging.getLogger(__name__)

# Validated categorical palette (light-surface steps, fixed order).
COLORWAY = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_INK, _INK_2, _GRID, _AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
_AGG = {"sum": "sum", "mean": "mean", "min": "min", "max": "max"}


def _prepare(
    spec: ChartSpec, columns: list[str], rows: list[list[Any]]
) -> tuple[pd.DataFrame, str | None]:
    """Apply aggregation, sorting and top-N. Returns the frame and the y column to plot."""
    df = pd.DataFrame(rows, columns=columns)
    y = spec.y
    keys = [c for c in (spec.x, spec.color) if c]
    if spec.aggregation != "none" and spec.chart_type != "histogram":
        if spec.aggregation == "count" or not y:
            df = df.groupby(keys, dropna=False, sort=False).size().reset_index(name="count")
            y = "count"
        else:
            df[y] = pd.to_numeric(df[y], errors="coerce")
            df = (
                df.groupby(keys, dropna=False, sort=False)[y]
                .agg(_AGG[spec.aggregation])
                .reset_index()
            )
    if y and spec.chart_type not in {"histogram", "scatter"}:
        df[y] = pd.to_numeric(df[y], errors="coerce")
    if spec.chart_type in {"line", "area"}:
        df = df.sort_values(spec.x, kind="stable")
    elif spec.sort != "none" and y and spec.chart_type in {"bar", "pie", "box"}:
        df = df.sort_values(y, ascending=spec.sort == "asc", kind="stable")
    if spec.top_n and spec.chart_type in {"bar", "pie", "box"}:
        if y and spec.sort == "none":
            df = df.sort_values(y, ascending=False, kind="stable")
        df = df.head(spec.top_n)
    return df, y


def render_chart(spec: ChartSpec, columns: list[str], rows: list[list[Any]]) -> go.Figure:
    """Build the Plotly figure for ``spec`` (caller must have validated it)."""
    spec = canonical_columns(spec, columns)
    df, y = _prepare(spec, columns, rows)
    labels = {spec.x: spec.x_label or spec.x}
    if y:
        labels[y] = spec.y_label or y
    common: dict[str, Any] = {
        "color": spec.color,
        "labels": labels,
        "color_discrete_sequence": COLORWAY,
    }
    kind = spec.chart_type
    if kind == "bar":
        fig = px.bar(df, x=spec.x, y=y, **common)
    elif kind == "line":
        fig = px.line(df, x=spec.x, y=y, markers=True, **common)
    elif kind == "area":
        fig = px.area(df, x=spec.x, y=y, **common)
    elif kind == "scatter":
        fig = px.scatter(df, x=spec.x, y=y, **common)
    elif kind == "pie":
        fig = px.pie(df, names=spec.x, values=y, color_discrete_sequence=COLORWAY)
    elif kind == "histogram":
        fig = px.histogram(df, x=spec.x, **common)
    else:
        fig = px.box(df, x=spec.x, y=y, **common)
    if kind == "line":
        fig.update_traces(
            line={"width": 2}, marker={"size": 8, "line": {"width": 2, "color": "#fcfcfb"}}
        )
    elif kind == "bar":
        fig.update_traces(marker={"line": {"width": 0}})
        fig.update_layout(bargap=0.35)
    fig.update_layout(
        title={"text": spec.title, "font": {"size": 16, "color": _INK}},
        template="plotly_white",
        colorway=COLORWAY,
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font={
            "family": "system-ui, -apple-system, 'Segoe UI', sans-serif",
            "color": _INK_2,
            "size": 12,
        },
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
        showlegend=bool(spec.color) or kind == "pie",
        legend={"title": {"text": ""}},
    )
    fig.update_xaxes(gridcolor=_GRID, linecolor=_AXIS, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=_GRID, linecolor=_AXIS, zeroline=False, automargin=True)
    return fig


def export_png(fig: go.Figure, path: Path) -> bool:
    """Write a PNG; returns False (never raises) if the static exporter is unavailable."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path), width=900, height=520, scale=2)
    except Exception as exc:
        log.warning("PNG export unavailable: %s", str(exc).splitlines()[0][:200])
        return False
    return True
