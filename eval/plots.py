"""Result plots (matplotlib). Palette, mark weights and chrome follow the dataviz method:
validated categorical colors in fixed order, thin marks, hairline grid, direct labels, one
axis per chart (never dual-axis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
DIFFICULTIES = ("easy", "medium", "hard", "extra")
LADDER_ORDER = [
    "zero_shot",
    "full_schema",
    "linker",
    "retry_r1",
    "retry_r2",
    "retry_r3",
    "few_shot",
    "critic",
    "full",
]
LABELS = {
    "zero_shot": "1 zero-shot", "full_schema": "2 + full schema", "linker": "3 + schema linker",
    "retry_r1": "4a + 1 retry", "retry_r2": "4b + 2 retries", "retry_r3": "4c + 3 retries",
    "few_shot": "5 + few-shot", "critic": "6 + critic", "full": "7 full system",
}  # fmt: skip
Summaries = dict[str, dict[str, Any]]


def _style(ax: Any, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(width: float = 8.5, height: float = 4.8) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    return fig, ax


def _ordered(summaries: Summaries) -> list[str]:
    known = [n for n in LADDER_ORDER if n in summaries]
    return known + [n for n in summaries if n not in known]


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_ex_by_difficulty(summaries: Summaries, path: Path, names: list[str] | None = None) -> Path:
    """Grouped bars: EX per difficulty for a handful of configs (<= 4 series, fixed order)."""
    names = [
        n for n in (names or ["zero_shot", "full_schema", "retry_r3", "full"]) if n in summaries
    ][:4]
    fig, ax = _fig()
    width = 0.8 / max(1, len(names))
    for i, name in enumerate(names):
        by = summaries[name].get("ex_by_difficulty", {})
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(DIFFICULTIES))]
        ys = [100 * by.get(d, {}).get("ex", 0) for d in DIFFICULTIES]
        ax.bar(
            xs,
            ys,
            width=width * 0.86,
            color=PALETTE[i],
            label=LABELS.get(name, name),
            edgecolor=SURFACE,
            linewidth=2,
        )
        for x, y, d in zip(xs, ys, DIFFICULTIES, strict=True):
            if d in by:
                ax.text(x, y + 1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(DIFFICULTIES)))
    counts = {
        d: next(
            (
                s.get("ex_by_difficulty", {}).get(d, {}).get("n")
                for s in summaries.values()
                if d in s.get("ex_by_difficulty", {})
            ),
            0,
        )
        for d in DIFFICULTIES
    }
    ax.set_xticklabels([f"{d}\n(n={counts[d]})" for d in DIFFICULTIES])
    ax.set_ylim(0, 108)
    _style(ax, "Execution accuracy by question difficulty", ylabel="EX (%)")
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=len(names),
    )
    return _save(fig, path)


def plot_ablation_bars(summaries: Summaries, path: Path) -> Path:
    """Horizontal bars of EX with thin 95% bootstrap CI whiskers, in ladder order."""
    names = _ordered(summaries)
    fig, ax = _fig(8.5, 0.55 * len(names) + 1.6)
    for i, name in enumerate(names):
        s = summaries[name]
        ex, (lo, hi) = 100 * s["ex"], (100 * s["ex_ci95"][0], 100 * s["ex_ci95"][1])
        y = len(names) - 1 - i
        ax.barh(y, ex, height=0.5, color=PALETTE[0])
        ax.plot([lo, hi], [y, y], color=INK, linewidth=1.2, solid_capstyle="butt")
        ax.text(
            min(hi, 100) + 1.2, y, f"{ex:.1f}%  (n={s['n']})", va="center", fontsize=9, color=INK_2
        )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS.get(n, n) for n in reversed(names)])
    ax.set_xlim(0, 118)
    _style(ax, "Ablation ladder: execution accuracy with 95% bootstrap CI", xlabel="EX (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return _save(fig, path)


def plot_retry_curve(summaries: Summaries, path: Path) -> Path:
    """EX vs number of allowed retries (rungs linker, r1, r2, r3), with the CI as a band."""
    chain = [(0, "linker"), (1, "retry_r1"), (2, "retry_r2"), (3, "retry_r3")]
    pts = [(r, summaries[n]) for r, n in chain if n in summaries]
    fig, ax = _fig(6.5, 4.4)
    if pts:
        xs = [r for r, _ in pts]
        ys = [100 * s["ex"] for _, s in pts]
        lo = [100 * s["ex_ci95"][0] for _, s in pts]
        hi = [100 * s["ex_ci95"][1] for _, s in pts]
        ax.fill_between(xs, lo, hi, color=PALETTE[0], alpha=0.15, linewidth=0)
        ax.plot(
            xs,
            ys,
            color=PALETTE[0],
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color=INK_2,
            )
    ax.set_xticks([0, 1, 2, 3])
    _style(ax, "Accuracy vs allowed retries", xlabel="max retries", ylabel="EX (%)")
    return _save(fig, path)


def plot_cost_vs_accuracy(summaries: Summaries, path: Path) -> Path:
    """Scatter of estimated $/question against EX, one directly-labelled dot per config."""
    fig, ax = _fig(7.5, 4.8)
    for name in _ordered(summaries):
        s = summaries[name]
        x, y = s["cost_usd_per_question"] * 1000, 100 * s["ex"]
        ax.scatter([x], [y], s=80, color=PALETTE[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(
            LABELS.get(name, name),
            (x, y),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8.5,
            color=INK_2,
        )
    _style(
        ax,
        "Cost vs accuracy",
        xlabel="estimated cost per question (USD x 1000, paid-tier list price)",
        ylabel="EX (%)",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return _save(fig, path)


def plot_run(summary: dict[str, Any], run_dir: Path) -> Path | None:
    """Per-run figure: EX by difficulty (or by dataset for the custom suite)."""
    groups = summary.get("ex_by_difficulty") or {}
    key = "ex_by_group" if not any(d in groups for d in DIFFICULTIES) else None
    data = (
        summary.get("ex_by_group", {})
        if key
        else {d: groups[d] for d in DIFFICULTIES if d in groups}
    )
    if not data:
        return None
    fig, ax = _fig(7.5, 4.2)
    names = list(data)
    ys = [100 * data[k]["ex"] for k in names]
    ax.bar(range(len(names)), ys, width=0.55, color=PALETTE[0])
    for i, y in enumerate(ys):
        ax.text(i, y + 1.5, f"{y:.0f}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{k}\n(n={data[k]['n']})" for k in names], fontsize=8)
    ax.set_ylim(0, 108)
    _style(ax, f"Execution accuracy · {summary.get('run_id', '')}", ylabel="EX (%)")
    return _save(fig, run_dir / "ex_breakdown.png")


def plot_all(summaries: Summaries, out_dir: Path) -> list[Path]:
    """Every cross-run plot the README references."""
    return [
        plot_ablation_bars(summaries, out_dir / "ablation_bars.png"),
        plot_ex_by_difficulty(summaries, out_dir / "ex_by_difficulty.png"),
        plot_retry_curve(summaries, out_dir / "retry_curve.png"),
        plot_cost_vs_accuracy(summaries, out_dir / "cost_vs_accuracy.png"),
    ]
