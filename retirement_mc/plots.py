"""Matplotlib charts for simulation results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .simulation import SimulationResult

BLUE_LIGHT, BLUE_MID, BLUE_DARK = "#cde2fb", "#86b6ef", "#104281"
SERIES = ["#eb6834", "#2a78d6", "#1baf7a", "#6250d6"]  # orange, blue, aqua, violet
INK, MUTED, GRID = "#0e1420", "#7a8293", "#e3e6ec"


def _money(x, _pos=None) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}k"
    return f"${x:.0f}"


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c8d2")
    ax.tick_params(colors=MUTED, length=0)


def fan_chart(result: SimulationResult, ax=None):
    """Percentile bands (10-90, 25-75) and median of real portfolio value by year."""
    ax = ax or plt.subplots(figsize=(9, 4.5))[1]
    p10, p25, p50, p75, p90 = result.percentiles()
    years = np.arange(p50.size)
    ax.fill_between(years, p10, p90, color=BLUE_LIGHT, label="10th to 90th percentile", linewidth=0)
    ax.fill_between(years, p25, p75, color=BLUE_MID, label="25th to 75th percentile", linewidth=0)
    ax.plot(years, p50, color=BLUE_DARK, linewidth=2, label="Median")
    ax.axhline(result.plan.invested_portfolio, color=MUTED, linestyle=(0, (3, 4)), linewidth=1)
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_xlim(0, years[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Years into retirement", color=MUTED)
    est = result.success_probability()
    ax.set_title(f"Real portfolio value (success {est.p:.1%})", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _style(ax)
    return ax


def success_heatmap(grid, withdrawal_rates, stock_shares, ax=None):
    """Grid of success probabilities: rows are stock shares, columns withdrawal rates."""
    ax = ax or plt.subplots(figsize=(9, 6))[1]
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(withdrawal_rates)), [f"{r:.1%}" for r in withdrawal_rates])
    ax.set_yticks(range(len(stock_shares)), [f"{s:.0%}" for s in stock_shares])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 else INK)
    ax.set_xlabel("Initial withdrawal rate (share of starting portfolio)", color=MUTED)
    ax.set_ylabel("Share in stocks", color=MUTED)
    ax.set_title("Probability the portfolio lasts", loc="left", color=INK, fontsize=12)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=PercentFormatter(1.0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    return ax


def reorder_chart(demo: dict, ax=None):
    """The same returns replayed worst-first, as simulated and best-first."""
    ax = ax or plt.subplots(figsize=(9, 4.5))[1]
    for (name, bal), color in zip(demo["balances"].items(), SERIES):
        fail = demo["fail_year"][name]
        end = fail if fail else bal.size - 1
        label = f"{name} (runs out in year {fail})" if fail else name
        ax.plot(np.arange(end + 1), bal[: end + 1], color=color, linewidth=2, label=label)
        if fail:
            ax.plot(fail, 0, "o", color=color, markersize=6)
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Years into retirement", color=MUTED)
    ax.set_title(f"Same {demo['annualized_return']:.1%} average return, three orders",
                 loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    return ax


def model_risk_chart(rates, curves: dict[str, list[float]], ax=None):
    """Success by withdrawal rate under different return models (model risk)."""
    ax = ax or plt.subplots(figsize=(9, 4.8))[1]
    colors = SERIES + ["#7a8293"]
    for (name, vals), color in zip(curves.items(), colors):
        dashed = name.startswith("Historical")
        ax.plot(rates, vals, color=color, linewidth=2, marker="o", markersize=5,
                linestyle=(0, (4, 3)) if dashed else "-", label=name)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(rates)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Initial withdrawal rate (60/40, 30 years)", color=MUTED)
    ax.set_title("Same plan, different return models", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    _style(ax)
    return ax


def essentials_chart(summaries: dict[str, dict], axes=None):
    """Three panels: essentials always covered, discretionary enjoyed, median bequest."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    names = list(summaries)
    y = np.arange(len(names))[::-1]
    panels = [
        ("Essentials always covered", [summaries[n]["floor_met"] for n in names], SERIES[2], lambda v: f"{v:.1%}"),
        ("Discretionary spending enjoyed", [summaries[n]["median_discretionary_share"] for n in names], SERIES[1],
         lambda v: f"{v:.0%}"),
        ("Median amount left at death", [summaries[n]["median_terminal"] for n in names], SERIES[0], _money),
    ]
    for i, (ax, (title, vals, color, fmt)) in enumerate(zip(axes, panels)):
        ax.barh(y, vals, color=color, height=0.55)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_yticks(y, names if i == 0 else [""] * len(names))
        for yi, v in zip(y, vals):
            ax.text(v, yi, "  " + fmt(v), va="center", fontsize=9, color=INK)
        ax.xaxis.set_major_formatter(FuncFormatter(_money) if i == 2 else PercentFormatter(1.0))
        if i < 2:
            ax.set_xlim(0, 1.0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
        ax.margins(x=0.28)
    return axes


def scenario_chart(summaries: dict[str, dict], axes=None):
    """Side-by-side bars: probability of a shortfall while alive, and median bequest."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(summaries)
    y = np.arange(len(names))[::-1]
    fail = [1 - summaries[n]["success"] for n in names]
    beq = [summaries[n]["median_terminal"] for n in names]
    axes[0].barh(y, fail, color=SERIES[0], height=0.55)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("Chance of running short while alive", loc="left", color=INK, fontsize=11)
    axes[1].barh(y, beq, color=SERIES[1], height=0.55)
    axes[1].xaxis.set_major_formatter(FuncFormatter(_money))
    axes[1].set_title("Median amount left at death", loc="left", color=INK, fontsize=11)
    for ax, vals, fmt in ((axes[0], fail, lambda v: f"{v:.1%}"), (axes[1], beq, _money)):
        ax.set_yticks(y, names if ax is axes[0] else [""] * len(names))
        for yi, v in zip(y, vals):
            ax.text(v, yi, "  " + fmt(v), va="center", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
        ax.margins(x=0.25)
    return axes
