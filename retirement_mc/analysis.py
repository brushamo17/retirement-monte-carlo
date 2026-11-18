"""Analyses built on the engine: policy grids, scenario comparisons and sequence risk."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .returns import generate_returns
from .simulation import (
    RetirementPlan,
    SimulationConfig,
    SimulationResult,
    run_paths,
    sample_lifespans,
    simulate,
)


def common_draws(plan: RetirementPlan, config: SimulationConfig):
    """Generate one set of market paths and lifespans to share across plans (CRN)."""
    returns = generate_returns(
        config.market, config.n_paths, plan.n_years,
        model=config.model, antithetic=config.antithetic, seed=config.seed,
    )
    return returns, sample_lifespans(plan, config)


def success_grid(
    plan: RetirementPlan,
    config: SimulationConfig,
    withdrawal_rates=(0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065),
    stock_shares=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0),
) -> np.ndarray:
    """Success probability for every (stock share, withdrawal rate) pair.

    Spending is set to ``rate * initial_portfolio``. Every cell is evaluated on
    the *same* market paths (common random numbers). Two consequences:

    * Differences between neighbouring cells reflect the policy, not noise,
      because the noise is shared and cancels in the comparison.
    * Under the fixed rule, success is monotone in the withdrawal rate along
      every single path (spending more can never help), so each row of the
      grid is non-increasing. Independent samples per cell would not guarantee that.

    Returns:
        Array of shape ``(len(stock_shares), len(withdrawal_rates))``.
    """
    returns, lifespans = common_draws(plan, config)
    grid = np.empty((len(stock_shares), len(withdrawal_rates)))
    for i, w in enumerate(stock_shares):
        for j, r in enumerate(withdrawal_rates):
            cell = replace(plan, stock_share=w, annual_spending=r * plan.initial_portfolio)
            grid[i, j] = simulate(cell, config, returns, lifespans).success.mean()
    return grid


def compare_scenarios(
    plans: dict[str, RetirementPlan], config: SimulationConfig
) -> dict[str, SimulationResult]:
    """Simulate several plans on the same market paths and lifespans (CRN).

    All plans must share ``n_years`` (same horizon or same mortality settings).
    """
    first = next(iter(plans.values()))
    returns, lifespans = common_draws(first, config)
    return {name: simulate(p, config, returns, lifespans) for name, p in plans.items()}


def sequence_quintiles(result: SimulationResult, window: int = 10) -> dict[str, list[dict]]:
    """Success rate by quintile of early-window vs late-window returns.

    Paths are ranked by their annualized (geometric) portfolio return over the
    first ``window`` years, split into fifths, and the success rate of each
    fifth is reported. The same is done for the last ``window`` years. If early
    returns drive outcomes far more than late ones, that is sequence risk.
    """
    port = result.portfolio_returns
    n_years = port.shape[1]
    window = min(window, n_years // 2)
    logs = np.log1p(port)
    early = np.expm1(logs[:, :window].mean(axis=1))
    late = np.expm1(logs[:, -window:].mean(axis=1))
    ok = result.success

    def by_quintile(metric: np.ndarray) -> list[dict]:
        order = np.argsort(metric)
        groups = np.array_split(order, 5)
        return [
            {"quintile": q + 1, "avg_return": float(np.expm1(np.log1p(metric[g]).mean())),
             "success": float(ok[g].mean())}
            for q, g in enumerate(groups)
        ]

    return {"window": window, "early": by_quintile(early), "late": by_quintile(late)}


def reorder_demo(result: SimulationResult) -> dict:
    """Replay one path's returns in three orders: worst-first, as simulated, best-first.

    The path chosen is the one with the median annualized return over the whole
    horizon. Reordering does not change the geometric mean (multiplication
    commutes), so any difference in outcome is purely sequence-of-returns risk.
    Spending, income and the withdrawal rule are the plan's own.
    """
    port = result.portfolio_returns
    n_years = port.shape[1]
    total = np.log1p(port).mean(axis=1)
    idx = int(np.argsort(total)[len(total) // 2])
    r = port[idx]
    orders = {
        "Worst years first": np.sort(r),
        "Simulated order": r,
        "Best years first": np.sort(r)[::-1],
    }
    stacked = np.vstack(list(orders.values()))
    out = run_paths(result.plan, stacked, np.full(3, float(n_years)), result.income)
    return {
        "annualized_return": float(np.expm1(total[idx])),
        "balances": dict(zip(orders, out["balances"])),
        "fail_year": dict(zip(orders, out["fail_year"].tolist())),
    }
