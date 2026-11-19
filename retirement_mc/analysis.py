"""Analyses built on the engine: policy grids, scenario comparisons, sequence risk,
historical backtests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .history import MarketHistory, load_us_history
from .returns import generate_returns
from .simulation import (
    Estimate,
    RetirementPlan,
    SimulationConfig,
    SimulationResult,
    estimate_mean,
    lifespan_uniforms,
    portfolio_returns_for,
    run_paths,
    simulate,
)


def common_draws(plans, config: SimulationConfig):
    """One set of market paths and lifespan uniforms to share across plans (CRN).

    Returns are generated for the *longest* plan; shorter plans use the first
    columns. Lifespans are shared as uniforms, so plans with different
    mortality tables or retirement ages still face matched longevity luck.
    """
    plans = list(plans) if not isinstance(plans, RetirementPlan) else [plans]
    n_years = max(p.n_years for p in plans)
    returns = generate_returns(
        config.market, config.n_paths, n_years, model=config.model,
        antithetic=config.antithetic, seed=config.seed, block_size=config.block_size,
    )
    return returns, lifespan_uniforms(config)


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
    returns, u = common_draws(plan, config)
    grid = np.empty((len(stock_shares), len(withdrawal_rates)))
    for i, w in enumerate(stock_shares):
        for j, r in enumerate(withdrawal_rates):
            cell = replace(plan, stock_share=w, annual_spending=r * plan.initial_portfolio)
            grid[i, j] = simulate(cell, config, returns, uniforms=u).success.mean()
    return grid


def compare_scenarios(
    plans: dict[str, RetirementPlan], config: SimulationConfig
) -> dict[str, SimulationResult]:
    """Simulate several plans on the same market paths and longevity draws (CRN).

    Plans may differ in anything, including mortality assumptions, retirement
    age and horizon: each plan converts the shared uniforms into lifespans with
    its own survival curve and uses as many years of the shared returns as it needs.
    """
    returns, u = common_draws(plans.values(), config)
    return {name: simulate(p, config, returns, uniforms=u) for name, p in plans.items()}


def paired_difference(a: SimulationResult, b: SimulationResult) -> Estimate:
    """Estimate P(success | b) - P(success | a) from a CRN comparison.

    Because both plans ran on the same paths, the per-path difference
    d_i = success_b,i - success_a,i is the right quantity to average. Its
    variance is Var(a) + Var(b) - 2 Cov(a, b), and CRN makes the covariance
    large, so this interval is much tighter than comparing two independent runs.
    """
    if a.success.shape != b.success.shape:
        raise ValueError("results must come from the same number of paths")
    d = b.success.astype(float) - a.success.astype(float)
    return estimate_mean(d, a.config.antithetic and b.config.antithetic)


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


def historical_backtest(plan: RetirementPlan, history: MarketHistory | None = None,
                        inflation: float | None = None) -> dict:
    """Run the plan through every complete historical window (rolling start years).

    For a 30-year plan and data for 1928-2025, that is 69 retirements starting
    in 1928, 1929, ..., 1996. These windows overlap heavily, so they are far
    from 69 independent observations: treat the result as a sanity check on the
    simulation, not as a better estimate.

    Returns:
        Dict with the start years, success flags, ending balances and the
        overall success rate.
    """
    if plan.mortality is not None:
        raise ValueError("backtests use a fixed horizon; set mortality=None")
    h = history or load_us_history()
    n_years = plan.n_years
    starts = h.years[: len(h.years) - n_years + 1]
    if len(starts) == 0:
        raise ValueError("not enough history for this horizon")
    idx = np.arange(len(starts))[:, None] + np.arange(n_years)
    port = portfolio_returns_for(plan, h.real_stock[idx], h.real_bond[idx])
    infl = h.inflation.mean() if inflation is None else inflation
    income = plan.income_schedule(infl)
    out = run_paths(plan, port, np.full(len(starts), float(n_years)), income)
    low = np.min(out["consumption"], axis=1)
    ok = (out["fail_year"] == 0) & (low >= plan.floor_amount - 1e-9)
    return {
        "start_years": starts,
        "success": ok,
        "fail_year": out["fail_year"],
        "ending_balance": out["balances"][:, -1],
        "success_rate": float(ok.mean()),
    }
