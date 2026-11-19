"""Monte Carlo retirement planning simulator.

Typical use::

    from retirement_mc import RetirementPlan, SimulationConfig, simulate

    plan = RetirementPlan(initial_portfolio=1_000_000, annual_spending=40_000)
    result = simulate(plan, SimulationConfig(n_paths=10_000, seed=3044))
    print(result.success_probability())
"""

from .analysis import (
    compare_scenarios,
    historical_backtest,
    paired_difference,
    reorder_demo,
    sequence_quintiles,
    success_grid,
)
from .history import MarketHistory, load_us_history
from .income import (
    Annuity,
    IncomeStream,
    annuity_factor,
    price_payout_rate,
    social_security,
    social_security_factor,
)
from .mortality import GompertzMortality
from .returns import HISTORICAL_US, MUTED_FORWARD, MarketAssumptions, generate_returns
from .simulation import (
    Estimate,
    RetirementPlan,
    SimulationConfig,
    SimulationResult,
    estimate_mean,
    run_paths,
    simulate,
)

__all__ = [
    "Annuity", "Estimate", "GompertzMortality", "HISTORICAL_US", "IncomeStream",
    "MUTED_FORWARD", "MarketAssumptions", "MarketHistory", "RetirementPlan",
    "SimulationConfig", "SimulationResult", "annuity_factor", "compare_scenarios",
    "estimate_mean", "generate_returns", "historical_backtest", "load_us_history",
    "paired_difference", "price_payout_rate", "reorder_demo", "run_paths",
    "sequence_quintiles", "simulate", "social_security", "social_security_factor",
    "success_grid",
]
