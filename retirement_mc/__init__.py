"""Monte Carlo retirement planning simulator.

Typical use::

    from retirement_mc import RetirementPlan, SimulationConfig, simulate

    plan = RetirementPlan(initial_portfolio=1_000_000, annual_spending=40_000)
    result = simulate(plan, SimulationConfig(n_paths=10_000, seed=3044))
    print(result.success_probability())
"""

from .analysis import compare_scenarios, reorder_demo, sequence_quintiles, success_grid
from .income import (
    Annuity,
    IncomeStream,
    annuity_factor,
    price_payout_rate,
    social_security,
    social_security_factor,
)
from .mortality import GompertzMortality
from .returns import LONG_RUN_US, MUTED_FORWARD, MarketAssumptions, generate_returns
from .simulation import (
    Estimate,
    RetirementPlan,
    SimulationConfig,
    SimulationResult,
    run_paths,
    simulate,
)

__all__ = [
    "Annuity", "Estimate", "GompertzMortality", "IncomeStream", "LONG_RUN_US",
    "MUTED_FORWARD", "MarketAssumptions", "RetirementPlan", "SimulationConfig",
    "SimulationResult", "annuity_factor", "compare_scenarios", "generate_returns",
    "price_payout_rate", "reorder_demo", "run_paths", "sequence_quintiles",
    "simulate", "social_security", "social_security_factor", "success_grid",
]
