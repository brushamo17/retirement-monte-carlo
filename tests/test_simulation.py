"""Tests that check the simulator against things we can work out by hand.

Run with:  pytest -q
"""

from dataclasses import replace

import numpy as np
import pytest

from retirement_mc import (
    Annuity,
    GompertzMortality,
    IncomeStream,
    MarketAssumptions,
    RetirementPlan,
    SimulationConfig,
    generate_returns,
    price_payout_rate,
    simulate,
    social_security_factor,
    success_grid,
)
from retirement_mc.returns import lognormal_params

FAST = SimulationConfig(n_paths=2_000, seed=7)


def test_same_seed_same_answer():
    plan = RetirementPlan()
    a = simulate(plan, FAST)
    b = simulate(plan, FAST)
    assert np.array_equal(a.balances, b.balances)


def test_zero_volatility_matches_closed_form():
    """With no randomness the balance follows the annuity-due recursion exactly.

    B_t = (1 + r)^t * B_0 - W * (1 + r) * ((1 + r)^t - 1) / r
    """
    r, b0, w = 0.03, 1_000_000.0, 60_000.0
    market = MarketAssumptions(stock_mean=r, stock_sd=0.0, bond_mean=r, bond_sd=0.0)
    plan = RetirementPlan(initial_portfolio=b0, annual_spending=w, fee=0.0, horizon_years=40)
    res = simulate(plan, SimulationConfig(n_paths=4, market=market))
    t = np.arange(41)
    expected = (1 + r) ** t * b0 - w * (1 + r) * ((1 + r) ** t - 1) / r
    funded = expected > 0
    assert np.allclose(res.balances[0, funded], expected[funded], rtol=1e-9)
    # First year the balance cannot cover the withdrawal = first shortfall year.
    first_short = int(np.argmax(expected < w)) + 1
    assert res.fail_year[0] == first_short


def test_lognormal_moments_match_inputs():
    """Simulated returns should reproduce the arithmetic mean and sd we asked for."""
    market = MarketAssumptions()
    stock, bond = generate_returns(market, 200_000, 1, seed=1)
    assert stock.mean() == pytest.approx(market.stock_mean, abs=0.002)
    assert stock.std() == pytest.approx(market.stock_sd, abs=0.002)
    assert np.corrcoef(np.log1p(stock[:, 0]), np.log1p(bond[:, 0]))[0, 1] == pytest.approx(
        market.correlation, abs=0.01
    )


def test_fat_tails_keep_mean_and_sd_but_add_crashes():
    market = MarketAssumptions()
    s_n, _ = generate_returns(market, 200_000, 1, model="lognormal", seed=2)
    s_t, _ = generate_returns(market, 200_000, 1, model="t", seed=2)
    assert s_t.std() == pytest.approx(market.stock_sd, abs=0.004)
    crash = market.stock_mean - 3 * market.stock_sd
    assert (s_t < crash).mean() > (s_n < crash).mean()


def test_antithetic_pairs_are_mirror_images():
    """In log space each pair sits symmetrically around mu."""
    market = MarketAssumptions()
    stock, _ = generate_returns(market, 10, 5, antithetic=True, seed=3)
    mu, _ = lognormal_params(market.stock_mean, market.stock_sd)
    logs = np.log1p(stock)
    assert np.allclose(logs[0::2] + logs[1::2], 2 * mu)


def test_income_that_covers_spending_never_fails():
    pension = IncomeStream("Pension", 50_000)
    plan = RetirementPlan(initial_portfolio=10_000, annual_spending=50_000, income=(pension,))
    assert simulate(plan, FAST).success.all()


def test_nominal_income_loses_value_to_inflation():
    stream = IncomeStream("Fixed pension", 20_000, inflation_adjusted=False)
    sched = stream.real_schedule(3, inflation=0.03)
    assert np.allclose(sched, [20_000, 20_000 / 1.03, 20_000 / 1.03**2])


def test_annuity_premium_leaves_the_portfolio():
    ann = Annuity(premium=300_000, payout_rate=0.07)
    plan = RetirementPlan(initial_portfolio=1_000_000, annuities=(ann,))
    assert plan.invested_portfolio == 700_000
    assert ann.annual_payment == pytest.approx(21_000)


def test_inflation_adjusted_annuity_pays_less_up_front():
    m = GompertzMortality()
    assert price_payout_rate(65, m, 0.02) < price_payout_rate(65, m, 0.045)


def test_social_security_claiming_factors():
    """Current-law factors for FRA 67 (SSA): 70% at 62, 100% at 67, 124% at 70."""
    assert social_security_factor(62) == pytest.approx(0.70)
    assert social_security_factor(64) == pytest.approx(0.80)
    assert social_security_factor(67) == pytest.approx(1.00)
    assert social_security_factor(70) == pytest.approx(1.24)


def test_gompertz_sampling_matches_survival_curve():
    m = GompertzMortality()
    life = m.sample_remaining_life(65, 200_000, np.random.default_rng(4))
    for t in (10, 20, 30):
        assert (life > t).mean() == pytest.approx(m.survival(65, t), abs=0.005)


def test_common_random_numbers_make_success_monotone():
    """Same paths for every cell: spending more can never raise success."""
    grid = success_grid(RetirementPlan(), FAST, stock_shares=(0.8, 0.4))
    assert np.all(np.diff(grid, axis=1) <= 0)


def test_guardrails_do_not_fail_more_often_than_fixed():
    """Guardrails cut spending in bad markets, so success should not fall much below fixed."""
    fixed = simulate(RetirementPlan(annual_spending=50_000), FAST).success.mean()
    guard = simulate(RetirementPlan(annual_spending=50_000, rule="guardrails"), FAST).success.mean()
    assert guard >= fixed - 0.02


def test_mortality_counts_only_years_lived():
    """A retiree who dies in year 3 cannot fail in year 10."""
    plan = RetirementPlan(initial_portfolio=100_000, annual_spending=20_000, horizon_years=20)
    lifespans = np.full(FAST.n_paths, 3.0)
    res = simulate(plan, FAST, lifespans=lifespans)
    assert res.success.all()
    res_full = simulate(replace(plan), FAST)
    assert not res_full.success.all()
