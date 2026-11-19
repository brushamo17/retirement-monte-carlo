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
from retirement_mc.analysis import compare_scenarios, paired_difference
from retirement_mc.returns import (
    arithmetic_correlation,
    calibrate_correlation,
    calibrate_marginal,
    lognormal_params,
)

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


@pytest.mark.parametrize("model", ["lognormal", "fat_tailed"])
def test_parametric_models_match_arithmetic_moments_and_correlation(model):
    """Both models must reproduce the arithmetic mean, sd *and* correlation we asked for."""
    market = MarketAssumptions(correlation=0.30)
    stock, bond = generate_returns(market, 400_000, 1, model=model, seed=1)
    assert stock.mean() == pytest.approx(market.stock_mean, abs=0.002)
    assert stock.std() == pytest.approx(market.stock_sd, abs=0.002)
    assert bond.std() == pytest.approx(market.bond_sd, abs=0.001)
    assert np.corrcoef(stock[:, 0], bond[:, 0])[0, 1] == pytest.approx(0.30, abs=0.01)


def test_numeric_calibration_matches_closed_form_lognormal():
    mu, sigma = calibrate_marginal(0.07, 0.18, "lognormal")
    assert (mu, sigma) == pytest.approx(lognormal_params(0.07, 0.18), abs=1e-9)


def test_correlation_calibration_matches_textbook_lognormal_formula():
    """Corr(e^X, e^Y) = (e^{rho sx sy} - 1) / sqrt((e^{sx^2} - 1)(e^{sy^2} - 1))."""
    sx, sy, rho = 0.18, 0.08, 0.4
    textbook = (np.exp(rho * sx * sy) - 1) / np.sqrt((np.exp(sx**2) - 1) * (np.exp(sy**2) - 1))
    assert arithmetic_correlation(rho, sx, sy) == pytest.approx(textbook, rel=1e-12)
    assert calibrate_correlation(textbook, sx, sy) == pytest.approx(rho, abs=1e-9)


def test_unreachable_correlation_is_rejected():
    _, s1 = calibrate_marginal(0.08, 1.5)   # extreme volatility caps the reachable correlation
    _, s2 = calibrate_marginal(0.02, 0.05)
    with pytest.raises(ValueError):
        calibrate_correlation(0.99, s1, s2)


def test_fat_tails_add_crashes_without_any_floor():
    market = MarketAssumptions()
    s_n, _ = generate_returns(market, 400_000, 1, model="lognormal", seed=2)
    s_f, _ = generate_returns(market, 400_000, 1, model="fat_tailed", seed=2)
    crash = market.stock_mean - 3 * market.stock_sd
    assert (s_f < crash).mean() > 1.5 * (s_n < crash).mean()
    assert s_f.min() > -1.0  # exp(x) - 1 can never reach -100%, so no floor is needed


def test_antithetic_pairs_are_mirror_images():
    """In log space each pair sits symmetrically around mu."""
    market = MarketAssumptions()
    stock, _ = generate_returns(market, 10, 5, antithetic=True, seed=3)
    mu, _ = calibrate_marginal(market.stock_mean, market.stock_sd)
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


def test_guardrails_rarely_run_out_but_the_floor_catches_deep_cuts():
    """Guardrails avoid shortfalls by cutting; the spending floor must count those cuts."""
    fixed = simulate(RetirementPlan(annual_spending=50_000), FAST)
    guard = simulate(RetirementPlan(annual_spending=50_000, rule="guardrails"), FAST)
    assert guard.funded.mean() >= fixed.funded.mean()
    assert np.all(guard.success <= guard.funded)
    deep_cut = guard.lowest_spending < 0.8
    assert deep_cut.any()
    assert not guard.success[deep_cut].any()
    no_floor = simulate(replace(guard.plan, spending_floor=0.0), FAST)
    assert np.array_equal(no_floor.success, no_floor.funded)


def test_fixed_rule_success_does_not_depend_on_the_floor():
    """Under fixed spending, consumption only drops when money runs out."""
    a = simulate(RetirementPlan(annual_spending=50_000, spending_floor=0.0), FAST)
    b = simulate(RetirementPlan(annual_spending=50_000, spending_floor=0.9), FAST)
    assert np.array_equal(a.success, b.success)


def test_lowest_spending_counts_actual_funded_spending_after_a_shortfall():
    plan = RetirementPlan(initial_portfolio=100_000, annual_spending=40_000, horizon_years=10)
    res = simulate(plan, FAST)
    assert np.all(res.lowest_spending[~res.funded] < 1.0)
    assert res.lowest_spending.min() == pytest.approx(0.0)


def test_mortality_counts_only_years_lived():
    """A retiree who dies in year 3 cannot fail in year 10."""
    plan = RetirementPlan(initial_portfolio=100_000, annual_spending=20_000, horizon_years=20)
    lifespans = np.full(FAST.n_paths, 3.0)
    res = simulate(plan, FAST, lifespans=lifespans)
    assert res.success.all()
    res_full = simulate(replace(plan), FAST)
    assert not res_full.success.all()


def test_crn_comparison_uses_each_plans_own_mortality():
    """Plans with different mortality must get different lifespans from the same uniforms."""
    base = dict(annual_spending=45_000, retirement_age=65)
    res = compare_scenarios(
        {"short-lived": RetirementPlan(**base, mortality=GompertzMortality(modal_age=85)),
         "long-lived": RetirementPlan(**base, mortality=GompertzMortality(modal_age=92))},
        FAST,
    )
    a, b = res["short-lived"], res["long-lived"]
    assert np.all(a.lifespan <= b.lifespan)                        # same uniform, later modal age
    assert a.lifespan.mean() < b.lifespan.mean()
    assert np.allclose(a.portfolio_returns, b.portfolio_returns)   # same markets
    assert np.all(b.success <= a.success)                          # living longer can only hurt


def test_crn_comparison_handles_different_horizons():
    res = compare_scenarios(
        {"25 years": RetirementPlan(horizon_years=25), "35 years": RetirementPlan(horizon_years=35)},
        FAST,
    )
    a, b = res["25 years"], res["35 years"]
    assert np.allclose(a.portfolio_returns, b.portfolio_returns[:, :25])
    assert np.all(b.success <= a.success)


def test_paired_difference_is_tighter_than_independent_comparison():
    plans = {"base": RetirementPlan(annual_spending=45_000), "less": RetirementPlan(annual_spending=40_000)}
    res = compare_scenarios(plans, FAST)
    d = paired_difference(res["base"], res["less"])
    independent_se = np.hypot(res["base"].success_probability().se, res["less"].success_probability().se)
    assert d.p > 0
    assert d.se < independent_se


def test_outliving_the_horizon_triggers_a_warning():
    plan = RetirementPlan(mortality=GompertzMortality(), retirement_age=65, max_age=80)
    with pytest.warns(UserWarning, match="outlive max_age"):
        res = simulate(plan, FAST)
    assert res.summary()["share_beyond_horizon"] > 0.5


def test_default_horizon_is_long_enough():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = simulate(RetirementPlan(mortality=GompertzMortality()), FAST)
    assert res.beyond_horizon.sum() == 0
