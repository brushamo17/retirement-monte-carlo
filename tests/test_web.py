"""Tests for the browser bridge (retirement_mc/web.py) and the bundled demo package."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from retirement_mc import (
    Annuity,
    GompertzMortality,
    RetirementPlan,
    SimulationConfig,
    price_payout_rate,
    simulate,
    social_security,
    web,
)

ROOT = Path(__file__).resolve().parent.parent
SMALL = {"n_paths": 2_000}


def scenario(sid):
    return next(s["inputs"] for s in web.presets()["scenarios"] if s["id"] == sid)


# ---- the demo reproduces the package exactly ----------------------------------

def test_default_inputs_reproduce_the_readme_4_percent_rule():
    out = web.run({})
    direct = simulate(RetirementPlan(), SimulationConfig(n_paths=10_000, seed=3044))
    assert out["success"]["p"] == direct.success.mean()
    assert round(out["success"]["p"], 3) == 0.894


def test_income_scenarios_match_building_the_plan_by_hand():
    mort = GompertzMortality(88, 10)
    real_rate = price_payout_rate(65, mort, discount_rate=0.020, load=0.05)
    plan = RetirementPlan(
        initial_portfolio=1_000_000, annual_spending=80_000, mortality=mort,
        income=(social_security(32_000, claim_age=67, retirement_age=65),),
        annuities=(Annuity(300_000, real_rate, inflation_adjusted=True),),
    )
    direct = simulate(plan, SimulationConfig(n_paths=10_000, seed=3044))
    out = web.run(scenario("annuity"))
    assert out["success"]["p"] == direct.success.mean()
    assert out["summary"]["median_terminal"] == pytest.approx(float(np.median(direct.terminal_wealth)))


def test_same_seed_same_answer_through_json():
    a = json.loads(web.call("run", json.dumps({"inputs": {**SMALL, "seed": 7}})))
    b = json.loads(web.call("run", json.dumps({"inputs": {**SMALL, "seed": 7}})))
    c = json.loads(web.call("run", json.dumps({"inputs": {**SMALL, "seed": 8}})))
    assert a == b
    assert a["percentiles"] != c["percentiles"]


@pytest.mark.parametrize("sid", [s["id"] for s in web.SCENARIOS])
def test_every_preset_runs_and_returns_valid_json(sid):
    inputs = {**scenario(sid), **SMALL}
    for name, payload in [("run", {"inputs": inputs}), ("grid", {"inputs": inputs, "max_paths": 500}),
                          ("sensitivity", {"inputs": inputs, "max_paths": 500})]:
        out = json.loads(web.call(name, json.dumps(payload)))  # allow_nan=False: no NaN/inf leaks into JSON
        assert out
    run = web.run(inputs)
    assert 0.0 <= run["success"]["p"] <= 1.0
    assert run["success"]["ci_low"] <= run["success"]["p"] <= run["success"]["ci_high"]


# ---- inputs -----------------------------------------------------------------

def test_bad_values_fall_back_or_clamp():
    x = web.normalize({"spending": "", "stock_share": 3, "ss_claim_age": 55, "n_paths": 1001, "fee": None})
    assert x["spending"] == web.DEFAULTS["spending"]
    assert x["stock_share"] == 1.0
    assert x["ss_claim_age"] == 62
    assert x["n_paths"] == 1002          # even, for antithetic pairs
    assert x["fee"] == web.DEFAULTS["fee"]


def test_unknown_inputs_are_a_bug_not_a_silent_default():
    with pytest.raises(ValueError, match="unknown inputs"):
        web.normalize({"spendng": 50_000})
    with pytest.raises(ValueError, match="unknown function"):
        web.call("simulate_everything", "{}")


def test_annuity_premium_cannot_exceed_the_portfolio():
    x = web.normalize({"portfolio": 100_000, "annuity_on": True, "annuity_premium": 300_000})
    assert x["annuity_premium"] == 100_000


def test_bootstrap_turns_off_antithetic():
    assert web.normalize({"model": "bootstrap", "antithetic": True})["antithetic"] is False


def test_named_markets_use_exact_assumptions_custom_uses_the_fields():
    plan, config = web.build({"market": "historical", "stock_mean": 0.02})
    assert config.market == web.HISTORICAL_US
    plan, config = web.build({"market": "custom", "stock_mean": 0.02})
    assert config.market.stock_mean == 0.02


def test_pension_without_cola_loses_value_and_starts_on_time():
    out = web.run({**SMALL, "retirement_age": 60, "pension_on": True, "pension_amount": 30_000,
                   "pension_start_age": 62, "pension_cola": False})
    values = out["income"]["streams"][0]["values"]
    assert values[0] == values[1] == 0.0
    assert values[2] == pytest.approx(30_000 / 1.0302**2, rel=0.01)
    assert values[10] < values[2]


# ---- what the charts show ------------------------------------------------------

def test_funded_curve_ends_at_the_funded_rate_with_a_fixed_horizon():
    out = web.run(SMALL)
    curves = out["curves"]
    assert curves["funded"][0] == 1.0
    assert curves["funded"][-1] == pytest.approx(out["summary"]["funded"])
    assert all(a >= b for a, b in zip(curves["funded"], curves["funded"][1:]))
    assert curves["alive"] is None


def test_alive_but_depleted_never_exceeds_either_curve():
    out = web.run({**scenario("ss67"), **SMALL})
    c = out["curves"]
    assert c["ages"][-1] == 100
    for f, a, ad in zip(c["funded"], c["alive"], c["alive_depleted"]):
        assert ad <= a + 1e-12
        assert ad <= 1 - f + 1e-12


def test_depletion_agrees_with_the_engine_for_the_living():
    plan, config = web.build({**scenario("ss67"), **SMALL})
    res = simulate(plan, config)
    d = web.depletion_year(res)
    alive_at_d = np.isfinite(d) & (res.lifespan > np.where(np.isfinite(d), d, 0))
    assert np.array_equal(res.fail_year > 0, alive_at_d)
    assert np.array_equal(res.fail_year[alive_at_d], d[alive_at_d].astype(int) + 1)


def test_grid_is_monotone_in_spending_under_the_fixed_rule():
    g = web.grid(SMALL)
    for row in g["values"]:
        assert all(a >= b for a, b in zip(row, row[1:]))


def test_grid_centers_on_the_plan_when_income_covers_part_of_spending():
    assert web.grid(SMALL)["rates"] == list(web.GRID_RATES)
    g = web.grid({**scenario("annuity"), **SMALL}, max_paths=500)
    assert g["centered"] and g["rates"][4] == pytest.approx(0.08) and len(g["rates"]) == 9
    assert min(web.grid({"spending": 5_000, **SMALL}, max_paths=500)["rates"]) > 0


# ---- comparison and sensitivity -------------------------------------------------

def test_comparing_a_plan_with_itself_gives_exactly_zero():
    c = web.compare(SMALL, SMALL)
    assert c["diff"]["p"] == 0.0 and c["diff"]["se"] == 0.0


def test_common_random_numbers_tighten_the_comparison():
    c = web.compare({**scenario("ss67"), **SMALL}, {**scenario("ss70"), **SMALL})
    assert c["diff"]["se"] < 0.5 * c["independent_se"]
    assert c["b"]["success"]["p"] > c["a"]["success"]["p"]


def test_plan_b_inherits_plan_as_simulation_settings():
    c = web.compare({**SMALL, "seed": 1}, {"n_paths": 500, "seed": 99})
    assert c["b"]["success"]["n"] == 2_000


def test_sensitivity_moves_in_the_right_direction():
    s = web.sensitivity(SMALL)
    f = {r["name"]: r for r in s["factors"]}
    # With shared paths these hold path by path, not just on average.
    assert f["Spending"]["p_high"] <= s["base"] <= f["Spending"]["p_low"]
    assert f["Fees"]["p_high"] <= s["base"] <= f["Fees"]["p_low"]
    assert f["Stock returns"]["p_low"] <= s["base"] <= f["Stock returns"]["p_high"]
    assert f["Retirement length"]["p_high"] <= s["base"] <= f["Retirement length"]["p_low"]


def test_inflation_only_matters_with_fixed_dollar_income():
    s = web.sensitivity(SMALL)
    infl = next(r for r in s["factors"] if r["name"] == "Inflation")
    assert infl["p_low"] == infl["p_high"] == s["base"]
    assert s["notes"]
    s = web.sensitivity({**scenario("pension"), **SMALL})
    infl = next(r for r in s["factors"] if r["name"] == "Inflation")
    assert infl["p_high"] < infl["p_low"]


def test_sensitivity_is_sorted_by_impact():
    s = web.sensitivity({**scenario("annuity"), **SMALL})
    swing = [max(abs(r["p_low"] - s["base"]), abs(r["p_high"] - s["base"])) for r in s["factors"]]
    assert swing == sorted(swing, reverse=True)


# ---- the bundled package the browser loads ------------------------------------------

def test_demo_bundle_matches_the_source():
    """docs/retirement_mc.zip must be rebuilt (python scripts/build_web.py) after any package change."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_web", ROOT / "scripts" / "build_web.py")
    build_web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_web)
    expected = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in build_web.package_files()}
    with zipfile.ZipFile(ROOT / "docs" / "retirement_mc.zip") as zf:
        bundled = {name: zf.read(name) for name in zf.namelist()}
    stale = sorted(n for n in expected if bundled.get(n) != expected[n])
    extra = sorted(set(bundled) - set(expected))
    assert not stale and not extra, (
        f"docs/retirement_mc.zip is out of date (changed: {stale}, removed: {extra}). "
        "Run: python scripts/build_web.py"
    )
