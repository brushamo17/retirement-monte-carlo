"""Reproduce every number and figure in the README.

    python run_example.py

Prints three reports and saves four charts to ./figures:

1. The classic question: $1M, 4% inflation-adjusted withdrawals, 60/40, 30 years.
2. The policy map: success for every withdrawal rate and stock share.
3. A household with Social Security, comparing claiming ages and annuities,
   with a random lifespan instead of a fixed horizon.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt

from retirement_mc import (
    Annuity,
    GompertzMortality,
    RetirementPlan,
    SimulationConfig,
    compare_scenarios,
    price_payout_rate,
    reorder_demo,
    sequence_quintiles,
    simulate,
    social_security,
    success_grid,
)
from retirement_mc.plots import fan_chart, reorder_chart, scenario_chart, success_heatmap

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
CONFIG = SimulationConfig(n_paths=10_000, seed=3044)


def money(x: float) -> str:
    return f"${x:,.0f}"


def part1_classic() -> None:
    print("=" * 72)
    print("1. $1M, 4% inflation-adjusted withdrawals, 60/40, 30 years, 0.5% fees")
    print("=" * 72)
    plan = RetirementPlan(initial_portfolio=1_000_000, annual_spending=40_000, stock_share=0.6)
    res = simulate(plan, CONFIG)
    s = res.summary()
    print(f"Probability the money lasts: {res.success_probability()}")
    print(f"Median ending balance:       {money(s['median_terminal'])}")
    print(f"10th percentile ending:      {money(s['p10_terminal'])}")
    print(f"Median failure year:         {s['median_fail_year']:.0f}")

    anti = simulate(plan, SimulationConfig(n_paths=10_000, seed=3044, antithetic=True))
    e = anti.success_probability()
    print(f"With antithetic variates:    {e}  (plain-MC SE would be {e.se_plain:.2%})")

    fat = simulate(plan, SimulationConfig(n_paths=10_000, seed=3044, model="t"))
    print(f"Fat-tailed (t, 5 df):        {fat.success_probability()}")

    guard = simulate(RetirementPlan(annual_spending=40_000, rule="guardrails"), CONFIG)
    g = guard.summary()
    print(f"Guardrails rule:             {guard.success_probability()}")
    print(f"  median lowest spending:    {g['median_lowest_spending']:.0%} of the starting $40,000")

    q = sequence_quintiles(res)
    print(f"\nSequence risk: success by quintile of returns in a {q['window']}-year window")
    print(f"{'quintile':>10} {'first years':>14} {'last years':>14}")
    for e_, l_ in zip(q["early"], q["late"]):
        print(f"{e_['quintile']:>10} {e_['success']:>14.1%} {l_['success']:>14.1%}")

    demo = reorder_demo(res)
    print(f"\nMedian path ({demo['annualized_return']:.2%}/yr) replayed in three orders:")
    for name, bal in demo["balances"].items():
        fy = demo["fail_year"][name]
        print(f"  {name:<18} " + (f"runs out in year {fy}" if fy else f"ends with {money(bal[-1])}"))

    fan_chart(res)
    plt.tight_layout(); plt.savefig(FIG / "fan_chart.png", dpi=150); plt.close()
    reorder_chart(demo)
    plt.tight_layout(); plt.savefig(FIG / "sequence_risk.png", dpi=150); plt.close()


def part2_grid() -> None:
    print("\n" + "=" * 72)
    print("2. Success by withdrawal rate and stock share (common random numbers)")
    print("=" * 72)
    rates = (0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065)
    stocks = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
    grid = success_grid(RetirementPlan(), CONFIG, rates, stocks)
    print("stocks " + " ".join(f"{r:>6.1%}" for r in rates))
    for s, row in zip(stocks, grid):
        print(f"{s:>6.0%} " + " ".join(f"{v:>6.0%}" for v in row))
    success_heatmap(grid, rates, stocks)
    plt.tight_layout(); plt.savefig(FIG / "success_grid.png", dpi=150); plt.close()


def part3_income() -> None:
    print("\n" + "=" * 72)
    print("3. Household: age 65, $1M, spends $80k/yr, Social Security $32k at FRA (67)")
    print("   Random lifespan (Gompertz, modal age 88, dispersion 10)")
    print("=" * 72)
    mort = GompertzMortality(modal_age=88, dispersion=10)
    print(f"Life expectancy at 65: {mort.life_expectancy(65):.1f} years; "
          f"P(reach 85) = {mort.survival(65, 20):.0%}, P(reach 95) = {mort.survival(65, 30):.0%}")

    fixed_rate = price_payout_rate(65, mort, discount_rate=0.045, load=0.05)
    real_rate = price_payout_rate(65, mort, discount_rate=0.020, load=0.05)
    dia_rate = price_payout_rate(65, mort, discount_rate=0.045, start_year=20, load=0.05)
    print(f"Priced payout rates: fixed SPIA {fixed_rate:.2%}, inflation-adjusted SPIA {real_rate:.2%}, "
          f"deferred to 85 (DIA) {dia_rate:.1%}")

    ss67 = social_security(32_000, claim_age=67, retirement_age=65)
    ss70 = social_security(32_000, claim_age=70, retirement_age=65)
    base = dict(initial_portfolio=1_000_000, annual_spending=80_000, mortality=mort, retirement_age=65)
    plans = {
        "SS at 67": RetirementPlan(**base, income=(ss67,)),
        "SS at 70": RetirementPlan(**base, income=(ss70,)),
        "SS at 67 + $300k fixed SPIA": RetirementPlan(
            **base, income=(ss67,), annuities=(Annuity(300_000, fixed_rate, name="Fixed SPIA"),)),
        "SS at 67 + $300k inflation-adj. SPIA": RetirementPlan(
            **base, income=(ss67,),
            annuities=(Annuity(300_000, real_rate, inflation_adjusted=True, name="Real SPIA"),)),
        "SS at 67 + $100k DIA at 85": RetirementPlan(
            **base, income=(ss67,), annuities=(Annuity(100_000, dia_rate, start_year=20, name="DIA"),)),
    }
    results = compare_scenarios(plans, CONFIG)
    summaries = {k: r.summary() for k, r in results.items()}
    print(f"\n{'scenario':<38}{'success':>9}{'median left':>14}{'floor if short':>16}")
    for k, s in summaries.items():
        floor = f"{s['median_floor_if_failed']:.0%}" if s["median_floor_if_failed"] is not None else "n/a"
        print(f"{k:<38}{s['success']:>9.1%}{money(s['median_terminal']):>14}{floor:>16}")
    print("\n'floor if short' = median lowest spending, as a share of $80k, among paths that ran short.")

    scenario_chart(summaries)
    plt.tight_layout(); plt.savefig(FIG / "income_scenarios.png", dpi=150); plt.close()


if __name__ == "__main__":
    part1_classic()
    part2_grid()
    part3_income()
    print(f"\nFigures saved to {FIG}/")
