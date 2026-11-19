"""Robustness experiments: how much do the headline results depend on the model?

    python research.py

Runs seven experiments, prints the results, saves them to research_results.txt
and saves figures/model_risk.png and figures/essentials.png. Takes under a minute.

A. Model risk: success across withdrawal rates under four return models, plus
   every historical 30-year window since 1928.
A2. Sensitivity: change one assumption at a time to see which one matters.
B. Convergence: how the Monte Carlo error shrinks with the number of paths.
C. The cost of guardrails: what retirees give up for a higher success rate.
D. Annuity trade-off: a paired (common random numbers) comparison with a
   confidence interval on the *difference*.
E. Historical backtest: which actual retirement start years would have failed.
F. Essential vs discretionary spending: protecting essentials first.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from retirement_mc import (
    HISTORICAL_US,
    MUTED_FORWARD,
    Annuity,
    GompertzMortality,
    RetirementPlan,
    SimulationConfig,
    compare_scenarios,
    historical_backtest,
    paired_difference,
    price_payout_rate,
    simulate,
    social_security,
)
from retirement_mc.plots import essentials_chart, model_risk_chart

ROOT = Path(__file__).parent
N = 50_000
SEED = 3044
RATES = (0.030, 0.035, 0.040, 0.045, 0.050)
BASE = RetirementPlan(initial_portfolio=1_000_000, annual_spending=40_000, stock_share=0.6)

MODELS = {
    "Lognormal (1928-2025 calibration)": SimulationConfig(N, SEED, "lognormal", market=HISTORICAL_US),
    "Fat-tailed (1928-2025 calibration)": SimulationConfig(N, SEED, "fat_tailed", market=HISTORICAL_US),
    "Block bootstrap of 1928-2025": SimulationConfig(N, SEED, "bootstrap", block_size=5),
    "Lognormal (muted forward-looking)": SimulationConfig(N, SEED, "lognormal", market=MUTED_FORWARD),
}


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def experiment_a() -> dict[str, list[float]]:
    header("A. Model risk: success of a 60/40, 30-year plan by withdrawal rate")
    curves: dict[str, list[float]] = {}
    for name, cfg in MODELS.items():
        curves[name] = [simulate(replace(BASE, annual_spending=r * 1e6), cfg).success.mean() for r in RATES]
    curves["Historical 30-year windows (69, overlapping)"] = [
        historical_backtest(replace(BASE, annual_spending=r * 1e6))["success_rate"] for r in RATES
    ]
    print(f"{'model':<46}" + "".join(f"{r:>8.1%}" for r in RATES))
    for name, vals in curves.items():
        print(f"{name:<46}" + "".join(f"{v:>8.1%}" for v in vals))
    i4 = RATES.index(0.040)
    spread = max(v[i4] for v in curves.values()) - min(v[i4] for v in curves.values())
    print(f"\nAt 4%, the models span {spread * 100:.1f} percentage points.")

    print("\nBootstrap block length (4% rule): longer blocks keep more multi-year history")
    for b in (1, 5, 10):
        cfg = SimulationConfig(N, SEED, "bootstrap", block_size=b)
        print(f"  {b:>2}-year blocks: {simulate(BASE, cfg).success_probability()}")
    return curves


def experiment_a2() -> None:
    header("A2. Sensitivity: change one assumption at a time (4% rule, 60/40, 30 years)")
    h, m = HISTORICAL_US, MUTED_FORWARD
    variants = {
        "Historical assumptions (baseline)": h,
        "Expected returns lowered only (5.5% / 1.5%)": replace(h, stock_mean=m.stock_mean, bond_mean=m.bond_mean),
        "Volatility lowered only (17% / 6%)": replace(h, stock_sd=m.stock_sd, bond_sd=m.bond_sd),
        "Correlation raised only (0.20)": replace(h, correlation=m.correlation),
        "All muted assumptions together": m,
    }
    base = simulate(BASE, SimulationConfig(N, SEED, market=h)).success.mean()
    print(f"{'assumption changed':<46}{'success':>9}{'change':>10}")
    for name, market in variants.items():
        p = simulate(BASE, SimulationConfig(N, SEED, market=market)).success.mean()
        print(f"{name:<46}{p:>9.1%}{(p - base) * 100:>+9.1f}pp")
    print("Inflation is left out: it only affects nominal income, and this plan has none.")
    print("Every row uses the same random draws (CRN), so the changes are not sampling noise.")


def experiment_b() -> None:
    header("B. Monte Carlo convergence (4% rule, lognormal)")
    print(f"{'paths':>9}{'success':>10}{'SE':>10}{'95% CI':>22}")
    ses = []
    for n in (1_000, 10_000, 100_000):
        e = simulate(BASE, SimulationConfig(n, SEED)).success_probability()
        ses.append(e.se)
        print(f"{n:>9,}{e.p:>10.1%}{e.se * 100:>9.2f}pp{f'{e.ci_low:.1%} to {e.ci_high:.1%}':>22}")
    print(f"SE falls {1 - ses[-1] / ses[0]:.0%} from 1,000 to 100,000 paths (theory: 1 - 1/sqrt(100) = 90%).")


def experiment_c() -> None:
    header("C. What guardrails cost (4% start, 60/40, 30 years)")
    rows = [
        ("fixed", HISTORICAL_US), ("guardrails", HISTORICAL_US),
        ("fixed", MUTED_FORWARD), ("guardrails", MUTED_FORWARD),
    ]
    print(f"{'rule':<11}{'market':<26}{'never ran out':>14}{'success*':>10}{'any cut':>9}"
          f"{'<80% of plan':>14}{'median low':>12}")
    for rule, market in rows:
        res = simulate(replace(BASE, rule=rule), SimulationConfig(N, SEED, market=market))
        s = res.summary()
        label = "1928-2025" if market is HISTORICAL_US else "muted forward"
        print(f"{rule:<11}{label:<26}{s['funded']:>14.1%}{s['success']:>10.1%}{s['p_any_cut']:>9.1%}"
              f"{s['p_below_floor']:>14.1%}{s['median_lowest_spending'] * 40_000:>12,.0f}")
    print("* success = never ran out AND spending never fell below 80% of the original $40,000")


def experiment_d() -> None:
    header("D. Annuity trade-off, paired comparison (age 65, $1M, $80k spending, SS $32k at 67)")
    mort = GompertzMortality()
    real_rate = price_payout_rate(65, mort, discount_rate=0.020, load=0.05)
    ss = social_security(32_000, claim_age=67, retirement_age=65)
    base = dict(initial_portfolio=1_000_000, annual_spending=80_000, mortality=mort)
    plans = {
        "No annuity": RetirementPlan(**base, income=(ss,)),
        "$300k inflation-adjusted SPIA": RetirementPlan(
            **base, income=(ss,), annuities=(Annuity(300_000, real_rate, inflation_adjusted=True),)),
    }
    res = compare_scenarios(plans, SimulationConfig(N, SEED))
    print(f"{'':<32}{'success':>9}{'median left':>14}{'floor if short':>16}")
    for k, r in res.items():
        s = r.summary()
        print(f"{k:<32}{s['success']:>9.1%}{s['median_terminal']:>14,.0f}{s['median_floor_if_short']:>16.0%}")
    d = paired_difference(res["No annuity"], res["$300k inflation-adjusted SPIA"])
    ind = np.hypot(res["No annuity"].success_probability().se,
                   res["$300k inflation-adjusted SPIA"].success_probability().se)
    print(f"\nPaired improvement in success: {d.p * 100:.2f} pp "
          f"(95% CI {d.ci_low * 100:.2f} to {d.ci_high * 100:.2f} pp)")
    print(f"SE of the difference: {d.se * 100:.3f} pp paired vs {ind * 100:.3f} pp if the runs were independent "
          f"({(ind / d.se) ** 2:.0f}x fewer paths needed for the same precision).")


def experiment_e() -> None:
    header("E. Historical backtest: 4% rule, 60/40, every 30-year retirement starting 1928-1996")
    bt = historical_backtest(BASE)
    failed = bt["start_years"][~bt["success"]]
    print(f"Success in {bt['success'].sum()} of {bt['success'].size} start years ({bt['success_rate']:.0%}).")
    print("Failed start years: " + (", ".join(map(str, failed)) if failed.size else "none"))
    worst = np.argsort(bt["ending_balance"])[:5]
    print("Lowest ending balances: " + ", ".join(
        f"{bt['start_years'][i]} (${bt['ending_balance'][i]:,.0f})" for i in worst))


def experiment_f() -> dict[str, dict]:
    header("F. Essential vs discretionary spending (age 65, $1M, $80k = $52k essential + $28k discretionary, SS $32k at 67)")
    mort = GompertzMortality()
    real_rate = price_payout_rate(65, mort, discount_rate=0.020, load=0.05)
    ss = social_security(32_000, claim_age=67, retirement_age=65)
    essential = 52_000
    # Premium for an inflation-adjusted SPIA that, with Social Security, covers the essentials.
    premium = round((essential - ss.amount) / real_rate, -3)
    base = dict(initial_portfolio=1_000_000, annual_spending=80_000, essential_spending=essential,
                mortality=mort, income=(ss,))
    plans = {
        "Fixed spending": RetirementPlan(**base),
        "Guardrails (cut everything)": RetirementPlan(**base, rule="guardrails"),
        "Essentials-first guardrails": RetirementPlan(**base, rule="essentials_first"),
        "Floor and upside": RetirementPlan(
            **base, rule="essentials_first",
            annuities=(Annuity(premium, real_rate, inflation_adjusted=True, name="Real SPIA"),)),
    }
    print(f"'Floor and upside' buys a ${premium:,.0f} inflation-adjusted SPIA paying "
          f"${premium * real_rate:,.0f}/yr, so annuity + Social Security cover the essentials from 67.")
    res = compare_scenarios(plans, SimulationConfig(N, SEED))
    out = {}
    print(f"\n{'strategy':<30}{'essentials':>11}{'never ran':>10}{'years below':>12}{'discretionary':>14}{'median left':>13}")
    print(f"{'':<30}{'covered':>11}{'out':>10}{'full plan':>12}{'enjoyed':>14}{'at death':>13}")
    for k, r in res.items():
        s = r.summary()
        out[k] = s
        print(f"{k:<30}{s['floor_met']:>11.1%}{s['funded']:>10.1%}{s['share_years_below_plan']:>12.0%}"
              f"{s['median_discretionary_share']:>14.0%}{s['median_terminal']:>13,.0f}")
    d = paired_difference(res["Guardrails (cut everything)"], res["Essentials-first guardrails"])
    print(f"\nEssentials-first vs cut-everything guardrails, success (essentials always covered and never ran out): "
          f"{d.p * 100:+.1f} pp (95% CI {d.ci_low * 100:+.1f} to {d.ci_high * 100:+.1f})")
    print("essentials covered = actual spending never fell below $52k in any year lived")
    print("years below full plan = average share of years lived with spending under $80k")
    print("discretionary enjoyed = median share of the planned $28k/yr actually spent over a lifetime")
    return out


def main() -> None:
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        curves = experiment_a()
        experiment_a2()
        experiment_b()
        experiment_c()
        experiment_d()
        experiment_e()
        ess = experiment_f()

    (ROOT / "research_results.txt").write_text(buf.getvalue())
    model_risk_chart(RATES, curves)
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "model_risk.png", dpi=150)
    plt.close()
    essentials_chart(ess)
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "essentials.png", dpi=150)
    plt.close()
    print("\nSaved research_results.txt, figures/model_risk.png and figures/essentials.png")


if __name__ == "__main__":
    main()
