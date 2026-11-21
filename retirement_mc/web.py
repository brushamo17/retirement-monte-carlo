"""Bridge between the browser demo and the simulator.

The interactive demo (``docs/index.html``) runs this package inside the browser
with Pyodide, a build of Python for WebAssembly. The page does no finance math
of its own: it collects the form into a dict, calls one of the functions in
``API`` below, and draws whatever comes back. Every number on the page therefore
comes from the same code, and with the same seed the same random paths, as the
results in the README.

Everything here takes and returns plain JSON-friendly values (dicts, lists,
numbers, strings, booleans, None). ``call`` is the single entry point the page
uses: a function name plus a JSON string in, a JSON string out.

The same functions work from ordinary Python::

    from retirement_mc import web
    out = web.run({"spending": 45_000, "ss_on": True})
    print(out["success"]["p"])
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import replace

import numpy as np

from .analysis import paired_difference, reorder_demo, sequence_quintiles
from .income import Annuity, IncomeStream, price_payout_rate, social_security
from .mortality import GompertzMortality
from .returns import HISTORICAL_US, MUTED_FORWARD, MarketAssumptions, generate_returns
from .simulation import (
    RetirementPlan,
    SimulationConfig,
    SimulationResult,
    lifespan_uniforms,
    simulate,
)

# Insurer assumptions used to price annuities (the same as run_example.py).
NOMINAL_RATE = 0.045    # discount rate for fixed-dollar annuities
REAL_RATE = 0.020       # discount rate for inflation-adjusted annuities
INSURER_LOAD = 0.05     # share of the premium the insurer keeps
DISPERSION = 10.0       # Gompertz dispersion; the page lets users move the modal age
DIA_START_AGE = 85      # when a deferred income annuity starts paying

SEQUENCE_END_AGE = 95   # with random lifespans, sequence charts use a fixed horizon to this age
DISPLAY_END_AGE = 100   # with random lifespans, charts stop at this age
MAX_GRID_PATHS = 5_000
MAX_SENSITIVITY_PATHS = 10_000

GRID_RATES = (0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065)
GRID_STOCKS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)

MARKETS = {"historical": HISTORICAL_US, "muted": MUTED_FORWARD}


def _market_fields(m: MarketAssumptions) -> dict:
    """Market assumptions as the page's form fields, rounded for display."""
    return {
        "stock_mean": round(m.stock_mean, 4), "stock_sd": round(m.stock_sd, 4),
        "bond_mean": round(m.bond_mean, 4), "bond_sd": round(m.bond_sd, 4),
        "correlation": round(m.correlation, 2), "inflation": round(m.inflation, 4),
    }


# Every input the page can send, with its default. Percentages are fractions
# (0.6 = 60%); money is in today's dollars.
DEFAULTS: dict = {
    # The household
    "portfolio": 1_000_000.0,
    "spending": 40_000.0,             # total yearly spending goal; income covers part of it
    "essential": 26_000.0,            # used only by the "essentials_first" rule
    "stock_share": 0.60,
    "fee": 0.005,
    "rule": "fixed",                  # "fixed", "guardrails" or "essentials_first"
    "target": 0.90,                   # the success rate the user is aiming for (display only)
    # Timeline
    "retirement_age": 65,
    "lifespan": "fixed",              # "fixed" horizon or "random" (Gompertz) lifespan
    "horizon_years": 30,
    "modal_age": 88.0,                # most common age at death, for random lifespans
    # Guaranteed income
    "ss_on": False, "ss_benefit": 32_000.0, "ss_claim_age": 67,
    "pension_on": False, "pension_amount": 20_000.0, "pension_start_age": 65, "pension_cola": True,
    "annuity_on": False, "annuity_premium": 300_000.0, "annuity_kind": "spia_real",
    # Markets
    "market": "historical",           # "historical", "muted" or "custom"
    **_market_fields(HISTORICAL_US),  # used when market == "custom"
    "model": "lognormal",             # "lognormal", "fat_tailed" or "bootstrap"
    # The experiment
    "n_paths": 10_000,
    "seed": 3044,
    "antithetic": False,
}

ANNUITY_KINDS = {
    # kind: (label, discount rate, inflation-adjusted, deferred to 85)
    "spia_real": ("Inflation-adjusted annuity", REAL_RATE, True, False),
    "spia_fixed": ("Fixed annuity", NOMINAL_RATE, False, False),
    "dia85": ("Deferred annuity from 85", NOMINAL_RATE, False, True),
}

# One-click starting points for the page. Each is a set of changes to DEFAULTS.
SCENARIOS = [
    {"id": "classic", "name": "Classic 4% rule",
     "description": "$1M, $40k a year adjusted for inflation, 60/40, 30 years. The textbook case.",
     "changes": {}},
    {"id": "early", "name": "Early retirement at 50",
     "description": "$1.5M at 50, spending $60k, 80% stocks, Social Security from 67, random lifespan.",
     "changes": {"portfolio": 1_500_000, "spending": 60_000, "stock_share": 0.8, "retirement_age": 50,
                 "lifespan": "random", "ss_on": True, "ss_benefit": 26_000, "ss_claim_age": 67}},
    {"id": "conservative", "name": "Conservative investor",
     "description": "The classic case with only 30% in stocks. Less volatility, but bonds barely beat inflation.",
     "changes": {"stock_share": 0.3}},
    {"id": "pension", "name": "Teacher with a pension",
     "description": "Retires at 62 with $400k and a $30k pension with no cost-of-living raises, plus Social Security at 67.",
     "changes": {"portfolio": 400_000, "spending": 60_000, "retirement_age": 62, "lifespan": "random",
                 "pension_on": True, "pension_amount": 30_000, "pension_start_age": 62, "pension_cola": False,
                 "ss_on": True, "ss_benefit": 14_000, "ss_claim_age": 67}},
    {"id": "ss67", "name": "Social Security at 67",
     "description": "$1M at 65, spending $80k, a $32k benefit claimed at 67, random lifespan.",
     "changes": {"spending": 80_000, "lifespan": "random", "ss_on": True}},
    {"id": "ss70", "name": "Delay Social Security to 70",
     "description": "The same household waits until 70 for a 24% larger benefit.",
     "changes": {"spending": 80_000, "lifespan": "random", "ss_on": True, "ss_claim_age": 70}},
    {"id": "annuity", "name": "Add an inflation-adjusted annuity",
     "description": "The same household moves $300k into a lifetime annuity that rises with prices.",
     "changes": {"spending": 80_000, "lifespan": "random", "ss_on": True, "annuity_on": True}},
    {"id": "flexible", "name": "Flexible spending",
     "description": "Of the $80k, $52k is essential. Guardrails cut only the discretionary $28k when markets fall.",
     "changes": {"spending": 80_000, "essential": 52_000, "rule": "essentials_first",
                 "lifespan": "random", "ss_on": True}},
]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def _num(v, default: float, lo: float, hi: float) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return min(max(v, lo), hi)


def _int(v, default: int, lo: int, hi: int) -> int:
    return int(round(_num(v, default, lo, hi)))


def _choice(v, options, default):
    return v if v in options else default


def normalize(inputs: dict | None = None) -> dict:
    """Fill in defaults and clamp every input to a sensible range.

    The page can send anything (an empty box, a typo), so bad *values* never
    raise: they fall back to the default or the nearest allowed value. Unknown
    *keys* do raise, because they mean the page and this module disagree about
    an input's name, which is a bug worth catching in the tests.
    """
    inputs = dict(inputs or {})
    unknown = set(inputs) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown inputs: {sorted(unknown)}")
    d = {**DEFAULTS, **inputs}
    x: dict = {}
    x["portfolio"] = _num(d["portfolio"], DEFAULTS["portfolio"], 0, 1e9)
    x["spending"] = _num(d["spending"], DEFAULTS["spending"], 0, 1e8)
    x["essential"] = _num(d["essential"], 0.65 * x["spending"], 0, x["spending"])
    x["stock_share"] = _num(d["stock_share"], DEFAULTS["stock_share"], 0, 1)
    x["fee"] = _num(d["fee"], DEFAULTS["fee"], 0, 0.03)
    x["rule"] = _choice(d["rule"], ("fixed", "guardrails", "essentials_first"), "fixed")
    x["target"] = _num(d["target"], DEFAULTS["target"], 0.5, 0.99)
    x["retirement_age"] = _int(d["retirement_age"], 65, 40, 80)
    x["lifespan"] = _choice(d["lifespan"], ("fixed", "random"), "fixed")
    x["horizon_years"] = _int(d["horizon_years"], 30, 5, 60)
    x["modal_age"] = _num(d["modal_age"], 88, 75, 100)
    x["ss_on"] = bool(d["ss_on"])
    x["ss_benefit"] = _num(d["ss_benefit"], DEFAULTS["ss_benefit"], 0, 200_000)
    x["ss_claim_age"] = _int(d["ss_claim_age"], 67, 62, 70)
    x["pension_on"] = bool(d["pension_on"])
    x["pension_amount"] = _num(d["pension_amount"], DEFAULTS["pension_amount"], 0, 1e6)
    x["pension_start_age"] = _int(d["pension_start_age"], 65, 40, 90)
    x["pension_cola"] = bool(d["pension_cola"])
    x["annuity_on"] = bool(d["annuity_on"])
    x["annuity_premium"] = _num(d["annuity_premium"], DEFAULTS["annuity_premium"], 0, x["portfolio"])
    x["annuity_kind"] = _choice(d["annuity_kind"], tuple(ANNUITY_KINDS), "spia_real")
    x["market"] = _choice(d["market"], ("historical", "muted", "custom"), "historical")
    x["stock_mean"] = _num(d["stock_mean"], DEFAULTS["stock_mean"], -0.5, 0.5)
    x["stock_sd"] = _num(d["stock_sd"], DEFAULTS["stock_sd"], 0, 1)
    x["bond_mean"] = _num(d["bond_mean"], DEFAULTS["bond_mean"], -0.5, 0.5)
    x["bond_sd"] = _num(d["bond_sd"], DEFAULTS["bond_sd"], 0, 1)
    x["correlation"] = _num(d["correlation"], DEFAULTS["correlation"], -0.95, 0.95)
    x["inflation"] = _num(d["inflation"], DEFAULTS["inflation"], -0.05, 0.2)
    x["model"] = _choice(d["model"], ("lognormal", "fat_tailed", "bootstrap"), "lognormal")
    n = _int(d["n_paths"], 10_000, 100, 50_000)
    x["n_paths"] = n + (n % 2)            # antithetic pairs need an even number
    x["seed"] = _int(d["seed"], 3044, 0, 2**32 - 1)
    x["antithetic"] = bool(d["antithetic"]) and x["model"] != "bootstrap"
    return x


def market_for(x: dict) -> MarketAssumptions:
    """The capital market assumptions an input dict asks for.

    The named presets use the package's exact assumptions (not the rounded
    numbers shown on the page), so the demo reproduces the README to the decimal.
    """
    if x["market"] in MARKETS:
        return MARKETS[x["market"]]
    return MarketAssumptions(
        stock_mean=x["stock_mean"], stock_sd=x["stock_sd"], bond_mean=x["bond_mean"],
        bond_sd=x["bond_sd"], correlation=x["correlation"], inflation=x["inflation"], label="Custom",
    )


def make_annuity(kind: str, premium: float, age: int, mortality: GompertzMortality) -> Annuity:
    """An annuity priced from the same mortality model as the simulated lifespans."""
    name, rate, real, deferred = ANNUITY_KINDS[kind]
    start = max(1, DIA_START_AGE - age) if deferred else 0
    payout = price_payout_rate(age, mortality, rate, start_year=start, load=INSURER_LOAD)
    return Annuity(premium, payout, start_year=start, inflation_adjusted=real, name=name)


def build(inputs: dict | None = None) -> tuple[RetirementPlan, SimulationConfig]:
    """Turn the page's inputs into the package's plan and config objects."""
    x = normalize(inputs)
    age = x["retirement_age"]
    mortality = GompertzMortality(modal_age=x["modal_age"], dispersion=DISPERSION)
    income = []
    if x["ss_on"] and x["ss_benefit"] > 0:
        income.append(social_security(x["ss_benefit"], x["ss_claim_age"], age))
    if x["pension_on"] and x["pension_amount"] > 0:
        income.append(IncomeStream(
            name="Pension", amount=x["pension_amount"],
            start_year=max(0, x["pension_start_age"] - age), inflation_adjusted=x["pension_cola"],
        ))
    annuities = []
    if x["annuity_on"] and x["annuity_premium"] > 0:
        annuities.append(make_annuity(x["annuity_kind"], x["annuity_premium"], age, mortality))
    plan = RetirementPlan(
        initial_portfolio=x["portfolio"],
        annual_spending=x["spending"],
        stock_share=x["stock_share"],
        fee=x["fee"],
        horizon_years=x["horizon_years"],
        rule=x["rule"],
        essential_spending=x["essential"] if x["rule"] == "essentials_first" else None,
        income=tuple(income),
        annuities=tuple(annuities),
        retirement_age=age,
        mortality=mortality if x["lifespan"] == "random" else None,
    )
    config = SimulationConfig(
        n_paths=x["n_paths"], seed=x["seed"], model=x["model"],
        antithetic=x["antithetic"], market=market_for(x),
    )
    return plan, config


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _draws(config: SimulationConfig, n_years: int):
    """Market paths for ``n_years``. The same seed and shape give the same
    underlying random numbers whatever the market assumptions, which is what
    makes comparisons between assumptions low-noise (common random numbers)."""
    return generate_returns(
        config.market, config.n_paths, n_years, model=config.model,
        antithetic=config.antithetic, seed=config.seed, block_size=config.block_size,
    )


def _simulate(plan, config, returns=None, uniforms=None) -> tuple[SimulationResult, list[str]]:
    """``simulate`` with warnings collected as notes for the page instead of printed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = simulate(plan, config, returns, uniforms=uniforms)
    return result, [str(w.message) for w in caught]


def display_years(plan: RetirementPlan) -> int:
    """How many years the charts show: the horizon, or up to age 100 with random lifespans."""
    if plan.mortality is None:
        return plan.n_years
    return min(plan.n_years, max(15, DISPLAY_END_AGE - plan.retirement_age))


def depletion_year(result: SimulationResult) -> np.ndarray:
    """Per path, the first year (0-based) the portfolio can't fund that year's plan,
    whether or not the retiree is still alive; ``inf`` if it never happens.

    Uses the same test as the engine (balance < spending target - income), so
    for anyone still alive it agrees with ``fail_year``."""
    need = np.maximum(result.target - result.income[None, :], 0.0)
    short = result.balances[:, :-1] < need
    first = np.argmax(short, axis=1).astype(float)
    first[~short.any(axis=1)] = np.inf
    return first


def _curves(result: SimulationResult) -> dict:
    """Age-by-age curves: is the money still there, is the retiree still alive?"""
    plan = result.plan
    D = display_years(plan)
    t = np.arange(D + 1)
    d = depletion_year(result)
    life = result.lifespan
    funded = (d[:, None] > t[None, :]).mean(axis=0)
    out = {
        "ages": (plan.retirement_age + t).tolist(),
        "funded": funded,
        "alive": None,
        "alive_depleted": None,
    }
    if plan.mortality is not None:
        alive = life[:, None] > t[None, :]
        out["alive"] = alive.mean(axis=0)
        out["alive_depleted"] = (alive & (d[:, None] <= t[None, :])).mean(axis=0)
    return out


def _success(result: SimulationResult) -> dict:
    e = result.success_probability()
    return {"p": e.p, "se": e.se, "ci_low": e.ci_low, "ci_high": e.ci_high,
            "se_plain": e.se_plain, "n": int(result.success.size)}


def _first_year_withdrawal(plan: RetirementPlan, inflation: float) -> dict:
    income0 = float(plan.income_schedule(inflation)[0]) if plan.n_years else 0.0
    need = max(plan.annual_spending - income0, 0.0)
    invested = plan.invested_portfolio
    return {"income": income0, "withdrawal": need, "invested": invested,
            "rate": need / invested if invested > 0 else None}


def _fail_age(result: SimulationResult) -> float | None:
    """Median age at the first shortfall, among retirees who ran short while alive."""
    short = result.fail_year > 0
    if not short.any():
        return None
    return float(result.plan.retirement_age + np.median(result.fail_year[short]) - 1)


def _brief(result: SimulationResult, inflation: float) -> dict:
    """The headline numbers for one plan."""
    s = result.summary()
    D = display_years(result.plan)
    return {
        "success": _success(result),
        "median_terminal": s["median_terminal"],
        "p10_terminal": s["p10_terminal"],
        "funded": s["funded"],
        "fail_age": _fail_age(result),
        "median_lowest_spending": s["median_lowest_spending"],
        "first_year": _first_year_withdrawal(result.plan, inflation),
        "retirement_age": result.plan.retirement_age,
        "median_path": np.median(result.balances[:, : D + 1], axis=0),
        "curves": _curves(result),
    }


def _jsonable(v):
    """Convert NumPy values to plain Python, and NaN/inf to None (JSON has neither)."""
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(i) for i in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else None
    return v


# --------------------------------------------------------------------------
# What the page calls
# --------------------------------------------------------------------------

def presets() -> dict:
    """Default inputs, one-click scenarios and the market presets' display values."""
    scenarios = [
        {"id": s["id"], "name": s["name"], "description": s["description"],
         "inputs": normalize(s["changes"])}
        for s in SCENARIOS
    ]
    return {
        "defaults": normalize({}),
        "scenarios": scenarios,
        "markets": {k: _market_fields(m) for k, m in MARKETS.items()},
    }


def run(inputs: dict | None = None) -> dict:
    """Simulate one plan and return everything the main page shows."""
    x = normalize(inputs)
    plan, config = build(x)
    result, notes = _simulate(plan, config)
    inflation = config.market.inflation
    s = result.summary()
    age = plan.retirement_age
    D = display_years(plan)

    # Guaranteed income, stream by stream, for the "where the money comes from" chart.
    streams = []
    for st in plan.all_income():
        info = {"name": st.name, "amount": st.amount, "start_age": age + st.start_year,
                "inflation_adjusted": st.inflation_adjusted,
                "values": st.real_schedule(plan.n_years, inflation)[:D]}
        streams.append(info)
    for a, info in zip(plan.annuities, streams[len(plan.income):]):
        info.update(premium=a.premium, payout_rate=a.payout_rate)

    # Sequence-of-returns risk needs a fixed horizon: with random lifespans,
    # use the same plan with a fixed horizon to age 95.
    if plan.mortality is None:
        seq_result = result
    else:
        seq_plan = replace(plan, mortality=None, horizon_years=max(15, SEQUENCE_END_AGE - age))
        seq_result, _ = _simulate(seq_plan, config)
    reorder = reorder_demo(seq_result)

    # The same plan under the other market preset, to show model risk.
    alt = None
    if config.model != "bootstrap":
        other = HISTORICAL_US if x["market"] == "muted" else MUTED_FORWARD
        alt_result, _ = _simulate(plan, replace(config, market=other))
        alt = {"label": other.label, "p": float(alt_result.success.mean())}

    longevity = None
    if plan.mortality is not None:
        m = plan.mortality
        reach = lambda target_age: float(m.survival(age, target_age - age)) if target_age > age else 1.0  # noqa: E731
        longevity = {"life_expectancy": m.life_expectancy(age), "p85": reach(85), "p95": reach(95)}

    out = {
        "inputs": x,
        "retirement_age": age,
        "n_years": plan.n_years,
        "display_years": D,
        "random_lifespan": plan.mortality is not None,
        "success": _success(result),
        "summary": s,
        "fail_age": _fail_age(result),
        "first_year": _first_year_withdrawal(plan, inflation),
        "percentiles": result.percentiles()[:, : D + 1],
        "curves": _curves(result),
        "income": {"streams": streams, "spending": plan.annual_spending,
                   "essential": plan.essential_spending, "ages": list(range(age, age + D))},
        "longevity": longevity,
        "sequence": {
            "horizon": seq_result.n_years,
            "quintiles": sequence_quintiles(seq_result),
            "reorder": {
                "annualized_return": reorder["annualized_return"],
                "names": list(reorder["balances"]),
                "balances": list(reorder["balances"].values()),
                "fail_year": list(reorder["fail_year"].values()),
            },
        },
        "alt": alt,
        "market_label": config.market.label if config.model != "bootstrap" else "Historical block bootstrap, 1928-2025",
        "notes": notes,
    }
    return out


def grid_rates(plan: RetirementPlan) -> tuple[list[float], bool]:
    """Spending rates for the heatmap's columns.

    Without guaranteed income, the classic 2.5%-6.5% grid. With income, or when
    the plan spends outside that range, nine columns centered on the plan's own
    rate: a household whose Social Security covers $32k would otherwise see
    a grid of near-100% cells that says nothing.
    """
    r0 = plan.annual_spending / plan.initial_portfolio if plan.initial_portfolio > 0 else 0.04
    if not plan.all_income() and GRID_RATES[0] <= r0 <= GRID_RATES[-1]:
        return list(GRID_RATES), False
    step = 0.005 if r0 < 0.10 else 0.01
    start = max(step, round(r0 / step) * step - 4 * step)
    return [round(start + k * step, 4) for k in range(len(GRID_RATES))], True


def grid(inputs: dict | None = None, max_paths: int = MAX_GRID_PATHS) -> dict:
    """Success for every (stock share, spending rate) pair, on shared market paths.

    Spending is set to ``rate x starting portfolio`` and everything else (income,
    lifespan, rule) stays as the user set it. With an essentials split, the
    essential share of spending stays the same.
    """
    plan, config = build(inputs)
    rates, centered = grid_rates(plan)
    config = replace(config, n_paths=min(config.n_paths, max_paths))
    returns = _draws(config, plan.n_years)
    u = lifespan_uniforms(config)
    ratio = None
    if plan.essential_spending is not None and plan.annual_spending > 0:
        ratio = plan.essential_spending / plan.annual_spending
    values = []
    for w in GRID_STOCKS:
        row = []
        for r in rates:
            spend = r * plan.initial_portfolio
            cell = replace(plan, stock_share=w, annual_spending=spend,
                           essential_spending=None if ratio is None else ratio * spend)
            res, _ = _simulate(cell, config, returns, u)
            row.append(float(res.success.mean()))
        values.append(row)
    return {
        "rates": rates, "stocks": list(GRID_STOCKS), "values": values,
        "n_paths": config.n_paths, "centered": centered,
        "current_rate": plan.annual_spending / plan.initial_portfolio if plan.initial_portfolio else None,
        "current_stock": plan.stock_share,
    }


def compare(a: dict | None = None, b: dict | None = None) -> dict:
    """Run two plans on the same market paths and lifespans (common random numbers).

    Plan B uses plan A's simulation settings (paths, seed, return model), so the
    two plans face identical luck and the difference between them is the plan,
    not noise. The paired confidence interval shows how much tighter that is
    than comparing two independent runs.
    """
    xa, xb = normalize(a), normalize(b)
    for k in ("n_paths", "seed", "model", "antithetic"):
        xb[k] = xa[k]
    pa, ca = build(xa)
    pb, cb = build(xb)
    n_years = max(pa.n_years, pb.n_years)
    u = lifespan_uniforms(ca)
    ra, _ = _simulate(pa, ca, _draws(ca, n_years), u)
    rb, _ = _simulate(pb, cb, _draws(cb, n_years), u)
    diff = paired_difference(ra, rb)
    sa, sb = ra.success_probability(), rb.success_probability()
    notes = []
    if pa.n_years != pb.n_years:
        notes.append("The plans simulate different numbers of years, so they share the first years of "
                     "each market path. Each plan's numbers can differ slightly from the main view.")
    return {
        "a": _brief(ra, ca.market.inflation),
        "b": _brief(rb, cb.market.inflation),
        "diff": {"p": diff.p, "se": diff.se, "ci_low": diff.ci_low, "ci_high": diff.ci_high},
        "independent_se": math.sqrt(sa.se_plain**2 + sb.se_plain**2),
        "notes": notes,
    }


def sensitivity(inputs: dict | None = None, max_paths: int = MAX_SENSITIVITY_PATHS) -> dict:
    """Which assumption moves the success rate most? (Data for a tornado chart.)

    Each assumption is moved down and up by a fixed, plausible amount while
    everything else stays put, and every run uses the same market paths and
    lifespans as the base case. Two caveats belong next to the chart: the
    ranking depends on the ranges chosen, and moving one input at a time
    ignores how they interact.
    """
    x = normalize(inputs)
    x["n_paths"] = min(x["n_paths"], max_paths)
    base_plan, base_config = build(x)
    m = base_config.market
    exact_market = {"market": "custom", "stock_mean": m.stock_mean, "stock_sd": m.stock_sd,
                    "bond_mean": m.bond_mean, "bond_sd": m.bond_sd, "correlation": m.correlation,
                    "inflation": m.inflation}
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731

    factors = []  # (name, note, low changes, high changes, low label, high label)
    if x["model"] != "bootstrap":
        factors.append(("Stock returns", "Expected real return, ±2 points",
                        {**exact_market, "stock_mean": m.stock_mean - 0.02},
                        {**exact_market, "stock_mean": m.stock_mean + 0.02},
                        pct(m.stock_mean - 0.02), pct(m.stock_mean + 0.02)))
        factors.append(("Bond returns", "Expected real return, ±1 point",
                        {**exact_market, "bond_mean": m.bond_mean - 0.01},
                        {**exact_market, "bond_mean": m.bond_mean + 0.01},
                        pct(m.bond_mean - 0.01), pct(m.bond_mean + 0.01)))
        lo_sd = max(0.0, m.stock_sd - 0.04)
        factors.append(("Stock volatility", "Standard deviation, ±4 points",
                        {**exact_market, "stock_sd": lo_sd}, {**exact_market, "stock_sd": m.stock_sd + 0.04},
                        pct(lo_sd), pct(m.stock_sd + 0.04)))
    factors.append(("Inflation", "2% vs 4% a year",
                    {**exact_market, "inflation": 0.02}, {**exact_market, "inflation": 0.04}, "2.0%", "4.0%"))
    s = x["spending"]
    factors.append(("Spending", "±10%",
                    {"spending": 0.9 * s, "essential": 0.9 * x["essential"]},
                    {"spending": 1.1 * s, "essential": 1.1 * x["essential"]},
                    f"${0.9 * s:,.0f}", f"${1.1 * s:,.0f}"))
    lo_fee = max(0.0, x["fee"] - 0.0025)
    factors.append(("Fees", "±0.25 points a year", {"fee": lo_fee}, {"fee": x["fee"] + 0.0025},
                    f"{lo_fee * 100:.2f}%", f"{(x['fee'] + 0.0025) * 100:.2f}%"))
    if x["lifespan"] == "random":
        age = x["retirement_age"]
        lo_m, hi_m = x["modal_age"] - 3, x["modal_age"] + 3
        le = lambda modal: GompertzMortality(modal, DISPERSION).life_expectancy(age)  # noqa: E731
        factors.append(("Longevity", "Most common age at death, ±3 years",
                        {"modal_age": lo_m}, {"modal_age": hi_m},
                        f"{le(lo_m):.1f} yrs left", f"{le(hi_m):.1f} yrs left"))
    else:
        h = x["horizon_years"]
        factors.append(("Retirement length", "±5 years", {"horizon_years": max(5, h - 5)},
                        {"horizon_years": h + 5}, f"{max(5, h - 5)} years", f"{h + 5} years"))

    variants = [(name, side, build({**x, **ch})) for name, _, lo, hi, _, _ in factors
                for side, ch in (("low", lo), ("high", hi))]
    n_years = max([base_plan.n_years] + [p.n_years for _, _, (p, _) in variants])
    u = lifespan_uniforms(base_config)
    cache: dict = {}

    def draws(config):
        if config.market not in cache:
            cache[config.market] = _draws(config, n_years)
        return cache[config.market]

    base, _ = _simulate(base_plan, base_config, draws(base_config), u)
    runs = {}
    for name, side, (p, c) in variants:
        res, _ = _simulate(p, c, draws(c), u)
        runs[(name, side)] = (float(res.success.mean()), paired_difference(base, res).se)

    rows = []
    for name, note, _, _, lo_label, hi_label in factors:
        (p_lo, se_lo), (p_hi, se_hi) = runs[(name, "low")], runs[(name, "high")]
        rows.append({"name": name, "note": note, "low_label": lo_label, "high_label": hi_label,
                     "p_low": p_lo, "p_high": p_hi, "se_low": se_lo, "se_high": se_hi})
    p0 = float(base.success.mean())
    rows.sort(key=lambda r: -max(abs(r["p_low"] - p0), abs(r["p_high"] - p0)))

    notes = []
    if all(st.inflation_adjusted for st in base_plan.all_income()):
        notes.append("Inflation has no effect on this plan: everything is in today's dollars, and none of its "
                     "income is a fixed dollar amount. Add a pension without raises or a fixed annuity to see it matter.")
    if x["model"] == "bootstrap":
        notes.append("The historical bootstrap replays actual returns, so return and volatility "
                     "assumptions don't apply. Switch to lognormal or fat tails to include them.")
    return {"base": p0, "n_paths": base_config.n_paths, "factors": rows, "notes": notes}


API = {"presets": presets, "run": run, "grid": grid, "compare": compare, "sensitivity": sensitivity}


def call(name: str, payload: str = "{}") -> str:
    """Single entry point for the page: ``call("run", '{"inputs": {...}}')`` returns JSON."""
    if name not in API:
        raise ValueError(f"unknown function {name!r}; expected one of {sorted(API)}")
    kwargs = json.loads(payload) if payload else {}
    return json.dumps(_jsonable(API[name](**kwargs)), allow_nan=False)
