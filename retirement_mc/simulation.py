"""Core Monte Carlo engine.

Design choice: the simulation is vectorized *across paths* and loops *over
years*. Each year depends on the balance from the year before, so the time
dimension is inherently sequential, but every path is independent of every
other path. Looping over ~30 years while NumPy processes 10,000+ paths at once
keeps the code readable and fast (well under a second for 10,000 x 30).

One simulated year, for every path at once:

    1. (Guardrails only) adjust the spending target.
    2. Portfolio withdrawal W = max(spending - guaranteed income, 0).
    3. If the balance B cannot cover W and the retiree is alive, record a
       shortfall. Withdraw what is there.
    4. Grow what is left: B <- (B - W) * (1 + r), where r is the portfolio's
       real return net of fees. The portfolio is rebalanced to its target mix
       every year, so r = w * r_stock + (1 - w) * r_bond - fee.

What counts as success
----------------------
A retirement succeeds if, in every year the retiree is alive,

    (a) that year's spending target was fully funded (no shortfall), and
    (b) actual spending never fell below ``spending_floor`` x the original goal.

Condition (b) matters for flexible spending rules. Guardrails avoid running
out of money by cutting spending, so without (b) a retiree whose spending was
cut to 60% of the plan would still count as a "success". The default floor is
80%. Under the fixed rule spending only drops when money runs out, so (b) adds
nothing beyond (a).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .income import Annuity, IncomeStream
from .mortality import GompertzMortality
from .returns import HISTORICAL_US, MarketAssumptions, generate_returns

# Guardrail parameters (a simplified Guyton-Klinger rule).
UPPER_GUARDRAIL = 1.20      # cut when the withdrawal rate is 20% above its reference
LOWER_GUARDRAIL = 0.80      # raise when it is 20% below
CUT = 0.10                  # size of a spending cut
RAISE = 0.10                # size of a spending raise
NO_CUTS_FINAL_YEARS = 15    # Guyton-Klinger skip cuts late in retirement

# If more than this share of lifespans outlive the simulation horizon, warn.
BEYOND_HORIZON_WARNING = 0.001


@dataclass(frozen=True)
class RetirementPlan:
    """Everything about the household and its strategy. Dollars are today's dollars.

    Attributes:
        initial_portfolio: Investable assets at retirement, before any annuity purchase.
        annual_spending: Total yearly spending goal. Guaranteed income covers
            part of it; the portfolio covers the rest.
        stock_share: Fraction of the portfolio in stocks (the rest is bonds).
        fee: Annual fees as a fraction of assets (advisory + fund expenses).
        horizon_years: Length of retirement when ``mortality`` is None.
        rule: ``"fixed"`` (same real spending every year) or ``"guardrails"``.
        spending_floor: Lowest acceptable spending, as a fraction of
            ``annual_spending``. A path where spending drops below it in any
            year lived counts as a failure even if every target was funded.
        income: Social Security, pensions and other income streams.
        annuities: Annuities bought with part of the portfolio at retirement.
        retirement_age: Age at the start of the simulation.
        mortality: If given, each path gets a random lifespan and success means
            "spending was sustained as long as the retiree lived".
        max_age: Last age simulated when ``mortality`` is used. The default of
            115 leaves well under a 1-in-a-million chance of outliving the
            horizon under the default Gompertz model.
    """

    initial_portfolio: float = 1_000_000.0
    annual_spending: float = 40_000.0
    stock_share: float = 0.60
    fee: float = 0.005
    horizon_years: int = 30
    rule: str = "fixed"
    spending_floor: float = 0.80
    income: tuple[IncomeStream, ...] = ()
    annuities: tuple[Annuity, ...] = ()
    retirement_age: int = 65
    mortality: GompertzMortality | None = None
    max_age: int = 115

    def __post_init__(self) -> None:
        if self.rule not in ("fixed", "guardrails"):
            raise ValueError("rule must be 'fixed' or 'guardrails'")
        if not 0.0 <= self.stock_share <= 1.0:
            raise ValueError("stock_share must be between 0 and 1")
        if not 0.0 <= self.spending_floor <= 1.0:
            raise ValueError("spending_floor must be between 0 and 1")
        if self.initial_portfolio < 0 or self.annual_spending < 0:
            raise ValueError("portfolio and spending must be non-negative")
        if self.invested_portfolio < 0:
            raise ValueError("annuity premiums exceed the starting portfolio")
        if self.n_years < 1:
            raise ValueError("the simulation needs at least one year")

    @property
    def n_years(self) -> int:
        """Number of simulated years."""
        if self.mortality is None:
            return self.horizon_years
        return self.max_age - self.retirement_age

    @property
    def invested_portfolio(self) -> float:
        """Portfolio left to invest after paying annuity premiums."""
        return self.initial_portfolio - sum(a.premium for a in self.annuities)

    def all_income(self) -> tuple[IncomeStream, ...]:
        return tuple(self.income) + tuple(a.income_stream() for a in self.annuities)

    def income_schedule(self, inflation: float) -> np.ndarray:
        """Total guaranteed income per year in today's dollars, shape ``(n_years,)``."""
        total = np.zeros(self.n_years)
        for stream in self.all_income():
            total += stream.real_schedule(self.n_years, inflation)
        return total


@dataclass(frozen=True)
class SimulationConfig:
    """How to run the experiment (as opposed to what the household does).

    Attributes:
        n_paths: Number of simulated retirements.
        seed: Random seed, for reproducibility.
        model: ``"lognormal"``, ``"fat_tailed"`` or ``"bootstrap"``.
        antithetic: Use antithetic variates (mirrored pairs of paths).
        market: Capital market assumptions (default: calibrated to 1928-2025).
        block_size: Years per block for the bootstrap model.
    """

    n_paths: int = 10_000
    seed: int = 3044
    model: str = "lognormal"
    antithetic: bool = False
    market: MarketAssumptions = HISTORICAL_US
    block_size: int = 5


@dataclass(frozen=True)
class Estimate:
    """A Monte Carlo estimate of a probability with its sampling error."""

    p: float
    se: float
    ci_low: float
    ci_high: float
    se_plain: float  # what the SE would be under plain i.i.d. sampling

    def __str__(self) -> str:
        return f"{self.p:.1%} (95% CI {self.ci_low:.1%} to {self.ci_high:.1%}, SE {self.se:.2%})"


def estimate_mean(x: np.ndarray, antithetic: bool = False) -> Estimate:
    """Monte Carlo estimate of E[x] with SE and 95% CI.

    Plain Monte Carlo: paths are i.i.d., so SE = sd(x) / sqrt(N). For a 0/1
    indicator this is the familiar sqrt(p (1 - p) / N).

    Antithetic variates: the two paths in a pair are *not* independent, so the
    right unit is the pair. Average each pair and take the SE of those N/2 pair
    averages. If mirrored paths are negatively correlated, the pair averages
    vary less and the SE shrinks.

    Works for indicators and for paired differences between two plans.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    p = float(x.mean())
    se_plain = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if antithetic:
        pairs = x.reshape(-1, 2).mean(axis=1)
        se = float(pairs.std(ddof=1) / np.sqrt(pairs.size))
    else:
        se = se_plain
    return Estimate(p, se, p - 1.96 * se, p + 1.96 * se, se_plain)


@dataclass
class SimulationResult:
    """Outputs of a simulation run. Arrays are indexed ``[path, year]``.

    Attributes:
        balances: Portfolio value at the start of each year and at the end,
            shape ``(n_paths, n_years + 1)``. Column 0 is the starting value.
        consumption: What the household actually spent each year (income plus
            the withdrawal it could actually make), shape ``(n_paths, n_years)``.
        target: What it planned to spend each year (changes under guardrails).
        fail_year: First year (1-based) with a shortfall while alive; 0 if never.
        lifespan: Years lived in retirement (equals the horizon if no mortality).
        portfolio_returns: Net real portfolio return per path and year.
        income: Guaranteed income per year, shape ``(n_years,)``.
    """

    plan: RetirementPlan
    config: SimulationConfig
    balances: np.ndarray
    consumption: np.ndarray
    target: np.ndarray
    fail_year: np.ndarray
    lifespan: np.ndarray
    portfolio_returns: np.ndarray
    income: np.ndarray
    _years_lived: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Year t (0-based) is lived if the retiree is alive at its start: lifespan > t.
        n_years = self.balances.shape[1] - 1
        self._years_lived = np.minimum(np.ceil(self.lifespan), n_years).astype(int)

    # ---- basic outcomes -------------------------------------------------
    @property
    def n_years(self) -> int:
        return self.consumption.shape[1]

    @property
    def alive(self) -> np.ndarray:
        """Boolean ``(n_paths, n_years)``: was the retiree alive in that year?"""
        return np.arange(self.n_years)[None, :] < self.lifespan[:, None]

    @property
    def funded(self) -> np.ndarray:
        """True where every year's spending target was met while alive."""
        return self.fail_year == 0

    @property
    def lowest_spending(self) -> np.ndarray:
        """Lowest actual spending in any year lived, as a share of the original goal."""
        if self.plan.annual_spending == 0:
            return np.ones(self.lifespan.size)
        low = np.where(self.alive, self.consumption, np.inf).min(axis=1)
        return low / self.plan.annual_spending

    @property
    def success(self) -> np.ndarray:
        """Funded every year *and* spending never fell below the plan's floor."""
        return self.funded & (self.lowest_spending >= self.plan.spending_floor - 1e-12)

    @property
    def beyond_horizon(self) -> np.ndarray:
        """True where the retiree outlived the simulated horizon (mortality runs only)."""
        if self.plan.mortality is None:
            return np.zeros(self.lifespan.size, dtype=bool)
        return self.lifespan > self.n_years

    @property
    def years_lived(self) -> np.ndarray:
        return self._years_lived

    @property
    def terminal_wealth(self) -> np.ndarray:
        """Portfolio value at death (or at the horizon): the bequest."""
        rows = np.arange(self.balances.shape[0])
        return self.balances[rows, self._years_lived]

    @property
    def shortfall_years(self) -> np.ndarray:
        """Number of lived years in which spending fell short of that year's target."""
        short = self.consumption < self.target * (1 - 1e-9)
        return np.sum(short & self.alive, axis=1)

    # ---- estimation -----------------------------------------------------
    def success_probability(self) -> Estimate:
        """Estimate P(success) with a standard error and 95% confidence interval."""
        return estimate_mean(self.success, self.config.antithetic)

    def percentiles(self, qs=(10, 25, 50, 75, 90)) -> np.ndarray:
        """Balance percentiles by year, shape ``(len(qs), n_years + 1)``."""
        return np.percentile(self.balances, qs, axis=0)

    def summary(self) -> dict:
        """Headline numbers for reports."""
        est = self.success_probability()
        short = ~self.funded
        low = self.lowest_spending
        cut = np.any(self.alive & (self.target < self.plan.annual_spending * (1 - 1e-9)), axis=1)
        return {
            "success": est.p,
            "se": est.se,
            "ci_low": est.ci_low,
            "ci_high": est.ci_high,
            "funded": float(self.funded.mean()),
            "p_any_cut": float(cut.mean()),
            "p_below_floor": float((low < self.plan.spending_floor - 1e-12).mean()),
            "median_lowest_spending": float(np.median(low)),
            "median_terminal": float(np.median(self.terminal_wealth)),
            "p10_terminal": float(np.percentile(self.terminal_wealth, 10)),
            "median_fail_year": float(np.median(self.fail_year[short])) if short.any() else None,
            "avg_shortfall_years_if_short": float(self.shortfall_years[short].mean()) if short.any() else 0.0,
            "median_floor_if_short": float(np.median(low[short])) if short.any() else None,
            "share_beyond_horizon": float(self.beyond_horizon.mean()),
        }


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def run_paths(
    plan: RetirementPlan,
    portfolio_returns: np.ndarray,
    lifespans: np.ndarray,
    income: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the year-by-year cash-flow loop for every path at once.

    Args:
        plan: The retirement plan.
        portfolio_returns: Net real portfolio return, shape ``(n_paths, n_years)``.
        lifespans: Years lived in retirement per path, shape ``(n_paths,)``.
        income: Guaranteed real income per year, shape ``(n_years,)``.

    Returns:
        Dict with ``balances``, ``consumption``, ``target`` and ``fail_year``.
    """
    n_paths, n_years = portfolio_returns.shape
    b0 = plan.invested_portfolio
    s0 = plan.annual_spending

    balance = np.full(n_paths, b0, dtype=float)
    spending = np.full(n_paths, s0, dtype=float)
    fail_year = np.zeros(n_paths, dtype=int)

    balances = np.empty((n_paths, n_years + 1))
    balances[:, 0] = balance
    consumption = np.empty((n_paths, n_years))
    target = np.empty((n_paths, n_years))

    for t in range(n_years):
        if plan.rule == "guardrails" and t > 0 and b0 > 0:
            # Reference rate: the withdrawal rate the original plan implies for
            # *this* year's income mix. Comparing against it (instead of the
            # year-0 rate) keeps a Social Security start date from looking like
            # a windfall that triggers raises.
            reference = max(s0 - income[t], 0.0) / b0
            if reference > 0:
                need = np.maximum(spending - income[t], 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rate = np.where(balance > 0, need / balance, np.inf)
                cut = (rate > UPPER_GUARDRAIL * reference) & (n_years - t > NO_CUTS_FINAL_YEARS)
                rise = rate < LOWER_GUARDRAIL * reference
                spending = np.where(cut, spending * (1 - CUT), np.where(rise, spending * (1 + RAISE), spending))

        need = np.maximum(spending - income[t], 0.0)
        withdrawal = np.minimum(need, balance)          # cannot take more than is there
        alive = lifespans > t
        short = (balance < need) & alive & (fail_year == 0)
        fail_year[short] = t + 1

        consumption[:, t] = income[t] + withdrawal
        target[:, t] = spending
        balance = (balance - withdrawal) * (1.0 + portfolio_returns[:, t])
        balances[:, t + 1] = balance

    return {"balances": balances, "consumption": consumption, "target": target, "fail_year": fail_year}


def portfolio_returns_for(plan: RetirementPlan, stock: np.ndarray, bond: np.ndarray) -> np.ndarray:
    """Net portfolio return with annual rebalancing: w * r_s + (1 - w) * r_b - fee."""
    w = plan.stock_share
    return w * stock + (1.0 - w) * bond - plan.fee


def lifespan_uniforms(config: SimulationConfig) -> np.ndarray:
    """Uniform(0, 1) draws that drive lifespans, one per path.

    They come from their own stream, seeded from ``(seed, 1)``, so turning
    mortality on or off never changes the market paths. Sharing the *uniforms*
    (rather than the lifespans) across plans gives common random numbers even
    when plans use different mortality tables or retirement ages: each plan
    turns the same uniform into a lifespan through its own survival curve.
    """
    return np.random.default_rng([config.seed, 1]).random(config.n_paths)


def lifespans_for(plan: RetirementPlan, u: np.ndarray) -> np.ndarray:
    """Years lived in retirement: from the plan's mortality model, or the fixed horizon."""
    if plan.mortality is None:
        return np.full(u.size, float(plan.n_years))
    return plan.mortality.remaining_life_from_uniform(plan.retirement_age, u)


def simulate(
    plan: RetirementPlan,
    config: SimulationConfig = SimulationConfig(),
    returns: tuple[np.ndarray, np.ndarray] | None = None,
    uniforms: np.ndarray | None = None,
    lifespans: np.ndarray | None = None,
) -> SimulationResult:
    """Simulate a retirement plan.

    Args:
        plan: The household and its strategy.
        config: Number of paths, seed, return model and market assumptions.
        returns: Optional pre-generated ``(stock, bond)`` arrays with at least
            ``plan.n_years`` columns (extra columns are ignored). Passing the
            same arrays to several plans gives *common random numbers*: every
            plan faces identical markets, so differences between plans are not
            sampling noise.
        uniforms: Optional pre-generated lifespan uniforms (same idea for longevity).
        lifespans: Optional lifespans in years, overriding mortality entirely
            (useful for tests and what-if analysis).

    Returns:
        A :class:`SimulationResult`.
    """
    n_years = plan.n_years
    if returns is None:
        returns = generate_returns(
            config.market, config.n_paths, n_years, model=config.model,
            antithetic=config.antithetic, seed=config.seed, block_size=config.block_size,
        )
    stock, bond = returns
    if stock.shape[0] != config.n_paths or stock.shape[1] < n_years:
        raise ValueError("returns must have config.n_paths rows and at least plan.n_years columns")
    stock, bond = stock[:, :n_years], bond[:, :n_years]

    if lifespans is None:
        u = lifespan_uniforms(config) if uniforms is None else uniforms
        lifespans = lifespans_for(plan, u)

    port = portfolio_returns_for(plan, stock, bond)
    income = plan.income_schedule(config.market.inflation)
    out = run_paths(plan, port, lifespans, income)
    result = SimulationResult(
        plan=plan, config=config, lifespan=lifespans, portfolio_returns=port, income=income, **out
    )
    share = result.beyond_horizon.mean()
    if share > BEYOND_HORIZON_WARNING:
        warnings.warn(
            f"{share:.2%} of simulated retirees outlive max_age={plan.max_age}; their years "
            "beyond it are not simulated. Raise max_age.",
            stacklevel=2,
        )
    return result
