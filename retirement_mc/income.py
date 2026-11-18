"""Guaranteed income: Social Security, pensions and annuities.

Guaranteed income changes the retirement problem in two ways:

1. It lowers how much the portfolio has to supply each year. Spending is set as
   a total need; the portfolio only covers ``spending - income``.
2. It puts a floor under consumption. If the portfolio runs dry, the retiree
   still has the income, so a "failure" means a cut to the income floor rather
   than zero.

Timing convention: year ``t = 0`` is the first year of retirement. Income, like
withdrawals, arrives at the start of each year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mortality import GompertzMortality


@dataclass(frozen=True)
class IncomeStream:
    """A stream of annual income.

    Attributes:
        name: Label used in reports, e.g. "Social Security".
        amount: Annual amount. For an inflation-adjusted stream this is in
            today's dollars. For a nominal stream (``inflation_adjusted=False``)
            it is the fixed dollar amount written in the contract, and its real
            value shrinks every year with inflation.
        start_year: First year the income is paid (0 = first year of retirement).
        end_year: First year it is *no longer* paid. ``None`` means it lasts for
            life (or to the end of the simulation horizon).
        inflation_adjusted: True for COLA-linked income such as Social Security.
    """

    name: str
    amount: float
    start_year: int = 0
    end_year: int | None = None
    inflation_adjusted: bool = True

    def real_schedule(self, n_years: int, inflation: float) -> np.ndarray:
        """Income in today's dollars for each year 0..n_years-1, shape ``(n_years,)``."""
        t = np.arange(n_years)
        end = n_years if self.end_year is None else self.end_year
        active = (t >= self.start_year) & (t < end)
        if self.inflation_adjusted:
            real = np.full(n_years, float(self.amount))
        else:
            # A fixed nominal payment received after t years of inflation is
            # worth amount / (1 + inflation)^t in today's dollars.
            real = self.amount / (1.0 + inflation) ** t
        return np.where(active, real, 0.0)


# --------------------------------------------------------------------------
# Social Security
# --------------------------------------------------------------------------

def social_security_factor(claim_age: float, full_retirement_age: float = 67.0) -> float:
    """Benefit as a fraction of the full-retirement-age benefit (the PIA).

    Current-law rules for workers born in 1960 or later (FRA 67):

    * Claiming early: the benefit is reduced 5/9 of 1% per month for the first
      36 months before FRA and 5/12 of 1% for each additional month. Claiming at
      62 (60 months early) gives 100% - 20% - 10% = 70%.
    * Claiming late: delayed retirement credits add 8% per year (2/3 of 1% per
      month) after FRA, up to age 70. Claiming at 70 gives 124%.

    Source: Social Security Administration, "Benefits Planner: Retirement" and
    "Effect of Early or Delayed Retirement on Retirement Benefits".
    """
    if not 62.0 <= claim_age <= 70.0:
        raise ValueError("Social Security can be claimed between ages 62 and 70")
    months = round((claim_age - full_retirement_age) * 12)
    if months >= 0:
        return 1.0 + months * (2.0 / 3.0) / 100.0
    early = -months
    first = min(early, 36)
    rest = max(early - 36, 0)
    return 1.0 - (first * 5.0 / 9.0 + rest * 5.0 / 12.0) / 100.0


def social_security(
    benefit_at_fra: float,
    claim_age: float,
    retirement_age: float,
    full_retirement_age: float = 67.0,
) -> IncomeStream:
    """Build a Social Security income stream from the full-retirement-age benefit.

    Args:
        benefit_at_fra: Annual benefit at full retirement age, today's dollars.
        claim_age: Age at which benefits start (62 to 70).
        retirement_age: Age when the simulation starts (stops working).
    """
    factor = social_security_factor(claim_age, full_retirement_age)
    start = max(0, int(round(claim_age - retirement_age)))
    return IncomeStream(
        name=f"Social Security (claim at {claim_age:g})",
        amount=benefit_at_fra * factor,
        start_year=start,
        inflation_adjusted=True,  # benefits get annual cost-of-living adjustments
    )


# --------------------------------------------------------------------------
# Annuities
# --------------------------------------------------------------------------

def annuity_factor(
    age: float,
    mortality: GompertzMortality,
    discount_rate: float,
    start_year: int = 0,
    max_age: float = 120.0,
) -> float:
    """Expected present value of $1 per year for life, paid at the start of each year.

        a = sum over t >= start_year of  S(t | age) / (1 + r)^t

    Each year's $1 is weighted by the probability of being alive to receive it
    and discounted back to today. ``start_year > 0`` gives a deferred annuity
    (a DIA, or "longevity insurance" when deferred to ~85).
    """
    t = np.arange(start_year, int(max_age - age) + 1)
    return float(np.sum(mortality.survival(age, t) / (1.0 + discount_rate) ** t))


def price_payout_rate(
    age: float,
    mortality: GompertzMortality,
    discount_rate: float,
    start_year: int = 0,
    load: float = 0.05,
) -> float:
    """Annual payout per $1 of premium an insurer could offer.

    A fairly priced annuity satisfies ``premium = payment * annuity_factor``.
    Insurers keep a share of the premium for costs, capital and profit (the
    ``load``), so ``payment / premium = (1 - load) / annuity_factor``.

    Use a *nominal* discount rate for a fixed-dollar annuity and a *real* rate
    for an inflation-adjusted one; the real version pays less at the start
    because its payments grow with prices.
    """
    return (1.0 - load) / annuity_factor(age, mortality, discount_rate, start_year)


@dataclass(frozen=True)
class Annuity:
    """A lifetime income annuity bought with part of the portfolio at retirement.

    Attributes:
        premium: Dollars moved from the portfolio to the insurer at t = 0.
        payout_rate: Annual payment divided by premium (e.g. 0.07 = 7%).
        start_year: 0 for an immediate annuity (SPIA); later for a deferred one.
        inflation_adjusted: True if payments rise with inflation.
        name: Label used in reports.
    """

    premium: float
    payout_rate: float
    start_year: int = 0
    inflation_adjusted: bool = False
    name: str = "Annuity"

    @property
    def annual_payment(self) -> float:
        return self.premium * self.payout_rate

    def income_stream(self) -> IncomeStream:
        return IncomeStream(
            name=self.name,
            amount=self.annual_payment,
            start_year=self.start_year,
            end_year=None,  # lifetime income
            inflation_adjusted=self.inflation_adjusted,
        )
