"""Annual real return generation for a two-asset (stock/bond) portfolio.

Everything in this package is in *real* terms (today's dollars). Modeling real
returns directly means a "fixed" withdrawal is automatically inflation-adjusted,
which is the framing behind the classic 4% rule.

Two return models are supported:

* ``"lognormal"``: log-returns are bivariate normal. This keeps every return
  above -100% and produces the right-skew real markets show over long periods.
* ``"t"``: arithmetic returns follow a multivariate Student-t distribution,
  rescaled to the same mean and variance. Same volatility, fatter tails, so
  crashes like 2008 show up more often.

Correlation between stocks and bonds is imposed with a Cholesky factor of the
correlation matrix, the standard way to turn independent standard normals into
correlated ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketAssumptions:
    """Capital market assumptions. All values are annual and real (after inflation).

    Means and standard deviations are *arithmetic*, which is how most
    capital-market-assumption tables quote them.

    Attributes:
        stock_mean: Expected real stock return, e.g. 0.08 for 8%.
        stock_sd: Standard deviation of real stock returns.
        bond_mean: Expected real bond return.
        bond_sd: Standard deviation of real bond returns.
        correlation: Correlation between stock and bond returns, in (-1, 1).
        inflation: Constant inflation rate. Used only to convert *nominal*
            income (for example a fixed annuity) into today's dollars.
    """

    stock_mean: float = 0.080
    stock_sd: float = 0.19
    bond_mean: float = 0.025
    bond_sd: float = 0.08
    correlation: float = 0.10
    inflation: float = 0.025

    def __post_init__(self) -> None:
        if not -1.0 < self.correlation < 1.0:
            raise ValueError("correlation must be strictly between -1 and 1")
        if self.stock_sd < 0 or self.bond_sd < 0:
            raise ValueError("standard deviations must be non-negative")


# Illustrative assumption sets. These are not forecasts; swap in a published
# source (for example a firm's capital market assumptions) before relying on them.
LONG_RUN_US = MarketAssumptions()
MUTED_FORWARD = MarketAssumptions(
    stock_mean=0.055, stock_sd=0.17, bond_mean=0.015, bond_sd=0.06, correlation=0.20
)


def lognormal_params(mean: float, sd: float) -> tuple[float, float]:
    """Convert an arithmetic mean/sd of returns into lognormal parameters.

    If ``1 + R = exp(X)`` with ``X ~ Normal(mu, sigma^2)``, then

        E[1 + R]   = exp(mu + sigma^2 / 2)
        Var[1 + R] = (exp(sigma^2) - 1) * exp(2 mu + sigma^2)

    Solving those two equations for ``mu`` and ``sigma`` gives the formulas
    below. The result: simulated returns have exactly the arithmetic mean and
    standard deviation the user asked for.

    Returns:
        (mu, sigma) of the log-return.
    """
    sigma2 = np.log(1.0 + sd**2 / (1.0 + mean) ** 2)
    mu = np.log(1.0 + mean) - sigma2 / 2.0
    return float(mu), float(np.sqrt(sigma2))


def generate_returns(
    market: MarketAssumptions,
    n_paths: int,
    n_years: int,
    model: str = "lognormal",
    df: int = 5,
    antithetic: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate annual real stock and bond returns.

    Args:
        market: Capital market assumptions.
        n_paths: Number of simulated retirements (rows).
        n_years: Number of years per retirement (columns).
        model: ``"lognormal"`` or ``"t"``.
        df: Degrees of freedom for the t model. Must be > 2 so variance exists.
        antithetic: If True, paths come in mirrored pairs (z, -z): rows 0 and 1
            are a pair, rows 2 and 3 are a pair, and so on. Requires an even
            ``n_paths``.
        seed: Seed for NumPy's PCG64 generator. Same seed, same paths.

    Returns:
        ``(stock, bond)``: two arrays of shape ``(n_paths, n_years)`` holding
        simple annual returns (0.07 means +7%).
    """
    if model not in ("lognormal", "t"):
        raise ValueError("model must be 'lognormal' or 't'")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic sampling needs an even number of paths")
    if model == "t" and df <= 2:
        raise ValueError("df must be greater than 2")

    rng = np.random.default_rng(seed)
    n_draw = n_paths // 2 if antithetic else n_paths

    # Independent standard normals: axis 0 is the asset (stock, bond).
    z = rng.standard_normal((2, n_draw, n_years))

    # Correlate them. For two assets the Cholesky factor of [[1, r], [r, 1]] is
    # [[1, 0], [r, sqrt(1 - r^2)]], so eps_bond = r * z_stock + sqrt(1 - r^2) * z_2.
    # Using np.linalg.cholesky keeps the code valid if more assets are added.
    rho = market.correlation
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    eps = np.einsum("ij,jnt->int", chol, z)

    if model == "t":
        # Multivariate t: divide correlated normals by sqrt(chi2_df / df), one
        # chi-square draw shared by both assets in a given year. Multiplying by
        # sqrt((df - 2) / chi2) instead rescales to unit variance, because
        # Var(t_df) = df / (df - 2).
        chi2 = rng.chisquare(df, size=(n_draw, n_years))
        eps = eps * np.sqrt((df - 2) / chi2)

    if antithetic:
        # Interleave each draw with its mirror image so pairs sit in adjacent rows.
        mirrored = np.empty((2, n_paths, n_years))
        mirrored[:, 0::2, :] = eps
        mirrored[:, 1::2, :] = -eps
        eps = mirrored

    if model == "lognormal":
        mu_s, sig_s = lognormal_params(market.stock_mean, market.stock_sd)
        mu_b, sig_b = lognormal_params(market.bond_mean, market.bond_sd)
        stock = np.expm1(mu_s + sig_s * eps[0])
        bond = np.expm1(mu_b + sig_b * eps[1])
    else:
        # Arithmetic t returns can, in principle, fall below -100%. Floor them at
        # -95% so a portfolio cannot go negative in a single year.
        stock = np.maximum(market.stock_mean + market.stock_sd * eps[0], -0.95)
        bond = np.maximum(market.bond_mean + market.bond_sd * eps[1], -0.95)

    return stock, bond
