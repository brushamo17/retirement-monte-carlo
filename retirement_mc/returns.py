"""Annual real return generation for a two-asset (stock/bond) portfolio.

Everything in this package is in *real* terms (today's dollars). Modeling real
returns directly means a "fixed" withdrawal is automatically inflation-adjusted,
which is the framing behind the classic 4% rule.

Three return models are supported:

* ``"lognormal"``: log-returns are bivariate normal. Returns can never fall
  below -100%, and compounding produces the right skew real markets show.
* ``"fat_tailed"``: a two-regime scale mixture of normals in log space. Most
  years come from a calm regime; 10% come from a regime with twice the
  volatility, hitting stocks and bonds in the same year. Same mean and
  volatility as the lognormal model, heavier tails. A Student-t is itself an
  (infinite) scale mixture of normals; the two-regime version keeps E[exp(X)]
  finite, so means can be calibrated exactly and no artificial floor is needed.
* ``"bootstrap"``: historical block bootstrap. Each path is stitched together
  from randomly chosen runs of consecutive years of actual 1928-2025 US real
  returns, so it keeps the real distribution, the stock/bond relationship and
  some multi-year patterns (for example the 1970s) with no distribution assumed.

For the parametric models, the means, standard deviations *and correlation*
the user specifies are all matched for **arithmetic** returns, which is how
capital market assumptions are quoted. Correlation is imposed in log space
through a Cholesky factor, with the log-space correlation solved numerically
so that the resulting arithmetic correlation hits the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .history import MarketHistory, geometric_mean, load_us_history

# Regimes for the scale mixtures: (probabilities, volatility multipliers).
REGIMES = {
    "lognormal": ((1.0,), (1.0,)),
    "fat_tailed": ((0.90, 0.10), (1.0, 2.0)),
}
MODELS = ("lognormal", "fat_tailed", "bootstrap")


@dataclass(frozen=True)
class MarketAssumptions:
    """Capital market assumptions. All values are annual and real (after inflation).

    Means, standard deviations and the correlation describe *arithmetic*
    returns, the way capital-market-assumption tables quote them.

    The defaults are the 1928-2025 US historical values (see ``calibrate``):
    S&P 500 and 10-year Treasuries, deflated by CPI.

    Attributes:
        stock_mean: Expected real stock return, e.g. 0.086 for 8.6%.
        stock_sd: Standard deviation of real stock returns.
        bond_mean: Expected real bond return.
        bond_sd: Standard deviation of real bond returns.
        correlation: Correlation between real stock and bond returns.
        inflation: Constant inflation rate, used only to convert *nominal*
            income (for example a fixed annuity) into today's dollars.
        label: Short name used in reports.
    """

    stock_mean: float = 0.086
    stock_sd: float = 0.193
    bond_mean: float = 0.018
    bond_sd: float = 0.089
    correlation: float = 0.09
    inflation: float = 0.030
    label: str = "US history 1928-2025"

    def __post_init__(self) -> None:
        if not -1.0 < self.correlation < 1.0:
            raise ValueError("correlation must be strictly between -1 and 1")
        if self.stock_sd < 0 or self.bond_sd < 0:
            raise ValueError("standard deviations must be non-negative")
        if self.stock_mean <= -1 or self.bond_mean <= -1:
            raise ValueError("mean returns must be above -100%")

    @classmethod
    def calibrate(cls, history: MarketHistory | None = None, label: str | None = None) -> "MarketAssumptions":
        """Estimate assumptions from historical annual data.

        Arithmetic means, sample standard deviations (ddof=1) and the sample
        correlation of real returns; inflation is the compound (geometric) rate,
        since it is used to compound nominal payments over many years.
        """
        h = history or load_us_history()
        rs, rb = h.real_stock, h.real_bond
        return cls(
            stock_mean=float(rs.mean()),
            stock_sd=float(rs.std(ddof=1)),
            bond_mean=float(rb.mean()),
            bond_sd=float(rb.std(ddof=1)),
            correlation=float(np.corrcoef(rs, rb)[0, 1]),
            inflation=geometric_mean(h.inflation),
            label=label or f"US history {h.years.min()}-{h.years.max()}",
        )


# Full-sample historical calibration (equals the dataclass defaults up to rounding).
HISTORICAL_US = MarketAssumptions.calibrate()

# A lower-return, forward-looking stress scenario. These are illustrative
# round numbers, in the spirit of the lower expected returns many firms have
# published in recent years; they are not taken from any one firm's forecast.
MUTED_FORWARD = MarketAssumptions(
    stock_mean=0.055, stock_sd=0.17, bond_mean=0.015, bond_sd=0.06,
    correlation=0.20, inflation=0.025, label="Muted forward-looking (illustrative)",
)


# --------------------------------------------------------------------------
# Calibration of the parametric models
# --------------------------------------------------------------------------

def lognormal_params(mean: float, sd: float) -> tuple[float, float]:
    """Closed-form lognormal parameters for a target arithmetic mean and sd.

    If ``1 + R = exp(X)`` with ``X ~ Normal(mu, sigma^2)``, then

        E[1 + R]   = exp(mu + sigma^2 / 2)
        Var[1 + R] = (exp(sigma^2) - 1) * exp(2 mu + sigma^2)

    Solving those two equations for ``mu`` and ``sigma``:

        sigma^2 = ln(1 + sd^2 / (1 + mean)^2),   mu = ln(1 + mean) - sigma^2 / 2
    """
    sigma2 = np.log(1.0 + sd**2 / (1.0 + mean) ** 2)
    mu = np.log(1.0 + mean) - sigma2 / 2.0
    return float(mu), float(np.sqrt(sigma2))


def _bisect(f, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Root of an increasing function f on [lo, hi] by bisection."""
    flo = f(lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _mix_exp(weights, scales, a: float, sigma: float) -> float:
    """E[exp(a * sigma * S * Z)] for regime scale S and standard normal Z."""
    return float(sum(p * np.exp(0.5 * (a * s * sigma) ** 2) for p, s in zip(weights, scales)))


def calibrate_marginal(mean: float, sd: float, model: str = "lognormal") -> tuple[float, float]:
    """Log-space (mu, sigma) so that arithmetic returns have the target mean and sd.

    In regime i the log-return is ``mu + s_i * sigma * Z``. The ratio
    ``E[(1+R)^2] / E[1+R]^2 = 1 + sd^2 / (1 + mean)^2`` depends only on sigma
    and increases with it, so sigma is found by bisection; mu then follows from
    the mean. For the lognormal model this reproduces ``lognormal_params``.
    """
    weights, scales = REGIMES[model]
    if sd == 0:
        return float(np.log1p(mean)), 0.0
    target = 1.0 + sd**2 / (1.0 + mean) ** 2
    ratio = lambda sig: _mix_exp(weights, scales, 2.0, sig) / _mix_exp(weights, scales, 1.0, sig) ** 2 - target
    sigma = _bisect(ratio, 0.0, 3.0)
    mu = np.log1p(mean) - np.log(_mix_exp(weights, scales, 1.0, sigma))
    return float(mu), float(sigma)


def arithmetic_correlation(rho_log: float, sig_s: float, sig_b: float, model: str = "lognormal") -> float:
    """Correlation of arithmetic returns implied by log-space correlation ``rho_log``.

    With a shared regime, E[exp(X + Y)] averages exp(s_i^2 (sx^2 + sy^2 +
    2 rho sx sy) / 2) over regimes. For the pure lognormal this reduces to the
    textbook result (exp(rho sx sy) - 1) / sqrt((exp(sx^2) - 1)(exp(sy^2) - 1)).
    The means mu cancel, so they are not needed.
    """
    w, s = REGIMES[model]
    ex = _mix_exp(w, s, 1.0, sig_s)
    ey = _mix_exp(w, s, 1.0, sig_b)
    exx = _mix_exp(w, s, 2.0, sig_s)
    eyy = _mix_exp(w, s, 2.0, sig_b)
    exy = float(sum(p * np.exp(0.5 * k**2 * (sig_s**2 + sig_b**2 + 2 * rho_log * sig_s * sig_b))
                    for p, k in zip(w, s)))
    return (exy - ex * ey) / np.sqrt((exx - ex**2) * (eyy - ey**2))


def calibrate_correlation(target: float, sig_s: float, sig_b: float, model: str = "lognormal") -> float:
    """Log-space correlation that produces the target arithmetic correlation."""
    if sig_s == 0 or sig_b == 0:
        return 0.0
    lo, hi = -0.999999, 0.999999
    f = lambda r: arithmetic_correlation(r, sig_s, sig_b, model) - target
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError(
            f"correlation {target} cannot be reached with these volatilities "
            f"(range {f(lo) + target:.3f} to {f(hi) + target:.3f})"
        )
    return _bisect(f, lo, hi)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def bootstrap_returns(
    history: MarketHistory, n_paths: int, n_years: int, block_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Circular block bootstrap of historical real stock and bond returns.

    Each path is built from blocks of ``block_size`` consecutive years with
    random starting years. Indices wrap around the end of the sample (circular),
    so every year is equally likely to be drawn. Stocks and bonds are always
    taken from the same calendar year, which preserves their correlation.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    h = len(history.years)
    n_blocks = -(-n_years // block_size)  # ceiling division
    starts = rng.integers(0, h, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_size)) % h
    idx = idx.reshape(n_paths, -1)[:, :n_years]
    return history.real_stock[idx], history.real_bond[idx]


def generate_returns(
    market: MarketAssumptions,
    n_paths: int,
    n_years: int,
    model: str = "lognormal",
    antithetic: bool = False,
    seed: int | None = None,
    block_size: int = 5,
    history: MarketHistory | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate annual real stock and bond returns.

    Args:
        market: Capital market assumptions (ignored by the bootstrap, which
            uses the historical returns themselves).
        n_paths: Number of simulated retirements (rows).
        n_years: Number of years per retirement (columns).
        model: ``"lognormal"``, ``"fat_tailed"`` or ``"bootstrap"``.
        antithetic: Paths come in mirrored pairs (z, -z) in adjacent rows.
            Requires an even ``n_paths``; not available for the bootstrap.
        seed: Seed for NumPy's PCG64 generator. Same seed, same paths.
        block_size: Years per block for the bootstrap.
        history: Data to bootstrap from (default: all of 1928-2025).

    Returns:
        ``(stock, bond)``: arrays of shape ``(n_paths, n_years)`` of simple
        annual real returns (0.07 means +7%).
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic sampling needs an even number of paths")
    rng = np.random.default_rng(seed)

    if model == "bootstrap":
        if antithetic:
            raise ValueError("antithetic variates are not defined for the bootstrap")
        return bootstrap_returns(history or load_us_history(), n_paths, n_years, block_size, rng)

    n_draw = n_paths // 2 if antithetic else n_paths
    mu_s, sig_s = calibrate_marginal(market.stock_mean, market.stock_sd, model)
    mu_b, sig_b = calibrate_marginal(market.bond_mean, market.bond_sd, model)
    rho_log = calibrate_correlation(market.correlation, sig_s, sig_b, model)

    # Correlated standard normals: eps = L z, with L the Cholesky factor of the
    # log-space correlation matrix. For two assets L = [[1, 0], [r, sqrt(1 - r^2)]].
    z = rng.standard_normal((2, n_draw, n_years))
    chol = np.linalg.cholesky(np.array([[1.0, rho_log], [rho_log, 1.0]]))
    eps = np.einsum("ij,jnt->int", chol, z)

    # Regime volatility multiplier, shared by both assets in a given year.
    weights, scales = REGIMES[model]
    if len(weights) > 1:
        regime = rng.choice(len(weights), size=(n_draw, n_years), p=weights)
        scale = np.asarray(scales)[regime]
    else:
        scale = np.ones((n_draw, n_years))

    if antithetic:
        # Mirror the normals but keep the same regimes, then interleave pairs.
        mirrored = np.empty((2, n_paths, n_years))
        mirrored[:, 0::2, :], mirrored[:, 1::2, :] = eps, -eps
        eps = mirrored
        scale = np.repeat(scale, 2, axis=0)

    stock = np.expm1(mu_s + sig_s * scale * eps[0])
    bond = np.expm1(mu_b + sig_b * scale * eps[1])
    return stock, bond
