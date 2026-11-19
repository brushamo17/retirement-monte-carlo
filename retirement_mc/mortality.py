"""Longevity model: the Gompertz law of mortality.

A fixed 30-year horizon answers "does the money last 30 years?" but a retiree
really wants to know "does the money last as long as I do?". Modeling lifespan
as random also lets us price lifetime annuities consistently with the same
mortality assumptions.

The Gompertz law says the force of mortality grows exponentially with age:

    lambda(age) = (1 / b) * exp((age - m) / b)

where ``m`` is the modal age at death and ``b`` is a dispersion parameter
(roughly, how spread out deaths are around ``m``). Integrating gives a closed
form for the probability that someone aged ``x`` survives ``t`` more years:

    S(t | x) = exp( exp((x - m) / b) * (1 - exp(t / b)) )

Because S has a closed-form inverse, remaining lifetimes can be sampled exactly
by inverse-transform sampling: draw U ~ Uniform(0, 1), solve S(t) = U for t.

The default parameters (m = 88, b = 10) are illustrative and give a 65-year-old
roughly a 50% chance of reaching 85, which is in the range of published
population tables. Replace them with fitted values for real planning work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GompertzMortality:
    """Gompertz mortality with modal age ``modal_age`` and dispersion ``dispersion``."""

    modal_age: float = 88.0
    dispersion: float = 10.0

    def survival(self, age: float, t: np.ndarray | float) -> np.ndarray:
        """Probability that a person aged ``age`` is still alive ``t`` years from now."""
        m, b = self.modal_age, self.dispersion
        t = np.asarray(t, dtype=float)
        return np.exp(np.exp((age - m) / b) * (1.0 - np.exp(t / b)))

    def remaining_life_from_uniform(self, age: float, u: np.ndarray) -> np.ndarray:
        """Turn Uniform(0, 1) draws into remaining lifetimes (inverse transform).

        Solving ``U = S(t)`` for ``t``:

            ln U         = exp((x - m) / b) * (1 - exp(t / b))
            exp(t / b)   = 1 - ln U * exp(-(x - m) / b)
            t            = b * ln(1 - ln U * exp(-(x - m) / b))

        Since ln U < 0, the argument of the outer log is above 1 and t > 0.
        """
        m, b = self.modal_age, self.dispersion
        u = np.clip(np.asarray(u, dtype=float), 1e-12, 1.0)  # guard against log(0)
        return b * np.log1p(-np.log(u) * np.exp(-(age - m) / b))

    def sample_remaining_life(
        self, age: float, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw ``size`` remaining lifetimes (in years) by inverse-transform sampling."""
        return self.remaining_life_from_uniform(age, rng.random(size))

    def life_expectancy(self, age: float, max_years: float = 60.0) -> float:
        """Expected remaining lifetime, E[T] = integral of S(t) dt, by the trapezoid rule."""
        t = np.linspace(0.0, max_years, 6001)
        s = self.survival(age, t)
        return float(np.sum((s[1:] + s[:-1]) / 2.0 * np.diff(t)))
