"""Historical US market data, used to calibrate the model and to bootstrap returns.

Data (``data/us_annual_returns.csv``, 1928-2025, see ``data/SOURCES.md``):

* ``sp500``: S&P 500 total return (with dividends), from Aswath Damodaran (NYU Stern)
* ``tbond_10y``: 10-year US Treasury bond total return, Damodaran
* ``tbill_3m``: 3-month Treasury bill return, Damodaran
* ``cpi_inflation``: CPI-U inflation, December to December, from BLS data

All are nominal calendar-year returns. Real returns are computed exactly as
``(1 + nominal) / (1 + inflation) - 1`` rather than the approximation
``nominal - inflation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).parent / "data" / "us_annual_returns.csv"


@dataclass(frozen=True)
class MarketHistory:
    """Annual US returns. Every array has one entry per year."""

    years: np.ndarray
    stock: np.ndarray          # nominal
    bond: np.ndarray           # nominal
    bill: np.ndarray           # nominal
    inflation: np.ndarray

    @property
    def real_stock(self) -> np.ndarray:
        return (1.0 + self.stock) / (1.0 + self.inflation) - 1.0

    @property
    def real_bond(self) -> np.ndarray:
        return (1.0 + self.bond) / (1.0 + self.inflation) - 1.0

    def window(self, start: int | None = None, end: int | None = None) -> "MarketHistory":
        """Restrict to calendar years ``start..end`` inclusive."""
        lo = self.years.min() if start is None else start
        hi = self.years.max() if end is None else end
        keep = (self.years >= lo) & (self.years <= hi)
        if not keep.any():
            raise ValueError(f"no data between {lo} and {hi}")
        return MarketHistory(self.years[keep], self.stock[keep], self.bond[keep],
                             self.bill[keep], self.inflation[keep])


@lru_cache(maxsize=1)
def _load() -> MarketHistory:
    d = np.genfromtxt(DATA_FILE, delimiter=",", names=True)
    return MarketHistory(
        years=d["year"].astype(int),
        stock=d["sp500"],
        bond=d["tbond_10y"],
        bill=d["tbill_3m"],
        inflation=d["cpi_inflation"],
    )


def load_us_history(start: int | None = None, end: int | None = None) -> MarketHistory:
    """Load the bundled 1928-2025 data, optionally restricted to a window of years."""
    return _load().window(start, end)


def geometric_mean(returns: np.ndarray) -> float:
    """Annualized (compound) return: exp(mean(log(1 + r))) - 1."""
    return float(np.expm1(np.log1p(returns).mean()))
