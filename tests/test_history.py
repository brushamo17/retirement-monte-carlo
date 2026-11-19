"""Tests for the historical data, the calibration and the bootstrap."""

import numpy as np
import pytest

from retirement_mc import (
    HISTORICAL_US,
    MarketAssumptions,
    RetirementPlan,
    generate_returns,
    historical_backtest,
    load_us_history,
)
from retirement_mc.history import geometric_mean


def test_data_covers_1928_to_2025_without_gaps():
    h = load_us_history()
    assert h.years[0] == 1928 and h.years[-1] == 2025
    assert np.all(np.diff(h.years) == 1)
    assert not np.isnan(np.c_[h.stock, h.bond, h.bill, h.inflation]).any()


def test_data_matches_published_long_run_returns():
    """Compound nominal returns for 1928-2024 are widely quoted as 9.94% (S&P) and 4.50% (bonds)."""
    h = load_us_history(1928, 2024)
    assert geometric_mean(h.stock) == pytest.approx(0.0994, abs=0.0001)
    assert geometric_mean(h.bond) == pytest.approx(0.0450, abs=0.0001)


def test_known_years():
    h = load_us_history()
    year = dict(zip(h.years, zip(h.stock, h.bond, h.inflation)))
    assert year[2008][0] == pytest.approx(-0.3655)
    assert year[2022][1] == pytest.approx(-0.1783)
    assert year[1946][2] == pytest.approx(0.181)


def test_defaults_are_the_historical_calibration():
    default, cal = MarketAssumptions(), HISTORICAL_US
    for field in ("stock_mean", "stock_sd", "bond_mean", "bond_sd", "correlation", "inflation"):
        assert getattr(default, field) == pytest.approx(getattr(cal, field), abs=0.0011)


def test_real_returns_use_exact_deflation():
    h = load_us_history(1974, 1974)
    assert h.real_stock[0] == pytest.approx((1 + h.stock[0]) / (1 + h.inflation[0]) - 1)


def test_bootstrap_draws_whole_blocks_of_real_history():
    h = load_us_history()
    stock, bond = generate_returns(HISTORICAL_US, 200, 30, model="bootstrap", block_size=5, seed=9)
    year_of = {round(v, 12): i for i, v in enumerate(h.real_stock)}
    for row_s, row_b in zip(stock[:20], bond[:20]):
        idx = [year_of[round(v, 12)] for v in row_s]
        assert np.allclose(row_b, h.real_bond[idx])               # stocks and bonds from the same year
        for k in range(0, 30, 5):                                  # consecutive years inside each block
            block = np.array(idx[k:k + 5])
            assert np.all(np.diff(block) % len(h.years) == 1)


def test_bootstrap_rejects_antithetic():
    with pytest.raises(ValueError):
        generate_returns(HISTORICAL_US, 10, 5, model="bootstrap", antithetic=True)


def test_backtest_uses_every_complete_window():
    res = historical_backtest(RetirementPlan(horizon_years=30))
    assert res["start_years"][0] == 1928 and res["start_years"][-1] == 1996
    assert 0.0 <= res["success_rate"] <= 1.0
