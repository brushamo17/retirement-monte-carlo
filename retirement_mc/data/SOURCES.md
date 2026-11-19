# Data sources

`us_annual_returns.csv`: annual US returns, 1928–2025, as decimals (0.0518 = 5.18%).

| Column | Series | Source |
|---|---|---|
| `sp500` | S&P 500 total return, including dividends | Aswath Damodaran, NYU Stern, "Historical Returns on Stocks, Bonds and Bills" (updated January 5, 2026): pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html |
| `tbond_10y` | 10-year US Treasury bond total return | Damodaran, same table |
| `tbill_3m` | 3-month Treasury bill return | Damodaran, same table |
| `cpi_inflation` | CPI-U inflation, December over December | US Bureau of Labor Statistics, as tabulated at chartrow.com/inflation/by-year (December year-over-year column) |

## Checks performed

- **Transcription:** every annual return compounds to Damodaran's published cumulative "value of $100" column. The only gaps are rounding (largest: 1933 S&P, 0.02 points) and one inconsistency in the source itself: the 2025 bond return is printed as 7.80% while the cumulative column implies 8.29%. The printed 7.80% is used. It moves the long-run bond mean by less than 0.01 points.
- **Spot checks of CPI:** Dec/Dec inflation of 18.1% (1946), 12.5% (1980), 0.1% (2008) and 6.5% (2022) match BLS.
- **Cross-check of results:** nominal compound returns for 1928–2024 come out at 9.94% (stocks) and 4.50% (bonds), matching the widely cited figures from this dataset. Real compound returns for 1928–2020 come out at 6.6% (stocks) and 1.9% (bonds), versus 6.8% and 1.9% published using Damodaran's own inflation series. The small gap in stocks comes from using Dec/Dec CPI.

`tests/test_history.py` re-runs the compound-return check, so a corrupted file fails the test suite.
