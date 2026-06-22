"""Expanding-window log-price residual.

For each date t the trend is fitted using only data up to t, so the residual
reflects exactly what a trader would have computed on that date — no look-ahead.

Cache: research/residual/output/cache.csv
  First run: ~5–20 s (one curve_fit per day, warm-started from prior solution).
  Subsequent runs: instant (cache reload).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CACHE_PATH  = Path(__file__).parent / 'output' / 'cache.csv'
MIN_PERIODS = 365   # days of history required before the first fit


def _log_curve(x, a, b, c):
    return a * np.log(x + c) + b


def compute(log_price: pd.Series) -> pd.Series:
    """Fit expanding log trend; return leak-free residual series."""
    n      = len(log_price)
    X_all  = (log_price.index - log_price.index[0]).days.to_numpy(dtype=float)
    y_all  = log_price.to_numpy(dtype=float)
    trend  = np.full(n, np.nan)
    p0     = [1.0, 1.0, 1.0]   # warm-started from previous fit after first success

    total = n - MIN_PERIODS
    print(f'  expanding log trend: {total} fits ...')

    for i in range(MIN_PERIODS, n):
        x_fit = X_all[:i + 1]
        y_fit = y_all[:i + 1]
        mask  = ~np.isnan(y_fit)
        if mask.sum() < MIN_PERIODS:
            continue
        try:
            (a, b, c), _ = curve_fit(
                _log_curve, x_fit[mask], y_fit[mask],
                p0=p0, maxfev=2000,
            )
            p0       = [a, b, c]
            trend[i] = _log_curve(X_all[i], a, b, c)
        except RuntimeError:
            pass   # failed to converge — leave as NaN

        if (i - MIN_PERIODS) % (total // 10 or 1) == 0:
            pct = 100 * (i - MIN_PERIODS) // total
            print(f'    {pct:3d}% ...', flush=True)

    trend_s = pd.Series(trend, index=log_price.index, name='log_price_trend')
    return (log_price - trend_s).rename('log_price_residual')


def load(data: pd.DataFrame) -> pd.Series:
    """Return expanding-window residual, computing and caching on first call."""
    if CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH, index_col='Date', parse_dates=True)
        if cached.index[-1] >= data.index[-1]:
            return cached['log_price_residual'].reindex(data.index)
        print(f'  residual cache stale (ends {cached.index[-1].date()}), recomputing ...')

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_price = np.log(data['price_usd']) if 'price_usd' in data.columns \
                else data['log_price_usd']
    resid = compute(log_price)
    resid.to_frame().to_csv(CACHE_PATH)
    print(f'  cached → {CACHE_PATH}')
    return resid
