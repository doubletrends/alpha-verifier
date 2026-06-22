"""Stage 2 — feature engineering.

Reads the ODS and builds the feature warehouse (DWD):

  - price decomposition : log price, fitted log trend, residual
  - supply-detrended    : hash rate, tx count, active addresses
  - valuation           : MVRV ratio
  - technical           : Williams %R (short + long), NMD variants, returns,
                          MA ratios/spreads, MACD, RSI, drawdown
  - volatility          : 30-day realized vol
  - cycle               : years since halving
  - macro               : DXY return / regime features

    Input : config.ODS_CSV
    Output: config.DWD_CSV, config.TREND_PARAMS_JSON

Run:  python pipeline/step_2_feature_engineering.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from config import ODS_CSV, DWD_CSV, ensure_dirs


SUPPLY_DETREND = [
    ('hash_rate',   'log_hash_rate'),
    ('tx_cnt',      'log_tx_cnt'),
    ('adr_act_cnt', 'log_adr_act_cnt'),
]

WILLIAMS_R = [
    ('short_percent_r', 21,  7),
    ('long_percent_r',  112, 3),
]

NMD = [
    ('nmd_730', 730, 730),
    ('nmd_365', 365, 730),
    ('nmd_90',   90, 730),
]

REALIZED_VOL_WINDOW = 30
TRADING_DAYS        = 252
HALVING_DATES = pd.to_datetime([
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-20',
])


# ── helpers ───────────────────────────────────────────────────────────────────

def _log_curve(x, a, b, c):
    return a * np.log(x + c) + b


def fit_log_trend(series: pd.Series):
    """Fit Y = a·log(X+c)+b on all available data. Returns (trend, a, b, c)."""
    valid = series.dropna()
    X_fit = (valid.index - series.index[0]).days.to_numpy(dtype=float)
    Y_fit = valid.to_numpy(dtype=float)
    (a, b, c), _ = curve_fit(_log_curve, X_fit, Y_fit, p0=[1.0, 1.0, 1.0])
    X_full = (series.index - series.index[0]).days.to_numpy(dtype=float)
    return pd.Series(_log_curve(X_full, a, b, c), index=series.index), a, b, c


def fit_supply_trend(series: pd.Series, supply: pd.Series):
    """Fit log(feature) = a·log(supply) + b. Returns (trend, a, b)."""
    valid = series.notna() & supply.notna() & (supply > 0)
    X_fit = np.log(supply[valid].to_numpy(dtype=float))
    Y_fit = series[valid].to_numpy(dtype=float)
    a, b  = np.polyfit(X_fit, Y_fit, 1)
    X_full = np.where(supply > 0, np.log(supply.clip(lower=1e-10)), np.nan)
    return pd.Series(a * X_full + b, index=series.index), a, b


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    hh = high.rolling(length).max()
    ll  = low.rolling(length).min()
    return 100.0 * (close - hh) / (hh - ll)


def ema(values: pd.Series, span: int) -> pd.Series:
    alpha = 2.0 / (span + 1.0)
    out   = np.full(len(values), np.nan)
    prev  = np.nan
    for i, v in enumerate(values.to_numpy(dtype=float)):
        if np.isnan(v):
            continue
        prev   = v if np.isnan(prev) else alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=values.index)


def compute_nmd(price: pd.Series, lookback: int, normalize_by: int) -> pd.Series:
    p      = price.to_numpy(dtype=float)
    n      = len(p)
    result = np.zeros(n)
    for i in range(lookback + 1, n):
        result[i] = np.sum(p[i - lookback:i] > p[i])
    return pd.Series(result / normalize_by, index=price.index)


def years_since_halving(index: pd.DatetimeIndex) -> pd.Series:
    dates = pd.to_datetime(index)
    years = np.zeros(len(dates), dtype=float)
    for i, dt in enumerate(dates):
        eligible     = HALVING_DATES[HALVING_DATES <= dt]
        last_halving = eligible.max() if len(eligible) else HALVING_DATES.min()
        years[i]     = max(0, (dt - last_halving).days) / 365.25
    return pd.Series(years, index=index)


def compute_rsi(price: pd.Series, length: int = 14) -> pd.Series:
    delta = price.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    rs    = up.rolling(length).mean() / down.rolling(length).mean()
    return 100.0 - (100.0 / (1.0 + rs))


# ── build ─────────────────────────────────────────────────────────────────────

def build_dwd() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ODS_CSV, index_col='Date', parse_dates=True)
    dwh = pd.DataFrame(index=raw.index)

    log_price = np.log(raw['price_usd'])
    price_trend, *_ = fit_log_trend(log_price)

    dwh['price_usd']          = raw['price_usd']
    dwh['log_price_usd']      = log_price
    dwh['log_price_trend']    = price_trend
    dwh['log_price_residual'] = log_price - price_trend

    supply = raw['sply_cur']
    for raw_col, dst in SUPPLY_DETREND:
        log_s = np.log(raw[raw_col].replace(0, np.nan))
        trend, *_ = fit_supply_trend(log_s, supply)
        dwh[dst]               = log_s
        dwh[f'{dst}_trend']    = trend
        dwh[f'{dst}_residual'] = log_s - trend

    dwh['mvrv'] = raw['mvrv']

    for col, length, ema_span in WILLIAMS_R:
        raw_wr   = williams_r(raw['high'], raw['low'], raw['close'], length)
        dwh[col] = (ema(raw_wr, ema_span) + 100) / 100
    dwh['percent_r_spread'] = dwh['short_percent_r'] - dwh['long_percent_r']

    print('Computing NMD features...')
    for col, lookback, normalize_by in NMD:
        dwh[col] = compute_nmd(raw['price_usd'], lookback, normalize_by)

    log_returns            = np.log(raw['price_usd'] / raw['price_usd'].shift(1))
    dwh['realized_vol_30'] = log_returns.rolling(REALIZED_VOL_WINDOW).std() * np.sqrt(TRADING_DAYS) * 100

    price = raw['price_usd']
    dwh['ret_14']           = price.pct_change(14)
    dwh['ret_30']           = price.pct_change(30)
    dwh['ret_spread_30_14'] = dwh['ret_30'] - dwh['ret_14']
    dwh['ma_ratio_30']      = price / price.rolling(30).mean() - 1.0
    dwh['ma_ratio_90']      = price / price.rolling(90).mean() - 1.0
    dwh['ma_spread_30_90']  = dwh['ma_ratio_30'] - dwh['ma_ratio_90']

    ema12        = price.ewm(span=12, adjust=False).mean()
    ema26        = price.ewm(span=26, adjust=False).mean()
    dwh['macd']  = (ema12 - ema26) / price
    dwh['rsi_14'] = compute_rsi(price, length=14)
    dwh['dd_90']  = price / price.rolling(90).max() - 1.0

    dwh['cpi_yoy']            = (raw['cpi'] / raw['cpi'].shift(TRADING_DAYS) - 1.0) * 100.0
    dwh['years_since_halving'] = years_since_halving(raw.index)
    dwh['dxy'] = raw['dxy']
    dwh['dxy_ret_5'] = raw['dxy'].pct_change(5)
    dwh['dxy_ret_10'] = raw['dxy'].pct_change(10)
    dwh['dxy_ret_20'] = raw['dxy'].pct_change(20)
    dwh['dxy_ret_30'] = raw['dxy'].pct_change(30)
    dwh['dxy_ret_50'] = raw['dxy'].pct_change(50)
    dwh['dxy_ret_100'] = raw['dxy'].pct_change(100)
    dwh['dxy_ma_ratio_20'] = raw['dxy'] / raw['dxy'].rolling(20).mean() - 1.0

    return dwh, raw


if __name__ == '__main__':
    ensure_dirs()
    dwh, raw = build_dwd()
    dwh.to_csv(DWD_CSV)
    print(f'\nDWD saved → {DWD_CSV}  ({dwh.shape[0]} rows × {dwh.shape[1]} cols)')
    print(f'Date range : {dwh.index[0].date()} → {dwh.index[-1].date()}')
    print(f'Columns    : {list(dwh.columns)}')
