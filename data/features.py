import numpy as np
import pandas as pd

HALVING_DATES = pd.to_datetime([
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-20',
])

_NEXT_HALVING_EST = pd.Timestamp('2028-04-20')
_ALL_HALVINGS = list(HALVING_DATES) + [_NEXT_HALVING_EST]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    rs    = up.rolling(period).mean() / down.rolling(period).mean()
    return 100.0 - (100.0 / (1.0 + rs))


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period).max()
    ll  = low.rolling(period).min()
    return ((close - hh) / (hh - ll) + 1.0)   # normalized to [0, 1]; 1=oversold, 0=overbought


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int) -> pd.Series:
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    return 100.0 * (close - ll) / (hh - ll)


def _macd(close: pd.Series, fast: int, slow: int) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow) / close   # normalized by price


def _macd_histogram(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    line = _macd(close, fast, slow)
    return line - line.ewm(span=signal, adjust=False).mean()


def _ma_ratio(close: pd.Series, period: int) -> pd.Series:
    return close / close.rolling(period).mean() - 1.0


def _ma_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    return close.rolling(fast).mean() / close.rolling(slow).mean() - 1.0


def _realized_vol(close: pd.Series, period: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252) * 100


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean() / close   # normalized by price


def _bb_pct(close: pd.Series, period: int, std_dev: float = 2.0) -> pd.Series:
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - (ma - std_dev * std)) / (2 * std_dev * std)


def _bb_width(close: pd.Series, period: int, std_dev: float = 2.0) -> pd.Series:
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std_dev * std) / ma


def _volume_ratio(volume: pd.Series, period: int) -> pd.Series:
    return volume / volume.rolling(period).mean()


def _vol_ratio(close: pd.Series, fast: int, slow: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    fast_vol = log_ret.rolling(fast).std()
    slow_vol = log_ret.rolling(slow).std()
    return fast_vol / slow_vol.replace(0, np.nan)


def _drawdown(close: pd.Series, period: int) -> pd.Series:
    return close / close.rolling(period).max() - 1.0


def _drawdown_recovery(close: pd.Series, short: int, long: int) -> pd.Series:
    """Difference between short- and long-period drawdowns. Positive = bouncing from deeper hole."""
    return _drawdown(close, short) - _drawdown(close, long)


def _roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period)


def _roc_spread(close: pd.Series, fast: int, slow: int) -> pd.Series:
    return close.pct_change(fast) - close.pct_change(slow)


def _dxy_ret(dxy: pd.Series, period: int) -> pd.Series:
    return dxy.pct_change(period)


def _dxy_ma_ratio(dxy: pd.Series, period: int) -> pd.Series:
    dxy = dxy.ffill()  # fill weekend gaps so rolling window sees consecutive values
    return dxy / dxy.rolling(period).mean() - 1.0


def _mvrv(data: pd.DataFrame) -> pd.Series:
    return data['mvrv']


def _hash_rate_ma_ratio(data: pd.DataFrame, period: int) -> pd.Series:
    hr = np.log(data['hash_rate'].replace(0, np.nan))
    return hr / hr.rolling(period).mean() - 1.0


def _adr_act_ma_ratio(data: pd.DataFrame, period: int) -> pd.Series:
    aa = np.log(data['adr_act_cnt'].replace(0, np.nan))
    return aa / aa.rolling(period).mean() - 1.0


def _cycle_phase(data: pd.DataFrame) -> pd.Series:
    """Fractional position in the current halving cycle: 0 = just after halving, 1 = just before next."""
    dates = pd.to_datetime(data.index)
    phases = []
    for dt in dates:
        past   = [h for h in _ALL_HALVINGS if h <= dt]
        future = [h for h in _ALL_HALVINGS if h > dt]
        last   = max(past) if past else _ALL_HALVINGS[0]
        nxt    = min(future) if future else _NEXT_HALVING_EST
        phases.append((dt - last).days / max((nxt - last).days, 1))
    return pd.Series(phases, index=data.index, dtype=float)


def _days_to_halving(data: pd.DataFrame) -> pd.Series:
    """Days remaining until the next halving event."""
    dates = pd.to_datetime(data.index)
    vals = []
    for dt in dates:
        future = [h for h in _ALL_HALVINGS if h > dt]
        nxt    = min(future) if future else _NEXT_HALVING_EST
        vals.append(max(0, (nxt - dt).days))
    return pd.Series(vals, index=data.index, dtype=float)


def _days_since_halving(data: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(data.index)
    days  = []
    for dt in dates:
        eligible = HALVING_DATES[HALVING_DATES <= dt]
        last     = eligible.max() if len(eligible) else HALVING_DATES.min()
        days.append(max(0, (dt - last).days))
    return pd.Series(days, index=data.index, dtype=float)


def compute(data: pd.DataFrame, feature: str, params: dict) -> pd.Series:
    c = data.get('close')
    h = data.get('high')
    l = data.get('low')
    v = data.get('volume')
    p = params

    dispatch = {
        'rsi':                lambda: _rsi(c, p['period']),
        'williams_r':         lambda: _williams_r(h, l, c, p['period']),
        'stoch_k':            lambda: _stoch_k(h, l, c, p['k_period']),
        'macd':               lambda: _macd(c, p['fast'], p['slow']),
        'macd_histogram':     lambda: _macd_histogram(c, p['fast'], p['slow'], p.get('signal', 9)),
        'ma_ratio':           lambda: _ma_ratio(c, p['period']),
        'ma_cross':           lambda: _ma_cross(c, p['fast'], p['slow']),
        'realized_vol':       lambda: _realized_vol(c, p['period']),
        'vol_ratio':          lambda: _vol_ratio(c, p['fast'], p['slow']),
        'atr':                lambda: _atr(h, l, c, p['period']),
        'bb_pct':             lambda: _bb_pct(c, p['period'], p.get('std_dev', 2.0)),
        'bb_width':           lambda: _bb_width(c, p['period'], p.get('std_dev', 2.0)),
        'volume_ratio':       lambda: _volume_ratio(v, p['period']),
        'drawdown':           lambda: _drawdown(c, p['period']),
        'drawdown_recovery':  lambda: _drawdown_recovery(c, p['short'], p['long']),
        'roc':                lambda: _roc(c, p['period']),
        'roc_spread':         lambda: _roc_spread(c, p['fast'], p['slow']),
        'dxy_ret':            lambda: _dxy_ret(data['dxy'], p['period']),
        'dxy_ma_ratio':       lambda: _dxy_ma_ratio(data['dxy'], p['period']),
        'mvrv':               lambda: _mvrv(data),
        'hash_rate_ma_ratio': lambda: _hash_rate_ma_ratio(data, p['period']),
        'adr_act_ma_ratio':   lambda: _adr_act_ma_ratio(data, p['period']),
        'days_since_halving': lambda: _days_since_halving(data),
        'cycle_phase':        lambda: _cycle_phase(data),
        'days_to_halving':    lambda: _days_to_halving(data),
    }

    if feature not in dispatch:
        raise ValueError(f"Unknown feature: '{feature}'. Available: {sorted(dispatch)}")
    return dispatch[feature]()
