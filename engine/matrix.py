import numpy as np
import pandas as pd

MIN_N = 20


def _price_up(close: pd.Series, h: int) -> pd.Series:
    """Return True/False/NaN: did close rise h bars later? NaN for the last h rows (no outcome yet)."""
    shifted = close.shift(-h)
    return (shifted > close).where(shifted.notna())


def compute_base_rate(data: pd.DataFrame, horizons: list[int], horizon_unit: str = 'd') -> pd.DataFrame:
    """
    Unconditional win rate P(close[t+h] > close[t]) for each horizon h.

    Returns a DataFrame indexed by horizon label (e.g. '+7d'), columns ['n', 'win_rate'].
    win_rate is in percent. Rows where n < MIN_N are NaN.
    """
    close = data['close']
    rows  = []
    for h in horizons:
        fu    = _price_up(close, h)
        valid = fu.notna()
        n     = int(valid.sum())
        p     = float(fu[valid].mean() * 100) if n >= MIN_N else np.nan
        rows.append({'n': n, 'win_rate': p})
    return pd.DataFrame(rows, index=pd.Index([f'+{h}{horizon_unit}' for h in horizons], name='horizon'))


def _row(mask: pd.Series, future_ups: dict, horizons: list[int]) -> list:
    """
    Compute win rates for one threshold slice across all horizons.

    Returns [n_last, p_h1, p_h2, ...] where n_last is the observation count
    for the longest horizon — the most conservative choice because longer horizons
    lose more tail rows (no known outcome yet), so n strictly decreases with h.
    Reporting n_last avoids overstating sample size for shorter horizons.
    """
    row     = []
    n_last  = 0
    for i, h in enumerate(horizons):
        fu    = future_ups[h]
        valid = mask & fu.notna()
        n     = int(valid.sum())
        p     = float(fu[valid].mean() * 100) if n >= MIN_N else np.nan
        if i == len(horizons) - 1:
            n_last = n
        row.append(p)
    return [n_last] + row


def compute_matrix(
    feature:      pd.Series,
    thresholds:   np.ndarray,
    horizons:     list[int],
    data:         pd.DataFrame,
    horizon_unit: str = 'd',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute CDF-style conditional win rates across a range of thresholds.

    For each threshold t, computes:
      p_below: P(up | feature < t) − one row per threshold, columns [n, +h1, +h2, ...]
      p_above: P(up | feature > t) − same layout

    Win rates are in percent (raw, not yet subtracted from base rate).
    The n column reports the longest-horizon count (most conservative); see _row.
    """
    close      = data['close']
    feature    = feature.reindex(data.index)
    future_ups = {h: _price_up(close, h) for h in horizons}
    col_labels = [f'+{h}{horizon_unit}' for h in horizons]

    rows_below, rows_above = [], []
    idx_below,  idx_above  = [], []

    for t in thresholds:
        rows_below.append(_row(feature < t, future_ups, horizons))
        rows_above.append(_row(feature > t, future_ups, horizons))
        idx_below.append(f'X < {t:.4g}')
        idx_above.append(f'X > {t:.4g}')

    cols    = ['n'] + col_labels
    p_below = pd.DataFrame(rows_below, index=idx_below, columns=cols)
    p_above = pd.DataFrame(rows_above, index=idx_above, columns=cols)
    p_below.index.name = p_above.index.name = 'condition'
    return p_below, p_above
