import numpy as np
import pandas as pd

from engine import outcomes

MIN_N = 20


def _outcome_series(
    data:           pd.DataFrame,
    h:              int,
    outcome:        str = 'up',
    outcome_params: dict | None = None,
) -> pd.Series:
    """
    Return 1.0/0.0/NaN: did the outcome event occur within h bars of t?

    NaN for bars whose outcome is not yet realized (the trailing h rows, plus any
    warmup the outcome itself requires). Delegates to the outcome registry; the
    default 'up' reproduces the original P(close[t+h] > close[t]).
    """
    return outcomes.compute(data, outcome, h, outcome_params)


# Retained under its original name: the directional special case.
def _price_up(close: pd.Series, h: int) -> pd.Series:
    shifted = close.shift(-h)
    return (shifted > close).where(shifted.notna())


def compute_base_rate(
    data:           pd.DataFrame,
    horizons:       list[int],
    horizon_unit:   str = 'd',
    outcome:        str = 'up',
    outcome_params: dict | None = None,
) -> pd.DataFrame:
    """
    Unconditional rate P(outcome within h bars) for each horizon h.

    Returns a DataFrame indexed by horizon label (e.g. '+7d'), columns ['n', 'win_rate'].
    win_rate is in percent and is the base rate of whatever outcome is configured —
    the column name is kept for backward compatibility with existing xlsx readers.
    Rows where n < MIN_N are NaN.
    """
    rows = []
    for h in horizons:
        ev    = _outcome_series(data, h, outcome, outcome_params)
        valid = ev.notna()
        n     = int(valid.sum())
        p     = float(ev[valid].mean() * 100) if n >= MIN_N else np.nan
        rows.append({'n': n, 'win_rate': p})
    return pd.DataFrame(rows, index=pd.Index([f'+{h}{horizon_unit}' for h in horizons], name='horizon'))


def _row(mask: pd.Series, events: dict, horizons: list[int]) -> list:
    """
    Compute outcome rates for one threshold slice across all horizons.

    Returns [n_last, p_h1, p_h2, ...] where n_last is the observation count
    for the longest horizon — the most conservative choice because longer horizons
    lose more tail rows (no known outcome yet), so n strictly decreases with h.
    Reporting n_last avoids overstating sample size for shorter horizons.
    """
    row     = []
    n_last  = 0
    for i, h in enumerate(horizons):
        ev    = events[h]
        valid = mask & ev.notna()
        n     = int(valid.sum())
        p     = float(ev[valid].mean() * 100) if n >= MIN_N else np.nan
        if i == len(horizons) - 1:
            n_last = n
        row.append(p)
    return [n_last] + row


def compute_matrix(
    feature:        pd.Series,
    thresholds:     np.ndarray,
    horizons:       list[int],
    data:           pd.DataFrame,
    horizon_unit:   str = 'd',
    outcome:        str = 'up',
    outcome_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute CDF-style conditional outcome rates across a range of thresholds.

    For each threshold t, computes:
      p_below: P(outcome | feature < t) − one row per threshold, columns [n, +h1, +h2, ...]
      p_above: P(outcome | feature > t) − same layout

    Rates are in percent (raw, not yet subtracted from base rate).
    The n column reports the longest-horizon count (most conservative); see _row.
    """
    feature = feature.reindex(data.index)
    events  = {h: _outcome_series(data, h, outcome, outcome_params) for h in horizons}
    col_labels = [f'+{h}{horizon_unit}' for h in horizons]

    rows_below, rows_above = [], []
    idx_below,  idx_above  = [], []

    for t in thresholds:
        rows_below.append(_row(feature < t, events, horizons))
        rows_above.append(_row(feature > t, events, horizons))
        idx_below.append(f'X < {t:.4g}')
        idx_above.append(f'X > {t:.4g}')

    cols    = ['n'] + col_labels
    p_below = pd.DataFrame(rows_below, index=idx_below, columns=cols)
    p_above = pd.DataFrame(rows_above, index=idx_above, columns=cols)
    p_below.index.name = p_above.index.name = 'condition'
    return p_below, p_above
