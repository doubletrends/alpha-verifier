import numpy as np
import pandas as pd

MIN_N    = 20
HORIZONS = list(range(1, 15))   # +1d … +14d


def _price_up(close: pd.Series, h: int) -> pd.Series:
    shifted = close.shift(-h)
    return (shifted > close).where(shifted.notna())


def compute_base_rate(data: pd.DataFrame, horizons: list[int] = HORIZONS, horizon_unit: str = 'd') -> pd.DataFrame:
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
    row     = []
    n_first = 0
    n_last  = 0
    for i, h in enumerate(horizons):
        fu    = future_ups[h]
        valid = mask & fu.notna()
        n     = int(valid.sum())
        p     = float(fu[valid].mean() * 100) if n >= MIN_N else np.nan
        if i == 0:
            n_first = n
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
