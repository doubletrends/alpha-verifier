"""Halving cycle-transition matrix.

Rows represent current cycle-age buckets. Columns represent the future
cycle-age bucket reached after a fixed forward horizon. Each cell contains:

    P(price_future > price_now | start bucket -> end bucket)

This makes the halving-cycle geometry explicit instead of mixing a cycle-age
threshold axis with a generic horizon axis.

Run:  python research/halving/cycle_transition.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from research.framework import load_data, write_xlsx, MIN_N

OUT_DIR = Path(__file__).parent / 'output'
BUCKET_EDGES = np.arange(0.0, 4.25, 0.25)   # 0.0, 0.25, ... 4.0


def bucket_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f'{lo:.1f}y→{hi:.1f}y')
    return labels


def build_transition_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_age = data['years_since_halving']
    price = data['price_usd'] if 'price_usd' in data.columns else np.exp(data['log_price_usd'])

    labels = bucket_labels(BUCKET_EDGES)
    prob = pd.DataFrame(index=labels, columns=['sample size (n)'] + labels, dtype=float)

    for start_label, start_lo, start_hi in zip(labels, BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        start_mask = (cycle_age >= start_lo) & (cycle_age < start_hi)
        n_total = int(start_mask.sum())
        prob.loc[start_label, 'sample size (n)'] = n_total

        for end_label, end_lo, end_hi in zip(labels, BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
            # Approximate the horizon needed to land in the target bucket by the midpoint gap.
            start_mid = (start_lo + start_hi) / 2.0
            end_mid = (end_lo + end_hi) / 2.0
            horizon_days = int(round((end_mid - start_mid) * 365.25))
            if horizon_days <= 0:
                prob.loc[start_label, end_label] = np.nan
                continue

            future_age = cycle_age + horizon_days / 365.25
            future_up = price.shift(-horizon_days) > price
            end_mask = (future_age >= end_lo) & (future_age < end_hi)
            valid = start_mask & end_mask & future_up.notna()
            n = int(valid.sum())
            prob.loc[start_label, end_label] = float(future_up[valid].mean() * 100) if n >= MIN_N else np.nan

    prob.index.name = 'start cycle age'
    bearish = prob.copy()
    prob_cols = labels
    bearish[prob_cols] = 100 - bearish[prob_cols]
    return prob, bearish


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    bullish, bearish = build_transition_matrix(data)
    write_xlsx(
        bullish,
        bearish,
        OUT_DIR / 'cycle_transition.xlsx',
        sheet_names=('bullish', 'bearish'),
        descriptions=(
            'P(price at target cycle-age bucket > price now | cycle-age transition)',
            'P(price at target cycle-age bucket < price now | cycle-age transition)',
        ),
    )
