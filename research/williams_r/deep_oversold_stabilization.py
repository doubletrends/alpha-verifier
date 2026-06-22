"""Williams %R deep oversold stabilization study.

Focus condition:
  short=deep_os | long=deep_os | short>=long

This script keeps the base pair-state and tests a small set of refinements to
separate absolute level, ordering strength, and day-over-day stabilization.

Run:  python research/williams_r/deep_oversold_stabilization.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from research.framework import load_data, compute_condition_matrix, write_xlsx

OUT_DIR = Path(__file__).parent / 'output'
HORIZONS = list(range(1, 31))   # +1d, +2d, ... +30d


def build_conditions(data: pd.DataFrame) -> dict[str, pd.Series]:
    short_wr = data['short_percent_r']
    long_wr = data['long_percent_r']
    spread = short_wr - long_wr
    short_up = short_wr > short_wr.shift(1)
    long_up = long_wr > long_wr.shift(1)

    base = (short_wr < 0.20) & (long_wr < 0.20) & (short_wr >= long_wr)

    conditions = {
        'base': base,
        'spread>=0.02': base & (spread >= 0.02),
        'spread>=0.05': base & (spread >= 0.05),
        'short_rising': base & short_up,
        'both_rising': base & short_up & long_up,
        'fresh_entry': base & ~base.shift(1, fill_value=False),
    }
    return conditions


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    conditions = build_conditions(data)
    bullish = compute_condition_matrix(conditions, HORIZONS, data)

    bearish = bullish.copy()
    prob_cols = [c for c in bearish.columns if c != 'sample size (n)']
    bearish[prob_cols] = 100 - bearish[prob_cols]

    write_xlsx(
        bullish,
        bearish,
        OUT_DIR / 'deep_oversold_stabilization.xlsx',
        sheet_names=('bullish', 'bearish'),
        descriptions=(
            'P(price at +t days > price now | deep oversold stabilization condition)',
            'P(price at +t days < price now | deep oversold stabilization condition)',
        ),
    )
