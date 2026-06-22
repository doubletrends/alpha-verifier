"""Williams %R pair-state matrix on short horizons.

Organizes conditions by ordering first (`short>=long` vs `short<long`), then by
the absolute bucket pair. For cross-bucket states, ordering is implied by the
bucket pair and is not duplicated in the condition logic.

Run:  python research/williams_r/pair_state.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from research.framework import load_data, compute_condition_matrix, write_xlsx

OUT_DIR = Path(__file__).parent / 'output'
HORIZONS = list(range(1, 31))   # +1d, +2d, ... +30d

BUCKETS = [
    ('deep_os', 0.00, 0.20),
    ('os',      0.20, 0.40),
    ('neutral', 0.40, 0.60),
    ('ob',      0.60, 0.80),
    ('ext_ob',  0.80, 1.01),
]


def in_bucket(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return (series >= lo) & (series < hi)


def build_conditions(data: pd.DataFrame) -> dict[str, pd.Series]:
    short_wr = data['short_percent_r']
    long_wr = data['long_percent_r']

    conditions: dict[str, pd.Series] = {}
    for ordering in ('short>=long', 'short<long'):
        for short_idx, (short_name, short_lo, short_hi) in enumerate(BUCKETS):
            short_mask = in_bucket(short_wr, short_lo, short_hi)
            for long_idx, (long_name, long_lo, long_hi) in enumerate(BUCKETS):
                long_mask = in_bucket(long_wr, long_lo, long_hi)
                pair_mask = short_mask & long_mask

                if short_idx == long_idx:
                    if ordering == 'short>=long':
                        mask = pair_mask & (short_wr >= long_wr)
                    else:
                        mask = pair_mask & (short_wr < long_wr)
                    name = f'{ordering} | short={short_name} | long={long_name}'
                elif short_idx > long_idx and ordering == 'short>=long':
                    mask = pair_mask
                    name = f'{ordering} | short={short_name} | long={long_name}'
                elif short_idx < long_idx and ordering == 'short<long':
                    mask = pair_mask
                    name = f'{ordering} | short={short_name} | long={long_name}'
                else:
                    continue

                conditions[name] = mask

    return conditions


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    conditions = build_conditions(data)

    print(f'pair-state conditions: {len(conditions)}')
    matrix = compute_condition_matrix(conditions, HORIZONS, data)

    bearish = matrix.copy()
    prob_cols = [c for c in bearish.columns if c != 'sample size (n)']
    bearish[prob_cols] = 100 - bearish[prob_cols]

    write_xlsx(
        matrix,
        bearish,
        OUT_DIR / 'pair_state.xlsx',
        sheet_names=('bullish', 'bearish'),
        descriptions=(
            'P(price at +t days > price now | Williams %R pair-state condition)',
            'P(price at +t days < price now | Williams %R pair-state condition)',
        ),
    )
