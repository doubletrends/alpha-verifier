"""Plain Williams %R(7) win-rate matrix on short horizons.

Run:  python research/williams_r/plain_7.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from research.framework import load_data, compute_matrix, write_xlsx

OUT_DIR = Path(__file__).parent / 'output'
HORIZONS = list(range(1, 31))   # +1d, +2d, ... +30d
N_THRESHOLDS = 30


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return 100.0 * (close - hh) / (hh - ll)


OUT_DIR.mkdir(exist_ok=True)
data = load_data()
price = pd.read_csv('pipeline/output/step1_ods.csv', index_col='Date', parse_dates=True)
wr7 = (williams_r(price['high'], price['low'], price['close'], 7) + 100) / 100

feat = wr7.dropna()
lo = np.nanpercentile(feat, 2)
hi = np.nanpercentile(feat, 98)
thresholds = np.linspace(lo, hi, N_THRESHOLDS)

print(f'plain_wr_7  range [{lo:.4g}, {hi:.4g}]')

p_above, p_below = compute_matrix(wr7.reindex(data.index), thresholds, HORIZONS, data)
write_xlsx(p_above, p_below, OUT_DIR / 'plain_7.xlsx')
