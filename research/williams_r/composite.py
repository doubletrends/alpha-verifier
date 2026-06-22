"""Williams %R oversold composite win-rate matrix.

Feature: (1 - short_percent_r) * (1 - long_percent_r)
  High value: both 21-period and 112-period WR simultaneously deep in oversold.
  Low value:  neither is oversold.

Signal lives in the ABOVE sheet (x > k): high composite = both oversold = bullish.

Run:  python research/williams_r/composite.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from research.framework import load_data, compute_matrix, write_xlsx

OUT_DIR      = Path(__file__).parent / 'output'
HORIZONS     = list(range(5, 91, 5))   # +5d, +10d, … +90d  (18 cols)
N_THRESHOLDS = 30

OUT_DIR.mkdir(exist_ok=True)
data      = load_data()
short_wr  = data['short_percent_r']
long_wr   = data['long_percent_r']
composite = (1 - short_wr) * (1 - long_wr)

feat = composite.dropna()
lo, hi = np.nanpercentile(feat, 2), np.nanpercentile(feat, 98)
thresholds = np.linspace(lo, hi, N_THRESHOLDS)
print(f'oversold composite  range [{lo:.4g}, {hi:.4g}]')

p_above, p_below = compute_matrix(composite, thresholds, HORIZONS, data)
write_xlsx(p_above, p_below, OUT_DIR / 'composite.xlsx')
