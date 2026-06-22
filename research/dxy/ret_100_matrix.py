"""DXY 100-day return win-rate matrix.

Run:  python research/dxy/ret_100_matrix.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from research.framework import load_data, compute_matrix, write_xlsx

OUT_DIR = Path(__file__).parent / 'output'
HORIZONS = list(range(15, 731, 15))   # +15d, +30d, ... +720d
N_THRESHOLDS = 30

OUT_DIR.mkdir(exist_ok=True)
data = load_data()
feat = data['dxy_ret_100'].dropna()
lo = np.nanpercentile(feat, 2)
hi = np.nanpercentile(feat, 98)
thresholds = np.linspace(lo, hi, N_THRESHOLDS)

print(f'dxy_ret_100  range [{lo:.4g}, {hi:.4g}]')

p_above, p_below = compute_matrix(data['dxy_ret_100'], thresholds, HORIZONS, data)
write_xlsx(p_above, p_below, OUT_DIR / 'ret_100_matrix.xlsx')
