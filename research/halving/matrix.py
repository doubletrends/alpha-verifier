"""Years-since-halving win-rate matrix.

Run:  python research/halving/matrix.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from research.framework import load_data, compute_matrix, write_xlsx

OUT_DIR = Path(__file__).parent / 'output'
HORIZONS = list(range(90, 1441, 90))   # +90d, +180d, ... +1440d
N_THRESHOLDS = 30

OUT_DIR.mkdir(exist_ok=True)
data = load_data()
feat = data['years_since_halving'].dropna()
lo = np.nanpercentile(feat, 2)
hi = np.nanpercentile(feat, 98)
thresholds = np.linspace(lo, hi, N_THRESHOLDS)

print(f'years_since_halving  range [{lo:.4g}, {hi:.4g}]')

p_above, p_below = compute_matrix(data['years_since_halving'], thresholds, HORIZONS, data)
write_xlsx(p_above, p_below, OUT_DIR / 'matrix.xlsx')
