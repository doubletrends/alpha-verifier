"""Log-price residual win-rate matrix.

Run:  python research/residual/matrix.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from research.framework import load_data, compute_matrix, write_xlsx

OUT_DIR      = Path(__file__).parent / 'output'
HORIZONS     = list(range(15, 731, 15))   # +15d, +30d, … +720d  (48 cols, 2 years)
N_THRESHOLDS = 30

OUT_DIR.mkdir(exist_ok=True)
data = load_data()
feat = data['log_price_residual'].dropna()
lo   = np.nanpercentile(feat, 2)
hi   = np.nanpercentile(feat, 98)
thresholds = np.linspace(lo, hi, N_THRESHOLDS)

print(f'log_price_residual  range [{lo:.4g}, {hi:.4g}]')

p_above, p_below = compute_matrix(data['log_price_residual'], thresholds, HORIZONS, data)
write_xlsx(p_above, p_below, OUT_DIR / 'matrix.xlsx')
