"""Residual mean-reversion analysis — binned P(price up) + Spearman correlation.

Run:  python research/residual/mean_reversion.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from research.framework import load_data

OUT_DIR  = Path(__file__).parent / 'output'
HORIZONS = [30, 60, 90]
BINS     = [-np.inf, -0.5, -0.25, 0.0, 0.25, 0.5, np.inf]
LABELS   = ['< -0.5', '-0.5→-0.25', '-0.25→0', '0→+0.25', '+0.25→+0.5', '> +0.5']

OUT_DIR.mkdir(exist_ok=True)
data  = load_data()
resid = data['log_price_residual']
price = data['price_usd'] if 'price_usd' in data.columns else np.exp(data['log_price_usd'])
bins  = pd.cut(resid, bins=BINS, labels=LABELS)

print(f'{"Residual bin":<15}', end='')
for h in HORIZONS:
    print(f'  +{h}d P(price up)    n', end='')
print()
print('─' * (15 + len(HORIZONS) * 20))

rows = []
for label in LABELS:
    mask = bins == label
    row  = {'bin': label}
    print(f'{label:<15}', end='')
    for h in HORIZONS:
        future_up = (price.shift(-h) > price).astype(float)
        valid     = mask & future_up.notna()
        n         = valid.sum()
        pct       = future_up[valid].mean() * 100 if n > 0 else float('nan')
        row[h]    = pct
        print(f'  {pct:>8.1f}%  {n:>4}', end='')
    print()
    rows.append(row)

print('\nSpearman correlation (current residual vs future price direction):')
print(f'  {"Horizon":<10}  {"ρ":>8}  {"p-value":>10}  {"significant?":>12}')
for h in HORIZONS:
    future_up = (price.shift(-h) > price).astype(float)
    valid     = future_up.notna() & resid.notna()
    rho, p    = spearmanr(resid[valid], future_up[valid])
    sig       = 'YES' if p < 0.05 else 'no'
    print(f'  +{h}d        {rho:>+8.4f}  {p:>10.4f}  {sig:>12}')

fig, axes = plt.subplots(1, len(HORIZONS), figsize=(14, 5), sharey=True)
fig.suptitle('P(price up) by current distance from log trend', fontsize=13)
df = pd.DataFrame(rows).set_index('bin')
for ax, h in zip(axes, HORIZONS):
    pcts   = df[h].values
    colors = ['steelblue' if p > 50 else 'tomato' for p in pcts]
    ax.bar(range(len(LABELS)), pcts, color=colors, edgecolor='white', width=0.7)
    ax.axhline(50, color='black', linestyle='--', linewidth=0.8)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha='right', fontsize=8)
    ax.set_title(f'+{h}d horizon')
    ax.set_ylim(0, 100)
    ax.set_ylabel('P(price up)' if h == HORIZONS[0] else '')
    ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
out = OUT_DIR / 'mean_reversion.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'\nSaved → {out}')
