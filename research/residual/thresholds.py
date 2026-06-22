"""Residual threshold sweep — find levels where P(price up/down) exceeds 90% / 95%.

Run:  python research/residual/thresholds.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from research.framework import load_data

OUT_DIR  = Path(__file__).parent / 'output'
HORIZONS = [30, 90, 180, 365]
MIN_N    = 30

OUT_DIR.mkdir(exist_ok=True)
data  = load_data()
resid = data['log_price_residual']
price = data['price_usd'] if 'price_usd' in data.columns else np.exp(data['log_price_usd'])

future_up   = {h: (price.shift(-h) > price) for h in HORIZONS}
future_down = {h: (price.shift(-h) < price) for h in HORIZONS}


def scan(thresholds, zone):
    future = future_up if zone == 'buy' else future_down
    rows   = []
    for t in thresholds:
        mask = resid < t if zone == 'buy' else resid > t
        row  = {'threshold': t}
        for h in HORIZONS:
            valid = mask & future[h].notna()
            n     = int(valid.sum())
            pct   = float(future[h][valid].mean() * 100) if n >= MIN_N else np.nan
            row[f'+{h}d_pct'] = pct
            row[f'+{h}d_n']   = n
        rows.append(row)
    return pd.DataFrame(rows)


buy  = scan(np.round(np.arange(-1.8, 0.05, 0.05), 2), 'buy')
sell = scan(np.round(np.arange( 0.0, 1.85, 0.05), 2), 'sell')


def print_zone(df, zone):
    direction = 'P(price up)' if zone == 'buy' else 'P(price down)'
    label     = 'BUY  residual < threshold' if zone == 'buy' else 'SELL residual > threshold'
    print(f'\n── {label} ──────────────────────────────────────────')
    header = f'  {"threshold":>9}'
    for h in HORIZONS:
        header += f'  +{h}d {direction}    n'
    print(header)
    print('  ' + '─' * (len(header) - 2))
    for _, row in df.iterrows():
        t    = row['threshold']
        line = f'  {t:>+9.2f}'
        flag = ''
        for h in HORIZONS:
            pct = row[f'+{h}d_pct']
            n   = int(row[f'+{h}d_n'])
            if np.isnan(pct):
                line += f'  {"—":>14}  {n:>4}'
            else:
                marker = ' ★' if pct >= 95 else (' ▶' if pct >= 90 else '  ')
                line  += f'  {pct:>10.1f}%{marker}  {n:>4}'
                if pct >= 90:
                    flag = '  ← ≥95%' if pct >= 95 else '  ← ≥90%'
        print(line + flag)


print_zone(buy,  'buy')
print_zone(sell, 'sell')
print('\n  ▶ = ≥90%   ★ = ≥95%')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['steelblue', 'darkorange', 'green', 'purple']
for ax, (df, zone, title) in zip(axes, [
    (buy,  'buy',  'BUY — residual < threshold\nP(price up at horizon)'),
    (sell, 'sell', 'SELL — residual > threshold\nP(price down at horizon)'),
]):
    for h, col in zip(HORIZONS, colors):
        valid = df[f'+{h}d_pct'].notna()
        ax.plot(df.loc[valid, 'threshold'], df.loc[valid, f'+{h}d_pct'],
                label=f'+{h}d', color=col, linewidth=1.6)
    ax.axhline(90, color='red',     linestyle='--', linewidth=0.9, label='90%')
    ax.axhline(95, color='darkred', linestyle=':',  linewidth=0.9, label='95%')
    ax.axhline(50, color='gray',    linestyle=':',  linewidth=0.6)
    ax.set_xlabel('log_price_residual threshold')
    ax.set_ylabel('Probability (%)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(linestyle=':', alpha=0.5)

plt.tight_layout()
out = OUT_DIR / 'thresholds.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'\nSaved → {out}')
