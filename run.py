"""Agent run entry point.

Usage:
    python run.py --workspace btc_daily --family rsi
    python run.py --workspace btc_daily --node rsi_14
    python run.py --workspace btc_daily --next
    python run.py --workspace btc_daily --list
    python run.py --workspace btc_daily --status
    python run.py --workspace btc_daily --read rsi_14
    python run.py --workspace btc_daily --regen
    python run.py --workspace btc_daily --regen --family rsi
    python run.py --workspace btc_daily --regen --node rsi_14
    python run.py --workspace btc_daily --probe
    python run.py --workspace btc_daily --probe --families vol,rsi
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import features, fetcher
from engine import matrix as engine
from engine import writer
from engine import combiner as cmb
from tree.tree import (
    load_tree, save_tree, all_nodes, find_node,
    pending_in_family, all_in_family, next_pending,
)

AGENT_DIR    = Path(__file__).resolve().parent
N_THRESHOLDS = 30
MIN_OBS      = 100

_READ_HORIZONS = ['+3d', '+7d', '+14d']
_READ_MIN_DEV  = 3.0
_READ_MIN_N    = 50


# ── workspace ─────────────────────────────────────────────────────────────────

class Workspace:
    def __init__(self, name: str):
        self.dir  = AGENT_DIR / 'workspaces' / name
        tree      = load_tree(self.dir / 'universe.json')
        meta      = tree['meta']
        self.horizons   = meta['horizons']
        self.start_date = meta['start_date']
        self.asset      = meta['asset']

    @property
    def tree_path(self) -> Path:
        return self.dir / 'universe.json'

    @property
    def log(self) -> Path:
        return self.dir / 'log.jsonl'

    def output_dir(self, family: str) -> Path:
        return self.dir / family

    def family_log(self, family: str) -> Path:
        return self.dir / family / 'log.jsonl'


# ── core run ──────────────────────────────────────────────────────────────────

def run_node(ws: Workspace, node_id: str, regen: bool = False) -> None:
    tree = load_tree(ws.tree_path)
    node = find_node(tree, node_id)

    family  = node['family']
    out_dir = ws.output_dir(family)

    print(f"\n{'=' * 60}")
    print(f"Node     : {node['id']}")
    print(f"Family   : {family}  |  Category: {node['category']}")
    print(f"Feature  : {node['feature']}  params={node['params']}")
    print(f"Sources  : {node['data']}")
    if node.get('derived_from'):
        print(f"Derived  : from {node['derived_from']}")
    print(f"{'=' * 60}\n")

    print('Fetching data...')
    data = fetcher.fetch(node['data'], start=ws.start_date, asset=ws.asset).dropna(subset=['close'])
    if data.empty:
        print('Fetch returned empty DataFrame — skipping.')
        return
    print(f'Rows: {len(data)}  ({data.index[0].date()} → {data.index[-1].date()})')

    print(f"Computing feature '{node['feature']}'...")
    feat = features.compute(data, node['feature'], node['params']).reindex(data.index)

    n_valid = int(feat.notna().sum())
    if n_valid < MIN_OBS:
        print(f'Only {n_valid} valid observations — skipping.')
        if not regen:
            node['status'] = 'skipped'
            save_tree(tree, ws.tree_path)
        return

    lo         = float(np.nanpercentile(feat.dropna(), 2))
    hi         = float(np.nanpercentile(feat.dropna(), 98))
    thresholds = np.linspace(lo, hi, N_THRESHOLDS)
    print(f'Feature p2→p98: [{lo:.4g}, {hi:.4g}]')

    base_rate        = engine.compute_base_rate(data, ws.horizons)
    p_below, p_above = engine.compute_matrix(feat, thresholds, ws.horizons, data)

    out_path = out_dir / f'{node_id}.xlsx'
    writer.write_xlsx(p_below, p_above, base_rate, out_path, node_id)

    br   = {row: round(float(base_rate.loc[row, 'win_rate']), 2) for row in base_rate.index}
    spot = '  '.join(f'+{h}d={br[f"+{h}d"]:.1f}%' for h in [1, 3, 7, 14])
    print(f'Base rates: {spot}')

    if not regen:
        log_entry = {
            'node_id':      node_id,
            'family':       family,
            'category':     node['category'],
            'feature':      node['feature'],
            'params':       node['params'],
            'derived_from': node.get('derived_from'),
            'timestamp':    datetime.now(timezone.utc).isoformat(),
            'matrix_path':  str(out_path.relative_to(ROOT)),
            'n_obs':        n_valid,
            'horizons':     ws.horizons,
            'n_thresholds': N_THRESHOLDS,
            'base_rate':    br,
        }
        for log_path in (ws.family_log(family), ws.log):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')

        node['status'] = 'tested'
        save_tree(tree, ws.tree_path)

    print(f"\nDone → {out_path.relative_to(ROOT)}")


# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_family(ws: Workspace, family_name: str) -> None:
    tree    = load_tree(ws.tree_path)
    pending = pending_in_family(tree, family_name)
    if not pending:
        families = sorted({n['family'] for n in all_nodes(tree)})
        print(f"No pending nodes in family '{family_name}'.")
        print(f"Available families: {', '.join(families)}")
        return
    print(f"Running {len(pending)} pending node(s) in family '{family_name}'...")
    for node in pending:
        run_node(ws, node['id'])


def cmd_next(ws: Workspace) -> None:
    tree = load_tree(ws.tree_path)
    node = next_pending(tree)
    if node is None:
        print('All nodes tested or skipped.')
        return
    run_node(ws, node['id'])


def cmd_list(ws: Workspace) -> None:
    tree   = load_tree(ws.tree_path)
    by_fam = defaultdict(list)
    for n in all_nodes(tree):
        if n.get('status') == 'pending':
            by_fam[n['family']].append(n['id'])
    if not by_fam:
        print('No pending nodes.')
        return
    print(f"\nPending nodes by family:")
    for fam in sorted(by_fam):
        ids = ', '.join(by_fam[fam])
        print(f"  {fam:<20s}  {ids}")


def cmd_status(ws: Workspace) -> None:
    tree   = load_tree(ws.tree_path)
    by_fam = defaultdict(lambda: {'pending': 0, 'tested': 0, 'skipped': 0})
    for n in all_nodes(tree):
        by_fam[n['family']][n.get('status', 'pending')] += 1
    print(f"\n{'family':<20s}  {'pending':>7}  {'tested':>6}  {'skipped':>7}")
    print('-' * 46)
    for fam in sorted(by_fam):
        c = by_fam[fam]
        print(f"  {fam:<18s}  {c['pending']:>7}  {c['tested']:>6}  {c['skipped']:>7}")
    totals = {k: sum(by_fam[f][k] for f in by_fam) for k in ('pending', 'tested', 'skipped')}
    print('-' * 46)
    print(f"  {'TOTAL':<18s}  {totals['pending']:>7}  {totals['tested']:>6}  {totals['skipped']:>7}")


def cmd_read(ws: Workspace, node_id: str) -> None:
    tree = load_tree(ws.tree_path)
    node = find_node(tree, node_id)
    path = ws.output_dir(node['family']) / f'{node_id}.xlsx'

    if not path.exists():
        print(f"No output file: {path}")
        return

    wb = openpyxl.load_workbook(path, data_only=True)

    print(f"\n{node_id}  [{node['family']}]  {node['feature']}  params={node['params']}")

    for sheet_name in (writer.SHEET_ABOVE, writer.SHEET_BELOW):
        ws_sheet = wb[sheet_name]
        rows = list(ws_sheet.iter_rows(values_only=True))
        hdrs = rows[4]
        ci   = {h: i for i, h in enumerate(hdrs) if h is not None}

        significant = []
        for row in rows[5:]:
            cond  = row[ci['condition']]
            n_raw = row[ci['n']]
            if not n_raw:
                continue
            n = int(n_raw)
            if n < _READ_MIN_N:
                continue
            devs = {h: row[ci[h]] for h in _READ_HORIZONS if h in ci and row[ci[h]] is not None}
            if any(abs(v) >= _READ_MIN_DEV for v in devs.values()):
                significant.append((cond, n_raw, devs))

        direction = 'above' if '>' in sheet_name else 'below'
        if not significant:
            print(f"\n  [{direction}]  no rows ≥ {_READ_MIN_DEV}pp with n ≥ {_READ_MIN_N}")
            continue

        print(f"\n  [{direction}]")
        print(f"  {'condition':<26}  {'n':>12}  {'+3d':>8}  {'+7d':>8}  {'+14d':>8}")
        print(f"  {'-'*26}  {'-'*12}  {'--------':>8}  {'--------':>8}  {'--------':>8}")
        for cond, n_raw, devs in significant:
            def fmt(h):
                v = devs.get(h)
                return f'{v:+.1f}pp' if v is not None else '   n/a'
            print(f"  {str(cond):<26}  {str(n_raw):>12}  {fmt('+3d'):>8}  {fmt('+7d'):>8}  {fmt('+14d'):>8}")


def cmd_probe(ws: Workspace, families_filter: list[str] | None = None) -> None:
    import datetime

    tree = load_tree(ws.tree_path)

    by_family: dict[str, list[dict]] = defaultdict(list)
    for node in all_nodes(tree):
        if node.get('status') == 'tested':
            by_family[node['family']].append(node)

    if families_filter:
        by_family = {f: ns for f, ns in by_family.items() if f in families_filter}

    best: dict[str, tuple[dict, float]] = {}
    for fam, nodes in by_family.items():
        for node in nodes:
            path = ws.output_dir(fam) / f'{node["id"]}.xlsx'
            if not path.exists():
                continue
            try:
                fn_tbl = cmb.load_fn_table(path)
                pk     = cmb.peak_signal(fn_tbl, '+7d', min_n=30)
                if fam not in best or pk > best[fam][1]:
                    best[fam] = (node, pk)
            except Exception:
                pass

    _cache: dict[tuple, pd.DataFrame] = {}

    def get_data(sources: list) -> pd.DataFrame:
        key = tuple(sorted(sources))
        if key not in _cache:
            _cache[key] = fetcher.fetch(list(sources), start=ws.start_date, asset=ws.asset).dropna(subset=['close'])
        return _cache[key]

    print('Fetching latest data...')
    ohlcv_data = get_data(['ohlcv'])
    base_rate  = engine.compute_base_rate(ohlcv_data, ws.horizons)
    br_series  = base_rate['win_rate']

    print(f"\n=== Signal Probe [{ws.dir.name}] — {datetime.date.today()} ===")
    print(f"\nOne node per family, auto-selected by peak |+7d| edge (n≥30 slices):\n")
    print(f"  {'family':<18}  {'node':<26}  {'current x':>11}  {'+3d':>8}  {'+7d':>8}  {'+14d':>8}")
    print(f"  {'-'*18}  {'-'*26}  {'-'*11}  {'-'*8}  {'-'*8}  {'-'*8}")

    contributions = []

    for fam in sorted(best):
        node, _ = best[fam]
        path    = ws.output_dir(fam) / f'{node["id"]}.xlsx'
        try:
            data = get_data(node['data'])
            if data.empty:
                raise ValueError('empty data')
            feat      = features.compute(data, node['feature'], node['params']).reindex(data.index)
            valid     = feat.dropna()
            if valid.empty:
                raise ValueError('no valid feature values')
            current_x = float(valid.iloc[-1])

            fn_tbl = cmb.load_fn_table(path)
            devs   = cmb.eval_at(fn_tbl, current_x)
            devs_h = devs[[c for c in devs.index if c.startswith('+')]]
            contributions.append((node['id'], devs_h))

            def fmtd(h: str) -> str:
                v = devs_h[h] if h in devs_h.index else np.nan
                return f'{float(v):+.1f}pp' if pd.notna(v) else '   n/a'

            print(f"  {fam:<18}  {node['id']:<26}  {current_x:>11.4g}  {fmtd('+3d'):>8}  {fmtd('+7d'):>8}  {fmtd('+14d'):>8}")

        except Exception as e:
            print(f"  {fam:<18}  {node['id']:<26}  {'[error]':>11}  {'':>8}  {'':>8}  {'':>8}  — {e}")

    if not contributions:
        print('\nNo contributions computed.')
        return

    key_h  = ['+3d', '+7d', '+14d']
    result = cmb.combine(contributions, br_series, horizons=key_h)

    n_sig = len(contributions)
    print(f"\nNaive Bayes combination [{n_sig} signals, one per family]:\n")
    print(f"  {'horizon':<8}  {'base rate':>10}  {'combined':>10}  {'edge':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for h in key_h:
        br   = float(br_series.loc[h]) if h in br_series.index else np.nan
        comb = float(result.loc[h, 'combined']) if h in result.index else np.nan
        edge = float(result.loc[h, 'edge'])     if h in result.index else np.nan
        print(f"  {h:<8}  {br:>9.1f}%  {comb:>9.1f}%  {edge:>+10.1f}pp")

    print()
    print('  * Assumes feature independence. Correlated signals may inflate the estimate.')
    print()


def cmd_backtest(ws: Workspace, families_filter: list[str] | None = None) -> None:
    tree = load_tree(ws.tree_path)

    by_family: dict[str, list[dict]] = defaultdict(list)
    for node in all_nodes(tree):
        if node.get('status') == 'tested':
            by_family[node['family']].append(node)
    if families_filter:
        by_family = {f: ns for f, ns in by_family.items() if f in families_filter}

    best: dict[str, tuple[dict, float]] = {}
    for fam, nodes in by_family.items():
        for node in nodes:
            path = ws.output_dir(fam) / f'{node["id"]}.xlsx'
            if not path.exists():
                continue
            try:
                fn_tbl = cmb.load_fn_table(path)
                pk     = cmb.peak_signal(fn_tbl, '+7d', min_n=30)
                if fam not in best or pk > best[fam][1]:
                    best[fam] = (node, pk)
            except Exception:
                pass

    _cache: dict[tuple, pd.DataFrame] = {}

    def get_data(sources: list) -> pd.DataFrame:
        key = tuple(sorted(sources))
        if key not in _cache:
            _cache[key] = fetcher.fetch(list(sources), start=ws.start_date, asset=ws.asset).dropna(subset=['close'])
        return _cache[key]

    print('Fetching data and computing feature series...')
    ohlcv_data = get_data(['ohlcv'])
    base_rate  = engine.compute_base_rate(ohlcv_data, ws.horizons)
    br_series  = base_rate['win_rate']

    selected: list[dict]               = []
    feat_map:  dict[str, pd.Series]    = {}
    fn_map:    dict[str, pd.DataFrame] = {}

    for fam in sorted(best):
        node, _ = best[fam]
        path    = ws.output_dir(fam) / f'{node["id"]}.xlsx'
        try:
            data = get_data(node['data'])
            if data.empty:
                continue
            feat = features.compute(data, node['feature'], node['params']).reindex(data.index)
            fn_map[node['id']]   = cmb.load_fn_table(path)
            feat_map[node['id']] = feat
            selected.append(node)
            print(f"  loaded  {node['id']}  ({fam})")
        except Exception as e:
            print(f"  [skip]  {node['id']}: {e}")

    if not selected:
        print('No nodes loaded.')
        return

    close    = ohlcv_data['close']
    min_date = close.index[250]
    max_date = close.index[-15]
    sample_dates = []
    for dt in pd.date_range(start=min_date, end=max_date, freq='MS'):
        pos = close.index.searchsorted(dt)
        if pos < len(close.index):
            sample_dates.append(close.index[pos])

    print(f'\nRunning combiner across {len(sample_dates)} monthly dates...')

    key_h   = ['+3d', '+7d', '+14d']
    results = []

    for dt in sample_dates:
        pos = close.index.get_loc(dt)

        contribs = []
        for node in selected:
            fs = feat_map[node['id']]
            if dt not in fs.index or pd.isna(fs.loc[dt]):
                continue
            devs   = cmb.eval_at(fn_map[node['id']], float(fs.loc[dt]))
            devs_h = devs[[c for c in devs.index if c.startswith('+')]]
            contribs.append((node['id'], devs_h))

        if not contribs:
            continue

        result = cmb.combine(contribs, br_series, horizons=key_h)
        p0     = close.iloc[pos]
        row    = {'date': dt}
        for h_str, h_int in [('3', 3), ('7', 7), ('14', 14)]:
            h_lbl = f'+{h_str}d'
            row[f'edge_{h_str}d']   = float(result.loc[h_lbl, 'edge']) if h_lbl in result.index else np.nan
            fp = pos + h_int
            row[f'actual_{h_str}d'] = (1 if close.iloc[fp] > p0 else 0) if fp < len(close) else np.nan
        results.append(row)

    df = pd.DataFrame(results)

    print(f"\n=== Backtest [{ws.dir.name}] — monthly sampling, N={len(df)}, {len(selected)} signals ===")
    print(f"  Nodes: {', '.join(n['id'] for n in selected)}\n")

    bins   = [-200, -10, -5,  0,  5, 10, 200]
    labels = ['< -10pp', '-10 to -5', '-5 to 0', '0 to +5', '+5 to +10', '> +10pp']
    df['bucket'] = pd.cut(df['edge_7d'], bins=bins, labels=labels)

    print(f"  Bucketed by +7d model edge:\n")
    print(f"  {'bucket':<14}  {'N':>4}  {'avg +7d edge':>13}  {'+3d acc':>8}  {'+7d acc':>8}  {'+14d acc':>9}")
    print(f"  {'-'*14}  {'-'*4}  {'-'*13}  {'-'*8}  {'-'*8}  {'-'*9}")
    for lbl in labels:
        sub = df[df['bucket'] == lbl]
        if len(sub) == 0:
            continue
        print(f"  {lbl:<14}  {len(sub):>4}  {sub['edge_7d'].mean():>+12.1f}pp"
              f"  {sub['actual_3d'].dropna().mean()*100:>7.0f}%"
              f"  {sub['actual_7d'].dropna().mean()*100:>7.0f}%"
              f"  {sub['actual_14d'].dropna().mean()*100:>8.0f}%")

    print(f"\n  Directional accuracy (edge > 0 → predicted up, edge < 0 → predicted down):")
    for h_str in ['3', '7', '14']:
        sub    = df.dropna(subset=[f'actual_{h_str}d', f'edge_{h_str}d'])
        pred   = sub[f'edge_{h_str}d'] > 0
        actual = sub[f'actual_{h_str}d'].astype(bool)
        n_cor  = (pred == actual).sum()
        print(f"    +{h_str}d:  {(pred==actual).mean()*100:.0f}%  ({n_cor}/{len(sub)})")

    print(f"\n  5 most bullish + 5 most bearish model calls (by +7d edge):\n")
    print(f"  {'date':<12}  {'edge +7d':>9}  {'+7d actual':>11}  {'+14d actual':>12}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*11}  {'-'*12}")
    for _, row in pd.concat([df.nlargest(5, 'edge_7d'), df.nsmallest(5, 'edge_7d')]).iterrows():
        a7  = ('up'   if row['actual_7d']  == 1 else 'DOWN') if pd.notna(row['actual_7d'])  else 'n/a'
        a14 = ('up'   if row['actual_14d'] == 1 else 'DOWN') if pd.notna(row['actual_14d']) else 'n/a'
        print(f"  {str(row['date'].date()):<12}  {row['edge_7d']:>+8.1f}pp  {a7:>11}  {a14:>12}")
    print()


def cmd_regen(ws: Workspace, family_name: str | None, node_id: str | None) -> None:
    tree = load_tree(ws.tree_path)

    if node_id:
        nodes = [find_node(tree, node_id)]
    elif family_name:
        nodes = all_in_family(tree, family_name)
        if not nodes:
            print(f"No nodes found in family '{family_name}'.")
            return
    else:
        nodes = all_nodes(tree)

    print(f"Regenerating {len(nodes)} node(s)...")
    failed = []
    for node in nodes:
        try:
            run_node(ws, node['id'], regen=True)
        except Exception as e:
            print(f"  [error] {node['id']}: {e} — will retry")
            failed.append(node['id'])

    if failed:
        print(f"\nRetrying {len(failed)} failed node(s)...")
        for nid in failed:
            run_node(ws, nid, regen=True)


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Winrate matrix research pipeline')
    parser.add_argument('--workspace', metavar='NAME', default='btc_daily_14days', help='Workspace name (default: btc_daily_14days)')
    parser.add_argument('--regen',    action='store_true', help='Regenerate xlsx without changing status')
    parser.add_argument('--family',   metavar='NAME',      help='Target family')
    parser.add_argument('--node',     metavar='ID',        help='Target node')
    parser.add_argument('--families', metavar='FAM,...',   help='Comma-separated families for --probe/--backtest')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--next',     action='store_true', help='Run the next pending node')
    group.add_argument('--list',     action='store_true', help='List pending nodes by family')
    group.add_argument('--status',   action='store_true', help='Show status table by family')
    group.add_argument('--read',     metavar='ID',        help='Print matrix summary for a node')
    group.add_argument('--probe',    action='store_true', help='Combined P(up) probe across all families')
    group.add_argument('--backtest', action='store_true', help='Backtest probe signal on monthly history')

    args = parser.parse_args()
    ws   = Workspace(args.workspace)

    fam_filter = [f.strip() for f in args.families.split(',')] if args.families else None

    if args.read:
        cmd_read(ws, args.read)
    elif args.probe:
        cmd_probe(ws, fam_filter)
    elif args.backtest:
        cmd_backtest(ws, fam_filter)
    elif args.regen:
        cmd_regen(ws, args.family, args.node)
    elif args.next:
        cmd_next(ws)
    elif args.list:
        cmd_list(ws)
    elif args.status:
        cmd_status(ws)
    elif args.family:
        cmd_family(ws, args.family)
    elif args.node:
        run_node(ws, args.node)
    else:
        parser.print_help()
