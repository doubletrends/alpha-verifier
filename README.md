# winrate-matrix

Empirical win-rate research pipeline for financial assets.

For any feature X and any forward horizon H, it answers:

> *P(price at t+H > price at t | X > threshold)* — does this condition predict up moves?

No model. No prediction. Just conditional counting from history, with shrinkage-weighted Naive Bayes combination across signal families.

---

## How it works

Each **node** is one feature × one parameter set (e.g. `rsi_14`, `vix_ma_ratio_20`). Running a node produces an `.xlsx` with three tabs:

| Tab | Question |
|-----|----------|
| **PDF** | P(up \| X ≈ x) − base rate — local win rate at each value of X |
| **CDF above** | P(up \| X > threshold) − base rate — running cumulative from top |
| **CDF below** | P(up \| X < threshold) − base rate — running cumulative from bottom |

Values are **deviations from the unconditional base rate** (so 0 = no edge). Color scale: green = bullish edge, red = bearish edge.

Nodes are organized into **families** (rsi, ma, vix, …). After all families are tested, `--probe` picks the best node per family and combines them with Naive Bayes log-odds to produce a current P(up) estimate across all horizons.

---

## Quick start

```bash
pip install -r requirements.txt

python run.py --workspace <name> --status
python run.py --workspace <name> --family rsi
python run.py --workspace <name> --read rsi_14
python run.py --workspace <name> --probe
python run.py --workspace <name> --backtest
python run.py --workspace <name> --findings
```

---

## CLI reference

```
python run.py --workspace <name> [command]
```

| Command | Description |
|---------|-------------|
| `--status` | Pending / tested / skipped counts per family |
| `--list` | List all pending node IDs |
| `--next` | Run the next pending node |
| `--family <name>` | Run all pending nodes in a family |
| `--node <id>` | Run a single node |
| `--read <id>` | Print significant rows from a node's matrix |
| `--probe` | Current P(up) estimate — best node per family, Naive Bayes combined |
| `--backtest` | Historical accuracy of the combined signal, bucketed by model edge |
| `--findings` | Write `findings.json` — structured verdict per family |
| `--regen` | Regenerate xlsx files without changing node status |

`--families a,b,c` narrows `--probe` and `--backtest` to specific families.

---

## Architecture

Adding a new workspace requires **no changes to root code**. Each workspace owns everything specific to it; root code owns the stable contracts.

```
run.py              ← thin CLI dispatcher
workspace.py        ← Workspace class; loads universe.json + plugin
data/
  features.py       ← FeatureRegistry; register() / compute()
  fetcher.py        ← SourceRegistry; register_source() / fetch()
engine/             ← matrix, combiner, writer — stable, never changes
tree/               ← node traversal — stable, never changes
workspaces/
  <name>/
    universe.json   ← asset, horizons, all workspace config
    plugin.py       ← (optional) custom features and data sources
    findings.json   ← structured family verdicts, written by --findings
```

---

## Built-in data sources

| Source key | Provider | Columns |
|------------|----------|---------|
| `ohlcv` | yfinance (workspace asset + interval) | open, high, low, close, volume |
| `dxy` | yfinance DX-Y.NYB (always daily) | dxy |
| `vix` | yfinance ^VIX (workspace interval) | vix |
| `treasury` | yfinance ^TNX (workspace interval) | tnx |
| `coinmetrics` | CoinMetrics community API *(btc_daily_14days plugin)* | mvrv, hash_rate, adr_act_cnt, tx_cnt |

Reference these in node definitions: `"data": ["ohlcv", "vix"]`.

---

## Adding a workspace

**Step 1** — create `workspaces/<name>/universe.json`:

```json
{
  "meta": {
    "workspace": "<name>",
    "asset": {"provider": "yfinance", "ticker": "SPY", "interval": "1d"},
    "horizons": [1, 2, 3, 5, 10, 20, 30],
    "start_date": "2010-01-01",
    "n_thresholds": 30,
    "display_horizons": [5, 10, 30],
    "sample_freq": "MS",
    "min_obs": 100,
    "read_min_dev": 10.0,
    "read_min_n": 50
  },
  "families": {}
}
```

| Field | Notes |
|-------|-------|
| `display_horizons` | 3–4 horizon numbers shown in `--read` and `--backtest` tables |
| `sample_freq` | pandas offset for backtest sampling: `"MS"` (monthly) for daily, `"W-MON"` (weekly) for intraday |
| `min_obs` | minimum valid observations required to run a node |
| `read_min_dev` | pp threshold for `--read` to show a row |
| `read_min_n` | minimum n for `--read` to show a row |

> **Note:** yfinance hourly data is limited to the last ~730 days. Set `start_date` accordingly.

**Step 2** — create `workspaces/<name>/findings.json`:

```json
{"workspace": "<name>", "generated": null, "display_horizons": [], "families": {}}
```

**Step 3** — if this workspace needs custom features or data sources, create `plugin.py` (see below). Otherwise omit it.

**Step 4** — verify: `python run.py --workspace <name> --status`

---

## Adding a feature

**Cross-asset features** (RSI, MA, ROC, …) belong in `data/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series:
    ...

register('my_feature', lambda d, p: _my_feature(d['close'], p['period']))
```

**Workspace-specific features** (on-chain metrics, options IV, …) belong in `workspaces/<name>/plugin.py`:

```python
from data import features

def _my_feature(data, period): ...

features.register('my_feature', lambda d, p: _my_feature(d, p['period']))
```

The plugin is loaded automatically when the workspace initializes. Root code is never touched.

---

## Adding a data source

**Universally available sources** belong in `data/fetcher.py`:

```python
def _my_source(start: str, asset: dict) -> pd.DataFrame: ...

register_source('my_source', _my_source)
```

**Workspace-specific sources** belong in `workspaces/<name>/plugin.py`:

```python
from data import fetcher

def _my_api(start: str, asset: dict) -> pd.DataFrame: ...

fetcher.register_source('my_source', _my_api)
```

Source functions must return a DataFrame with a `DatetimeIndex`. Add to node definitions: `"data": ["ohlcv", "my_source"]`.
