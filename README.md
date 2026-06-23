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

# Check status
python run.py --workspace btc_daily_14days --status

# Run a family of seeds
python run.py --workspace btc_daily_14days --family rsi

# Read a result
python run.py --workspace btc_daily_14days --read rsi_14

# Combined signal probe (after all families tested)
python run.py --workspace btc_daily_14days --probe

# Backtest the combined signal
python run.py --workspace btc_daily_14days --backtest
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
| `--family <name>` | Run all pending seeds in a family |
| `--node <id>` | Run a single node |
| `--read <id>` | Print significant rows from a node's matrix |
| `--probe` | Current P(up) estimate — best node per family, Naive Bayes combined |
| `--backtest` | Historical accuracy of the combined signal, bucketed by model edge |
| `--regen` | Regenerate xlsx files without changing node status |

`--families a,b,c` narrows `--probe` and `--backtest` to specific families.

---

## Workspaces

Each workspace is a self-contained research unit: one asset, one interval, one horizon range.

### `btc_daily_14days`

| Field | Value |
|-------|-------|
| Asset | BTC-USD |
| Interval | 1d |
| Horizons | +1d → +14d |
| Start date | 2015-01-01 |
| Data sources | yfinance (ohlcv), CoinMetrics (on-chain), yfinance (DXY) |
| Status | **65 nodes tested** across 12 families |

**Key findings:**

| Family | Verdict | Best signal |
|--------|---------|-------------|
| on_chain | strong | MVRV < 0.79 → +35pp at +14d; MVRV > 3.3 → −14pp |
| volatility | strong | Realized vol < 10 (compression) → +24pp at +14d |
| rsi | strong | RSI > 87 → +20pp at +7d; oversold bounces short-term then reverses |
| ma | strong | Price > 130% above 200MA → −26pp at +14d |
| macd | strong | Extreme negative histogram → +11pp at +3d |
| roc | strong | ROC-7 > 0.11 → +10pp at +7d (momentum continuation) |
| volume | strong | volume_ratio_14 > 1.83 → +16pp at +3d (short-horizon only) |
| williams_r | moderate | Extreme overbought → +15pp at +14d |
| drawdown | moderate | Recovery ratio > 0.43 → +18pp at +14d |
| dxy | moderate | DXY +3% over 20d → −15pp BTC at +7d |
| cycle | moderate | Phase 0.15–0.40 (4–18 months post-halving) → +11pp at +14d |
| stoch | redundant | Mirrors RSI/WR mathematically |

---

### `nasdaq_hourly_24hrs`

| Field | Value |
|-------|-------|
| Asset | ^IXIC (NASDAQ Composite) |
| Interval | 1h |
| Horizons | +1h → +24h |
| Start date | 2024-07-01 (yfinance 730-day hourly limit) |
| Data sources | yfinance (ohlcv, ^VIX, ^TNX) |
| Status | **29 seed nodes pending** across 10 families |

Families: `rsi`, `ma`, `macd`, `roc`, `volatility`, `volume`, `vix`, `treasury`, `time_of_day`, `day_of_week`

Loop not yet run. Run with:

```bash
python run.py --workspace nasdaq_hourly_24hrs --family rsi
```

---

## Adding a new workspace

1. Create `workspaces/<name>/universe.json`:

```json
{
  "meta": {
    "workspace": "<name>",
    "asset": {"provider": "yfinance", "ticker": "^IXIC", "interval": "1h"},
    "horizons": [1, 2, ..., 24],
    "start_date": "2024-07-01"
  },
  "families": {
    "rsi": [
      {
        "id": "rsi_14", "family": "rsi", "category": "price_momentum",
        "feature": "rsi", "params": {"period": 14},
        "data": ["ohlcv"], "derived_from": null, "status": "pending"
      }
    ]
  }
}
```

2. Create `workspaces/<name>/findings.json` with `{}`.
3. Run `python run.py --workspace <name> --status` to verify.

> **Note:** yfinance hourly data is limited to the last ~730 days. Set `start_date` accordingly.

---

## Adding a new feature

In `data/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series:
    return close / close.rolling(period).mean() - 1.0

# Add to compute() dispatch dict:
'my_feature': lambda: _my_feature(c, p['period']),
```

Then add a node to the relevant workspace's `universe.json` with `"feature": "my_feature"` and run it.

---

## Available data sources

| Source key | Ticker / API | Columns |
|------------|-------------|---------|
| `ohlcv` | Workspace asset (yfinance) | open, high, low, close, volume |
| `dxy` | DX-Y.NYB (yfinance daily) | dxy |
| `vix` | ^VIX (yfinance, same interval as workspace) | vix |
| `treasury` | ^TNX (yfinance, same interval as workspace) | tnx |
| `coinmetrics` | CoinMetrics community API | mvrv, hash_rate, adr_act_cnt, tx_cnt |

Specify sources in the node's `"data"` array, e.g. `["ohlcv", "vix"]`.
