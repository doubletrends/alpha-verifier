# Agent Loop — Procedure

Purely procedural. Follow these steps in order each session.

---

## 0. Orient

```
python run.py --status
```

Read the table. Pick the next family with `pending > 0` that has not been worked on yet. Prefer families in this order: work through seed families before moving to derived-only families.

---

## 1. Run seeds for the chosen family

```
python run.py --family <name>
```

This runs all pending seed nodes in the family and writes:
- `workspaces/btc_daily_14days/<family>/<node_id>.xlsx` — the matrix
- `workspaces/btc_daily_14days/<family>/log.jsonl` — metadata + base rates

---

## 2. Read the results

```
python run.py --read <node_id>
```

Each xlsx has three tabs:
- `base_rate` — unconditional P(price_up) per horizon (reference)
- `above` — P(price_up | feature > threshold), first row = base rate
- `below` — P(price_up | feature < threshold), first row = base rate

Read the `above` and `below` tabs. Compare each cell against the **base rate row at the top of the same tab**.

---

## 3. Assess edge — BTC uptrend bias warning

BTC has a persistent uptrend. The unconditional win rate is ~52–56% depending on horizon. **Do not treat any value above 50% as edge.** The relevant comparison is always conditional vs. base rate.

A condition shows **meaningful edge** when:
- It deviates from base rate by **> 10pp** (positive or negative) on at least one horizon
- The deviation is **consistent across consecutive horizons** (not a single spike)
- The sample size `n` is **≥ 50**

A condition that is within ±10pp of base rate across all horizons — record it and move on.

---

## 4. Decide: derive or exhaust

**Derive** if any of the following are true:
- One seed showed meaningful edge → test parameter variants (e.g. RSI14 showed edge → try RSI10, RSI18)
- Two seeds show opposing behavior → test their spread or ratio
- A seed shows edge only in extreme quantiles → test a normalized distance-from-extreme feature
- A pattern is visible but noisy → test a smoothed version

**Exhaust** if:
- All seeds are coinflips with no consistent pattern
- All reasonable derivatives have been tested and none improve on seeds
- You have tested ≥ 3 derived features with no meaningful edge found

When exhausted, go to step 5 before moving to the next family.

---

## 5. Record findings

After exhausting a family, update `workspaces/btc_daily_14days/findings.json`:

```json
{
  "rsi": {
    "verdict": "strong",
    "best_node": "rsi_18",
    "key_finding": "RSI > 87 → +20pp at +7d; deep oversold bounces short-term then reverses at +14d",
    "notes": "rsi_divergence_7_14 shows strong mean-reversion character"
  }
}
```

`verdict` should be one of: `strong`, `moderate`, `weak`, `redundant`.

Then return to step 0 and pick the next family.

---

## 6. Add a derived feature

**Step 6a — Write the feature function** into `data/features.py`:

```python
def _my_new_feature(close: pd.Series, ...) -> pd.Series:
    ...
    return result

# Add to the dispatch dict inside compute():
'my_new_feature': lambda: _my_new_feature(c, p['param1'], ...),
```

**Step 6b — Append the node** to `workspaces/btc_daily_14days/universe.json` under the correct family:

```json
{
  "id": "rsi_spread_7_14",
  "family": "rsi",
  "category": "price_momentum",
  "feature": "rsi_spread",
  "params": {"fast": 7, "slow": 14},
  "data": ["ohlcv"],
  "derived_from": "rsi_7",
  "status": "pending"
}
```

**Step 6c — Run it:**

```
python run.py --node <new_node_id>
```

Then return to step 2.

---

## 7. After all families exhausted — probe and backtest

Once `--status` shows 0 pending across all families, run the combined signal probe:

```
python run.py --probe
```

This selects the best node per family (by peak |+7d| edge), evaluates current feature values, and outputs a Naive Bayes combined probability estimate across all families.

Then run the historical backtest:

```
python run.py --backtest
```

This backtests the combined signal on monthly sampling across the full history, bucketed by model edge, to validate whether the combined signal has directional accuracy.

---

## Key file locations

| Path | Purpose |
|---|---|
| `workspaces/btc_daily_14days/universe.json` | Master node list — add derived nodes here |
| `workspaces/btc_daily_14days/findings.json` | Human-readable family verdicts and key findings |
| `data/features.py` | Feature functions — add new ones here |
| `workspaces/btc_daily_14days/<family>/` | Matrix xlsx files + per-family log |
| `log.jsonl` | Global run log (all families) |
| `run.py` | CLI — `--family`, `--node`, `--status`, `--list`, `--read`, `--probe`, `--backtest` |
