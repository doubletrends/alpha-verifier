# Agent Loop — Procedure

Purely procedural. Follow these steps in order each session.
Pass `--workspace <name>` to all commands.

---

## 0. Orient

```
python run.py --workspace <name> --status
```

Read the table. Pick the next family with `pending > 0`. Prefer seed families before derived-only families.

---

## 1. Run seeds for the chosen family

```
python run.py --workspace <name> --family <family>
```

Writes:
- `workspaces/<name>/<family>/<node_id>.xlsx` — the matrix
- `workspaces/<name>/<family>/log.jsonl` — metadata + base rates

---

## 2. Read the results

```
python run.py --workspace <name> --read <node_id>
```

Prints significant rows from the `above` and `below` tabs. All values are deviations from the base rate — 0 = no edge.

---

## 3. Assess edge

A condition shows **meaningful edge** when:
- Deviates from base rate by **> 10pp** on at least one horizon
- Deviation is **consistent across consecutive horizons** (not a single spike)
- Sample size `n` **≥ 50**

Within ±10pp across all horizons — record it and move on.

---

## 4. Decide: derive or exhaust

**Derive** if any of the following are true:
- A seed showed meaningful edge → test parameter variants (e.g. RSI14 showed edge → try RSI10, RSI18)
- Two seeds show opposing behavior → test their spread or ratio
- A seed shows edge only in extreme quantiles → test a normalized distance-from-extreme feature
- A pattern is visible but noisy → test a smoothed version

**Exhaust** if:
- All seeds are coinflips with no consistent pattern
- All reasonable derivatives have been tested and none improve on seeds
- ≥ 3 derived features tested with no meaningful edge found

When exhausted, go to step 5 before picking the next family.

---

## 5. Record findings

Run `--findings` to auto-populate `best_node`, `peak_signal`, `verdict`, and `conditions` for every tested family:

```
python run.py --workspace <name> --findings
```

Then open `workspaces/<name>/findings.json` and add qualitative `notes` for this family — what the pattern means, what to watch for, cross-family observations. The `notes` field is preserved on every subsequent `--findings` run.

Return to step 0 and pick the next family.

---

## 6. Add a derived feature

**Step 6a — Register the feature.**

If the feature is specific to this workspace, add it to `workspaces/<name>/plugin.py`:

```python
from data import features

def _my_feature(data, period): ...

features.register('my_feature', lambda d, p: _my_feature(d, p['period']))
```

If the feature is cross-asset (useful across workspaces), add it to `data/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series: ...

register('my_feature', lambda d, p: _my_feature(d['close'], p['period']))
```

**Step 6b — Append the node** to `workspaces/<name>/universe.json` under the correct family:

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
python run.py --workspace <name> --node <new_node_id>
```

Then return to step 2.

---

## 7. After all families exhausted — probe and backtest

Once `--status` shows 0 pending across all families:

```
python run.py --workspace <name> --probe
python run.py --workspace <name> --backtest
python run.py --workspace <name> --findings
```

`--probe` evaluates current feature values and outputs a Naive Bayes combined P(up) estimate.
`--backtest` validates directional accuracy on sampled historical dates, bucketed by model edge.
`--findings` writes the final structured record for the completed workspace.

---

## Key file locations

| Path | Purpose |
|------|---------|
| `workspaces/<name>/universe.json` | Master node list — add derived nodes here |
| `workspaces/<name>/findings.json` | Family verdicts — machine-generated, human-annotated |
| `workspaces/<name>/plugin.py` | Workspace-specific features and data sources |
| `workspaces/<name>/<family>/` | Matrix xlsx files + per-family log |
| `data/features.py` | Cross-asset feature registry |
| `data/fetcher.py` | Cross-asset source registry |
