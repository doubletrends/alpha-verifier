![Python](https://img.shields.io/badge/Python-v3.12-3776AB?logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-v1.13-8CAAE6?logo=scipy&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-v2.0-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-v2.2-150458?logo=pandas&logoColor=white)

# Empirical Winning-Condition Testing Pipeline

An agentic, fully automatic pipeline that batch-tests empirical *winning conditions* on any financial asset. You declare a universe of conditions — each a feature $X$ (RSI, moving-average deviation, realized volatility, on-chain valuation, …) paired with a forward horizon $h$ — and an agent drives the pipeline through every one against history, unattended, computing:

$$ P\big(\text{price}_{t+h} > \text{price}_t \mid X_t \in \text{condition}\big) - P\big(\text{price}_{t+h} > \text{price}_t\big) $$

That is: conditioned on the feature, how much does the probability of an up-move deviate from its unconditional baseline? There is no fitted model and no forecast in the machine-learning sense — only conditional counting over the historical record, run in batch across hundreds of feature-and-parameter combinations and then reduced to a single estimate via shrinkage-weighted Naive Bayes.

The pipeline is organized around **workspaces**, each pairing one asset and sampling interval with its own feature universe. Two are included: `btc_daily_14days` (Bitcoin, daily bars, $+1$ to $+14$ day horizons) and `nasdaq_hourly_24hrs` (NASDAQ Composite, hourly bars, $+1$ to $+24$ hour horizons).

* * *

## Abstract

Most technical indicators are asserted rather than measured. This project inverts that: it takes an indicator as a raw feature, slices history by the value of that feature, and reports the empirical probability of a subsequent up-move at each horizon, expressed as a deviation from the unconditional base rate. The unit of work is a **node** — one feature evaluated at one parameter set (e.g. `rsi_14`, `dxy_ma_ratio_50`). Each node produces a self-contained spreadsheet describing the conditional win-rate surface. Nodes are grouped into **families** (rsi, ma, volatility, on_chain, …); the best node per family is selected by peak signal strength, and the survivors are combined into a single current probability estimate under a Naive Bayes independence assumption. A sampled walk-forward backtest reports the historical directional accuracy of that combined signal.

* * *

## 1. Methodology (Conditional Winrate Estimation)

### 1.1 Base rate

For a horizon $h$ (in bars), the outcome at time $t$ is the indicator $\mathbb{1}[\text{close}_{t+h} > \text{close}_t]$. The unconditional **base rate** is its historical mean:

$$ p_0(h) = P\big(\text{close}_{t+h} > \text{close}_t\big) $$

Rows with fewer than a minimum count ($n < 20$) are treated as undefined. Because the last $h$ bars have no realized outcome, they are dropped from every horizon's sample.

### 1.2 Conditional surface

A node computes a feature series $X_t$, then evaluates 30 thresholds spanning its 2nd–98th empirical percentiles. For each threshold $x$ and horizon $h$ it records two cumulative (CDF-style) deviations:

$$ \Delta_{>}(x, h) = P\big(\text{close}_{t+h} > \text{close}_t \mid X_t > x\big) - p_0(h) $$

$$ \Delta_{<}(x, h) = P\big(\text{close}_{t+h} > \text{close}_t \mid X_t < x\big) - p_0(h) $$

A value of $0$ means the condition carries no edge; positive is a bullish tilt, negative a bearish tilt. All figures are reported in percentage points (pp) of deviation from $p_0$.

### 1.3 Local (PDF) recovery

The surfaces of §1.2 are cumulative — the CDF of the win-outcome in the feature $X$. In the textbook sense the cumulative distribution function and its density are

$$ F(x) = P(X \leq x) = \int_{-\infty}^{x} f(u)\,du, \qquad f(x) = \frac{dF}{dx}. $$

The pipeline applies the discrete counterpart: the **local** (PDF) win-rate over a threshold interval is recovered by differencing adjacent cumulative rows, giving the edge for observations whose feature value falls *within* an interval rather than above/below a threshold. This yields the PDF sheet, from which the combiner interpolates the edge at any current feature value.

Each node is written as an `.xlsx` with three colour-scaled tabs (green = bullish edge, red = bearish edge):

| Tab | Quantity |
|-----|----------|
| **PDF** — `P(up \| X ≈ x) − P(up)` | Local win rate at each feature value |
| **CDF above** — `P(up \| X > x) − P(up)` | Cumulative from the top threshold down |
| **CDF below** — `P(up \| X < x) − P(up)` | Cumulative from the bottom threshold up |

Observation counts throughout report the longest-horizon count, which is the most conservative (longer horizons lose more tail rows to unrealized outcomes).

* * *

## 2. Signal Combination (Naive Bayes over Families)

For each family, `--probe` selects the single node with the highest peak absolute deviation at the pivot horizon (subject to $n \geq 30$ slices), then combines the survivors in log-odds space. Assuming the features are conditionally independent given the outcome, the combined estimate at horizon $h$ is

$$ \operatorname{logit} p_\text{comb}(h) = \operatorname{logit} p_0(h) + \sum_i w_i \Big[\operatorname{logit}\big(p_0(h) + \delta_i(h)\big) - \operatorname{logit} p_0(h)\Big], $$

where $\delta_i(h)$ is family $i$'s local deviation at the current feature value and $p_\text{comb}$ is recovered with the logistic $\sigma$. The independence assumption is deliberately optimistic — correlated signals (e.g. RSI and Williams %R measure nearly the same thing) will inflate the combined edge, and the tool flags this explicitly in its output.

### 2.1 Shrinkage weighting

Thin slices are down-weighted so a handful of coincidental observations cannot dominate. With a CLT floor at $n_0 = 30$ and half-weight scale $N_0 = 50$:

$$ w(n) = \begin{cases} 0 & n < 30 \\[4pt] \dfrac{n - 30}{(n - 30) + 50} & n \geq 30 \end{cases} $$

The weight is $0$ at the floor, $0.5$ at $n = 80$, and asymptotes to $1$ as $n \to \infty$.

* * *

## 3. Repository Structure

Adding a workspace requires **no changes to root code**. Root code owns the stable engine contracts; each workspace owns everything specific to its asset.

```
run.py               ← thin CLI dispatcher
workspace.py         ← Workspace class; loads universe.json + optional plugin
data/
  features.py        ← FeatureRegistry: register() / compute()
  fetcher.py         ← SourceRegistry:  register_source() / fetch()
engine/
  matrix.py          ← base rate + conditional CDF surfaces
  writer.py          ← xlsx rendering, colour scales, PDF recovery
  combiner.py        ← Naive Bayes log-odds combination + shrinkage
tree/
  tree.py            ← node traversal over universe.json
workspaces/
  <name>/
    universe.json    ← asset, horizons, feature families, all config
    plugin.py        ← (optional) custom features and data sources
    findings.json    ← structured per-family verdicts, written by --findings
```

Both the feature layer and the data-source layer are registries: a plugin calls `register()` / `register_source()` at import time and the engine picks it up with no root edits.

* * *

## 4. Pipeline (CLI Workflow)

All commands take `--workspace <name>` (default `btc_daily_14days`):

```bash
pip install -r requirements.txt
python run.py --workspace btc_daily_14days --status
python run.py --workspace btc_daily_14days --family rsi
python run.py --workspace btc_daily_14days --read rsi_14
python run.py --workspace btc_daily_14days --probe
python run.py --workspace btc_daily_14days --backtest
python run.py --workspace btc_daily_14days --findings
```

| Command | Stage | Output |
|---------|-------|--------|
| `--status` | Inventory | Pending / tested / skipped counts per family |
| `--list` | Inventory | All pending node IDs, grouped by family |
| `--next` | Compute | Runs the next pending node |
| `--family <name>` | Compute | Runs all pending nodes in a family |
| `--node <id>` | Compute | Runs a single node → writes its `.xlsx` |
| `--regen` | Compute | Regenerates `.xlsx` files without changing node status |
| `--read <id>` | Inspect | Prints significant rows from a node's surface |
| `--probe` | Combine | Current $P(\text{up})$ estimate — best node per family, Naive Bayes combined |
| `--backtest` | Validate | Walk-forward directional accuracy of the combined signal, bucketed by model edge |
| `--findings` | Report | Writes `findings.json` — structured verdict per family |

`--families a,b,c` narrows `--probe` / `--backtest` to a subset of families.

* * *

## 5. Workspaces (Asset Configuration)

A workspace is defined entirely by `universe.json`. Key `meta` fields:

| Field | Purpose |
|-------|---------|
| `asset` | `{provider, ticker, interval}` — the traded series |
| `horizons` | Forward horizons in bars (days or hours, inferred from `interval`) |
| `start_date` | History start; intraday requests are clamped to Yahoo's trailing ~730-day window |
| `n_thresholds` | Threshold grid resolution (default 30) |
| `display_horizons` | 3–4 horizons shown in `--read` / `--backtest` tables |
| `sample_freq` | pandas offset for backtest sampling — `MS` (monthly) for daily, `W-MON` (weekly) for intraday |
| `min_obs` | Minimum valid observations required to run a node |
| `read_min_dev`, `read_min_n` | Thresholds for a row to count as "significant" in `--read` |

| Workspace | Asset | Interval | Horizons | History | Extras |
|-----------|-------|----------|----------|---------|--------|
| `btc_daily_14days` | BTC-USD | 1d | $+1$…$+14$ d | from 2015 | CoinMetrics on-chain, halving-cycle features |
| `nasdaq_hourly_24hrs` | ^IXIC | 1h | $+1$…$+24$ h | trailing ~730 d | VIX / treasury cross-asset, time-of-day |

### 5.1 Built-in data sources

| Source key | Provider | Columns |
|------------|----------|---------|
| `ohlcv` | yfinance (workspace asset + interval) | open, high, low, close, volume |
| `vix` | yfinance `^VIX` (workspace interval) | vix |
| `treasury` | yfinance `^TNX` (workspace interval) | tnx |
| `dxy` | yfinance `DX-Y.NYB` (always daily) | dxy |
| `coinmetrics` | CoinMetrics community API *(btc plugin)* | mvrv, hash_rate, adr_act_cnt, tx_cnt |

* * *

## 6. Extending the Engine (Plugins & Registries)

**A new workspace** needs only `workspaces/<name>/universe.json` (asset + `meta` + `families`) and a stub `findings.json`; verify with `--status`.

**A cross-asset feature** (works for any OHLCV asset) is registered in `data/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series:
    ...

register('my_feature', lambda d, p: _my_feature(d['close'], p['period']))
```

**A workspace-specific feature or data source** (on-chain metric, options IV, custom API) goes in `workspaces/<name>/plugin.py`, which is imported automatically when the workspace initializes:

```python
from data import features, fetcher

def _my_source(start: str, asset: dict) -> pd.DataFrame:      # must return a DatetimeIndex-ed frame
    ...

fetcher.register_source('my_source', _my_source)
features.register('my_feature', lambda d, p: _my_feature(d, p['period']))
```

Root code is never touched. Reference new sources in a node definition via `"data": ["ohlcv", "my_source"]`.

* * *

## 7. Results

Findings are batch-generated: any asset reachable through the data layer can be dropped into a workspace and scanned end-to-end with no changes to the engine. To illustrate that capability across asset classes and timeframes, we ran two workspaces — Bitcoin on daily bars and the NASDAQ Composite on hourly bars — and summarize the per-family findings each produced. All figures are deviations from the unconditional base rate $p_0$ in percentage points (pp); $n$ is the longest-horizon observation count. The pipeline evaluates every family and records it whether or not an edge is found.

### 7.1 Bitcoin (BTC-USD, daily, $+1$ to $+14$ d)

The largest and most horizon-persistent edges came from **valuation and volatility extremes** rather than oscillators. Deep undervaluation on the on-chain MVRV ratio ($X < 0.79$, $n = 85$) preceded up-moves at $+12.6$ / $+23.0$ / $+34.3$ pp above baseline at the $+3$ / $+7$ / $+14$ day horizons — the strongest single edge in the universe — while the symmetric euphoria condition ($X > 3.3$) reached $-13.9$ pp at $+14$ d. Volatility compression showed the same monotonic-in-horizon shape: 21-day realized volatility below $19$ ($n = 190$) preceded $+11.3$ / $+18.0$ / $+19.5$ pp. Both strengthen with horizon, consistent with slow mean reversion rather than short-term timing.

**Overextension** carried the opposite sign: price more than $72\%$ above its 100-day moving average ($n = 83$) preceded $-20.1$ pp at $+14$ d, and price stretched more than $33\%$ below it preceded $+14.8$ pp — reversion visible from both tails. Momentum, in contrast, showed **continuation**: 5-day rate-of-change above $0.14$ ($n = 241$) preceded $+8.2$ / $+12.2$ / $+13.9$ pp, building over the horizon. Elevated 7-day volume was **short-lived** ($+12.2$ pp at $+3$ d fading to $+6.8$ pp by $+14$ d), and the stochastic, Williams %R, and RSI families were largely **redundant**, producing near-identical surfaces with no incremental value recorded for the stochastic family over RSI.

### 7.2 NASDAQ Composite (^IXIC, hourly, $+1$ to $+24$ h)

On intraday NASDAQ data the dominant edges were **cross-asset and volatility-driven**. The 10-year treasury yield gave the single strongest signal — a low-rate regime ($X < 3.73\%$, $n = 63$) preceded $+12.8$ / $+21.5$ / $+22.9$ / $+38.7$ pp at $+3$ / $+6$ / $+12$ / $+24$ h — though, over a window covering only 2024-07 onward, this is best read as a regime marker than a repeatable trigger. VIX behaved as a clean leading indicator: hourly VIX spikes preceded an immediate NASDAQ drop, while an elevated VIX level relative to its moving average ($n = 187$) preceded a $+10.9$ pp recovery by $+24$ h. Bollinger band width was the sharpest volatility signal, and asymmetric: band expansion preceded up to $+21.9$ pp at $+24$ h, whereas extreme compression preceded $-45.2$ pp — volatility expansion as bullish continuation, compression as a bearish trap.

The momentum families (MACD, RSI, moving-average cross) all showed **continuation across every horizon** with no mean-reversion component; notably, overbought RSI stayed bullish here, the opposite of Bitcoin's mixed intraday-versus-longer-term behavior. Three families returned **no edge** above threshold — day-of-week, hour-of-day, and volume — the last a direct contrast with Bitcoin, where 7-day volume did carry a short-horizon edge. Taken together, the two workspaces show the same engine surfacing structurally different, asset-specific behavior from identical machinery.

* * *

## References

**Data sources**
- Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) — OHLCV, VIX, 10-year treasury yield, US dollar index
- [CoinMetrics Community API](https://docs.coinmetrics.io/api/v4) — Bitcoin on-chain metrics (MVRV, hash rate, active addresses)

**Statistical methods**
- Naive Bayes classification and the conditional-independence assumption — Hand & Yu (2001), *Idiot's Bayes — Not So Stupid After All?*, International Statistical Review 69(3)
- Log-odds (logit) additivity of evidence — Good (1950), *Probability and the Weighing of Evidence*
- Shrinkage / partial pooling of small-sample rates — Efron & Morris (1975), *Data Analysis Using Stein's Estimator and Its Generalizations*, JASA 70(350)

**Technical indicators**
- Wilder (1978), *New Concepts in Technical Trading Systems* — RSI, ATR
- Appel (2005), *Technical Analysis: Power Tools for Active Investors* — MACD
- Bollinger (2001), *Bollinger on Bollinger Bands*
