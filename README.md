![Python](https://img.shields.io/badge/Python-v3.12-3776AB?logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-v1.13-8CAAE6?logo=scipy&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-v2.0-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-v2.2-150458?logo=pandas&logoColor=white)

# Empirical Winning-Condition Testing Pipeline on Any Financial Asset

An agentic, fully automatic pipeline that batch-tests empirical *winning conditions* on any financial asset. You declare a universe of conditions — each a feature $X$ (RSI, moving-average deviation, realized volatility, on-chain valuation, …) paired with a forward horizon $h$ — and an agent drives the pipeline through every one against history, unattended, computing:

$$ P\big(\text{price}_{t+h} > \text{price}_t \mid X_t \in \text{condition}\big) - P\big(\text{price}_{t+h} > \text{price}_t\big) $$

That is: conditioned on the feature, how much does the probability of an up-move deviate from its unconditional baseline? There is no fitted model and no forecast in the machine-learning sense — only conditional counting over the historical record, run in batch across hundreds of feature-and-parameter combinations and then reduced to a single estimate via shrinkage-weighted Naive Bayes.

The pipeline is organized around **workspaces**, each pairing one asset and sampling interval with its own feature universe. Two are included: `btc_daily_14days` (Bitcoin, daily bars, +1 to +14 day horizons) and `nasdaq_hourly_24hrs` (NASDAQ Composite, hourly bars, +1 to +24 hour horizons).

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

## 2. Signal Combination and Validation

For each family, `--probe` selects the single node with the highest peak absolute deviation at the pivot horizon (subject to $n \geq 30$ slices), then combines the survivors in log-odds space. Assuming the features are conditionally independent given the outcome, the combined estimate at horizon $h$ is

$$ \mathrm{logit}\,p_\text{comb}(h) = \mathrm{logit}\,p_0(h) + \sum_i w_i \Big[\mathrm{logit}\big(p_0(h) + \delta_i(h)\big) - \mathrm{logit}\,p_0(h)\Big], $$

where $\delta_i(h)$ is family $i$'s local deviation at the current feature value and $p_\text{comb}$ is recovered with the logistic $\sigma$. The independence assumption is deliberately optimistic — correlated signals (e.g. RSI and Williams %R measure nearly the same thing) will inflate the combined edge, and the tool flags this explicitly in its output.

### 2.1 Shrinkage weighting

Thin slices are down-weighted so a handful of coincidental observations cannot dominate. With a CLT floor at $n_0 = 30$ and half-weight scale $N_0 = 50$:

$$ w(n) = \begin{cases} 0 & n < 30 \\ \dfrac{n - 30}{(n - 30) + 50} & n \geq 30 \end{cases} $$

The weight is $0$ at the floor, $0.5$ at $n = 80$, and asymptotes to $1$ as $n \to \infty$.

* * *

### 2.2 Shuffle-null validation

A node's headline number is the peak absolute deviation across ~30 threshold bins. That
statistic is a **maximum over many noisy estimates**, so it is biased upward even when the
feature carries no information at all: with $n \approx 50$ observations in a bin the standard
error of a rate is $\sqrt{0.25/50} \approx 7$ pp, and the maximum over 30 such bins lands near
10 pp by construction. Comparing a peak against a fixed pp threshold therefore cannot separate
signal from noise — it mostly measures how many bins were searched.

`--validate` builds the null distribution of that same statistic directly. The outcome series is
circularly shifted against the feature many times; each shift preserves the autocorrelation of
both series — which matters, because overlapping $h$-bar outcomes are strongly serially
correlated — while destroying any real alignment between them. The node's real peak is then read
as a quantile of that null:

$$ p = \frac{1 + \#\{\text{shifts with peak} \geq \text{real peak}\}}{1 + n_\text{shifts}} $$

The add-one estimator (Davison & Hinkley) keeps $p$ strictly positive: the observed statistic is
itself one draw from the null, so $p = 0$ is never a valid conclusion. Its floor, $1/(n_\text{shifts}+1)$,
is also the resolution limit — a sweep of $k$ tests needs a Bonferroni $\alpha/k$, and no number
below the floor can be resolved. Verdicts reflect this explicitly:

| Verdict | Meaning |
|---------|---------|
| `structure` | Clears $\alpha/k$; survives correction for the size of the search |
| `nominal` | Clears $\alpha$ alone — what a single-node view would call an edge |
| `underpowered` | Sits at the resolution floor; might clear $\alpha/k$, but this many shifts cannot show it |
| `noise` | Indistinguishable from the null |

Results are written to `validation.json` (or `validation.<event>.json` for a non-default outcome).
Re-running with `--families` **merges** into that file rather than replacing it, and the Bonferroni
denominator is taken over every test the file contains — narrowing the run must not quietly shrink
the correction for a search that was already made wide.

* * *

### 2.3 Outcomes

The engine measures $P(\text{outcome} \mid X \in \text{condition}) - P(\text{outcome})$. The
outcome was historically fixed to "price rose over the next $h$ bars"; it is now pluggable, so the
same conditional-counting machinery can be aimed at events where the structure is stronger.

| `--outcome` | Event | Default base rate (BTC, +14d) |
|-------------|-------|------------------------------|
| `up` | $\text{close}_{t+h} > \text{close}_t$ — the original, still the default | 56.5% |
| `drawdown` | $\min(\text{close}_{t+1..t+h}) / \text{close}_t - 1 < -\theta$ — a peak-to-trough loss from $t$, not merely a lower close at $t+h$ | 21.7% ($\theta = 0.10$) |
| `runup` | Mirror of `drawdown`: a gain exceeding $+\theta$ at any point within $h$ | — |
| `vol_high` | Realized volatility over $(t, t+h]$ exceeded its own trailing median. The reference median is built from *backward*-looking $h$-bar volatility only, so nothing unavailable at $t$ enters the comparison | 47.4% |

`--threshold` sets $\theta$ for `drawdown` / `runup` (default `0.10`). Outcomes are a registry
like features and data sources, so a workspace plugin can call `outcomes.register(...)` at import
time with no root edits.

Node status (`pending` / `tested`) tracks the **default outcome only**. Under any other outcome
the engine writes to a per-event subdirectory (`<family>/dd10/<node>.xlsx`), leaves node status
untouched, and `--family` targets every node in the family rather than only pending ones — so the
65 existing surfaces stay exactly where they are and no two outcomes overwrite each other.

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
  outcomes.py        ← OutcomeRegistry: the event being measured (up / drawdown / vol)
  writer.py          ← xlsx rendering, colour scales, PDF recovery
  combiner.py        ← Naive Bayes log-odds combination + shrinkage
  validate.py        ← circular-shift null test + Bonferroni verdicts
tree/
  tree.py            ← node traversal over universe.json
workspaces/
  <name>/
    universe.json    ← asset, horizons, feature families, all config
    plugin.py        ← (optional) custom features and data sources
    findings.json    ← structured per-family verdicts, written by --findings
    validation.json  ← per-node null-test p-values and verdicts, written by --validate
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
python run.py --workspace btc_daily_14days --validate
python run.py --workspace btc_daily_14days --validate --outcome drawdown --threshold 0.10
python run.py --workspace btc_daily_14days --family ma --outcome drawdown
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
| `--validate` | Falsify | Shuffle-null test of every tested node; writes `validation.json` |

`--families a,b,c` narrows `--probe` / `--backtest` / `--validate` to a subset of families.
`--outcome <name>` (with `--threshold` where applicable) re-aims any compute or reporting
command at a different event; `--shifts N` sets the resampling depth for `--validate`.

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
| `outcome` | `{name, params}` — the event to measure; defaults to `{"name": "up"}` |

| Workspace | Asset | Interval | Horizons | History | Extras |
|-----------|-------|----------|----------|---------|--------|
| `btc_daily_14days` | BTC-USD | 1d | +1…+14 d | from 2015 | CoinMetrics on-chain, halving-cycle features |
| `nasdaq_hourly_24hrs` | ^IXIC | 1h | +1…+24 h | trailing ~730 d | VIX / treasury cross-asset, time-of-day |

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

Findings are batch-generated: any asset reachable through the data layer can be dropped into a
workspace and scanned end-to-end with no changes to the engine. To exercise that across asset
classes and timeframes we ran two workspaces — Bitcoin on daily bars and the NASDAQ Composite on
hourly bars — and then put every node through `--validate`.

Everything below is reported **against the null**, not against a fixed pp threshold. A peak
deviation on its own is not evidence: the maximum over ~30 bins is biased upward even for an
uninformative feature (§2.2), and for this universe the null median of that statistic is ~10 pp
at +3d and ~17 pp at +14d — the same magnitude as most "edges" a naive read would report.

### 7.1 Direction is not predictable in either workspace

| Workspace | Tests | `structure` | `nominal` (p<0.05) | Expected by chance |
|-----------|-------|-------------|--------------------|--------------------|
| `btc_daily_14days` | 195 | **0** | 14 | 9.8 |
| `nasdaq_hourly_24hrs` | 132 | **0** | 5 | 6.6 |

Nothing survives Bonferroni correction in either. Bitcoin's 14 nominal hits against 9.8 expected
is the excess a search of this size produces from noise alone; NASDAQ returns *fewer* nominal hits
than chance would give. The strongest directional family per workspace:

| Workspace | Family | Node | $h$ | Real | Null p95 | $p$ | Verdict |
|-----------|--------|------|-----|------|----------|-----|---------|
| BTC | ma | `ma_ratio_100` | +3d | 29.7 | 23.2 | 0.0026 | nominal |
| BTC | volatility | `atr_14` | +14d | 38.7 | 32.8 | 0.0065 | nominal |
| BTC | on_chain | `mvrv` | +14d | 34.2 | 35.6 | 0.081 | **noise** |
| BTC | cycle | `days_since_halving` | +14d | 31.5 | 40.9 | 0.717 | **noise** |
| NASDAQ | volatility | `bb_width_20` | +24h | 46.3 | 37.8 | 0.015 | nominal |
| NASDAQ | treasury | `tnx_ret_1` | +3h | 21.9 | 21.4 | 0.045 | nominal |

Two of these are worth naming because they were the headline results before validation. The MVRV
valuation signal — deep on-chain undervaluation preceding up-moves at +34 pp over baseline at
+14d — is a genuinely large deviation, and it is *below* its own null p95 of 35.6: MVRV moves
slowly enough that a 30-bin maximum on a 4,200-bar sample reaches that size by chance. The same
holds for the halving-cycle features, where the null p95 reaches 40.9 pp. On the NASDAQ side the
low-rate treasury regime, previously reported at +38.7 pp at +24h, survives only nominally over a
window that is mostly one regime.

### 7.2 Risk is predictable — the same features, a different question

Re-aiming the identical 65 nodes at `--outcome drawdown --threshold 0.10` (a 10% peak-to-trough
loss from $t$ within $h$ bars) inverts the picture. Same features, same history, same machinery,
39,000 shifts, same $\alpha = 2.56 \times 10^{-4}$:

| Outcome | `structure` | `nominal` | Expected by chance |
|---------|-------------|-----------|--------------------|
| `up` — direction | 0 / 195 | 14 / 195 | 9.8 |
| `drawdown` — 10% loss within $h$ | **4** / 195 | **58** / 195 | 9.8 |

58 nominal hits is a 5.9× excess over chance, and four tests clear Bonferroni outright:

| Node | Family | $h$ | Real | Null p95 | $p$ |
|------|--------|-----|------|----------|-----|
| `bb_pct_20` | volatility | +3d | 27.7 | 9.4 | 0.00003 |
| `bb_pct_20` | volatility | +7d | 29.2 | 13.6 | 0.00003 |
| `drawdown_7` | drawdown | +3d | 24.8 | 9.9 | 0.00003 |
| `ma_cross_7_21` | ma | +3d | 19.8 | 11.6 | 0.00013 |

Nine of twelve families reach at least nominal significance on drawdown, against five on
direction — and the drawdown p-values are three to four orders of magnitude smaller. The
strongest single condition is monotone and large: price more than 120% above its 200-day moving
average precedes a 10% drawdown within 14 days at **+39 pp over a 21.7% base rate** ($n = 82$),
rising monotonically across every threshold beneath it.

Note what does *not* transfer. The oscillator families that dominate technical-analysis folklore
(`stoch`, `williams_r`) are the weakest on both outcomes, and `cycle` — the halving-phase overlay —
is the single worst performer against the null on both. What carries the drawdown signal is
volatility position (`bb_pct_20`), recent drawdown itself (`drawdown_7`), and trend structure
(`ma_cross_7_21`, `ma_ratio_100`): volatility clustering and the leverage effect, both of which
are well documented and neither of which arbitrage removes.

### 7.3 Reading the two together

The same features, machinery and history carry no usable information about *whether* price rises,
and substantial information about *whether it falls hard*. That is the expected shape rather than
a surprise: direction is the most competed-away quantity in a liquid market, while tail risk is
driven by mechanisms that persist precisely because they are not arbitrageable in the same way.

Two caveats bound the risk result:

1. **The null is conservative, not liberal, here.** Circular shifting preserves each series' own
   autocorrelation but not the joint regime structure, so a shift can align an early-cycle feature
   with late-cycle outcomes. Where both series share a slow regime component — as they do for
   crash risk — this inflates the null, so the drawdown result is if anything understated. A block
   bootstrap would tighten it.
2. **Significance is not tradability.** These are conditional probabilities measured in-sample
   across the full history, not a walk-forward backtest with costs. `--backtest` remains the check
   that a combined signal would have been usable in real time, and it is not run for the drawdown
   outcome here.

Per-family verdicts, p-values and null quantiles are written to `validation.json` and summarised
into `findings.json` by `--findings`; `verdict` there is the null-test result, and a family with
no validation sweep reports `unvalidated` rather than claiming an edge.

* * *

## References

**Data sources**
- Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) — OHLCV, VIX, 10-year treasury yield, US dollar index
- [CoinMetrics Community API](https://docs.coinmetrics.io/api/v4) — Bitcoin on-chain metrics (MVRV, hash rate, active addresses)

**Statistical methods**
- Naive Bayes classification and the conditional-independence assumption — Hand & Yu (2001), *Idiot's Bayes — Not So Stupid After All?*, International Statistical Review 69(3)
- Log-odds (logit) additivity of evidence — Good (1950), *Probability and the Weighing of Evidence*
- Shrinkage / partial pooling of small-sample rates — Efron & Morris (1975), *Data Analysis Using Stein's Estimator and Its Generalizations*, JASA 70(350)
- Add-one permutation p-values and resampling resolution — Davison & Hinkley (1997), *Bootstrap Methods and Their Application*, CUP, §4.2
- Resampling that preserves serial dependence — Politis & Romano (1994), *The Stationary Bootstrap*, JASA 89(428)
- Multiple testing over a searched universe of rules — White (2000), *A Reality Check for Data Snooping*, Econometrica 68(5)

**Technical indicators**
- Wilder (1978), *New Concepts in Technical Trading Systems* — RSI, ATR
- Appel (2005), *Technical Analysis: Power Tools for Active Investors* — MACD
- Bollinger (2001), *Bollinger on Bollinger Bands*
