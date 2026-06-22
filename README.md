# winrate-matrix

**One question, answered empirically:**

> If condition Y is true today, what is the probability that price will be higher on day Z?

No model. No prediction. Just counting from history.

---

## Output

Every analysis produces one `.xlsx` file with two sheets:

| Sheet | Question |
|-------|----------|
| `above` | P(price at +t days > price now \| condition) — green = strong bullish signal |
| `below` | P(price at +t days < price now \| condition) — red = strong bearish signal |

- **Rows** — conditions swept across the feature's historical range (`x < k` or `x > k`)
- **Columns** — horizons: `+15d`, `+30d`, … `+720d`
- **Values** — probability as a percentage (e.g. `87.3%`)
- **`sample size (n)`** — number of observations matching the condition
- **Color scale** — above sheet: red→white→green. Below sheet: green→white→red.

---

## Architecture

```
winrate-matrix/
  config.py                         shared paths (ODS_CSV, DWD_CSV)
  pipeline/
    step_1_data_ingestion.py        fetch BTC price, on-chain, macro → ODS
    step_2_feature_engineering.py   compute all features             → DWD
    output/                         pipeline artifacts (gitignored)
      step1_ods.csv
      step2_dwd.csv
  research/
    framework.py                    shared engine: load_data, compute_matrix,
                                    compute_signal_matrix, write_xlsx
    residual/                       log-price residual signal
      features.py                   expanding-window trend fit (leak-free)
      matrix.py                     threshold × horizon win-rate matrix
      mean_reversion.py             binned P(price up) + Spearman correlation
      thresholds.py                 find 90% / 95% confidence levels
      output/                       signal artifacts (gitignored)
        cache.csv                   expanding-window residual cache
        matrix.xlsx
    mvrv/                           MVRV ratio signal
      matrix.py
      output/
        matrix.xlsx
    dxy/                            Dollar Index regime signal
      matrix.py                     20-day DXY return
      ret_30_matrix.py              30-day DXY return
      ret_100_matrix.py             100-day DXY return
      output/
        ret_20_matrix.xlsx
        ret_30_matrix.xlsx
        ret_100_matrix.xlsx
    volatility/                     30-day realized volatility signal
      matrix.py
      output/
        matrix.xlsx
    halving/                        Bitcoin cycle-position signal
      matrix.py
      cycle_transition.py
      output/
        matrix.xlsx
        cycle_transition.xlsx
    rsi/                            RSI 14 short-horizon signal
      matrix.py
      output/
        matrix.xlsx
    williams_r/                     Williams %R oversold composite signal
      composite.py
      plain_7.py
      plain_14.py
      output/
        composite.xlsx
        plain_7.xlsx
        plain_14.xlsx
```

**Data contract:** `pipeline/output/step2_dwd.csv` is the single input for all research scripts. No research script reads the ODS directly.

**Leak-free residual:** `research/residual/features.py` computes the log-price residual using an expanding window — for each date t, the trend is fitted on data[start:t] only. Result is cached to `research/residual/output/cache.csv` and reloaded on subsequent runs.

---

## Engine API (`research/framework.py`)

```python
# Load the feature warehouse (injects leak-free residual automatically)
data = load_data()

# Continuous feature × threshold sweep
p_above, p_below = compute_matrix(
    feature    = data['log_price_residual'],
    thresholds = np.linspace(-1.5, 1.5, 30),
    horizons   = list(range(15, 731, 15)),
    data       = data,
)
write_xlsx(p_above, p_below, OUT_DIR / 'matrix.xlsx')

# Event-based signal × parameter sweep
p_bull, p_bear = compute_signal_matrix(
    bull_fn  = lambda t: os_reversal(t),
    bear_fn  = lambda t: ob_reversal(t),
    sweep    = list(range(2, 51, 2)),
    horizons = list(range(15, 731, 15)),
    data     = data,
)
write_xlsx(p_bull, p_bear, path, sheet_names=('bullish', 'bearish'))
```

---

## Quick start

```bash
pip install -r requirements.txt

# First time: fetch data and build features
python pipeline/step_1_data_ingestion.py
python pipeline/step_2_feature_engineering.py

# Run research
python research/residual/matrix.py
python research/mvrv/matrix.py
python research/williams_r/composite.py
```

---

## Adding a new signal

1. Create `research/<signal>/` folder
2. Write your script — define `HORIZONS`, thresholds or sweep, signal logic
3. Call `compute_matrix` or `compute_signal_matrix`, then `write_xlsx`
4. Output goes to `research/<signal>/output/`

The framework handles computation, colour scale, and file writing.

---

## Key findings

| Signal | Condition | Horizon | P(price up) | n |
|--------|-----------|---------|-------------|---|
| Log-price residual | `x < -0.80` | +180d | ~87% | ~400 |
| Log-price residual | `x < -0.80` | +365d | ~100% | ~400 |
| Log-price residual | `x > +0.65` | +91d | ~5% (bearish) | ~500 |
| Realized vol 30d | `x < 18.31` | broad multi-horizon | ~89% to ~99% | 113 |
| Halving transition | `0.0y→0.5y` to `1.0y→1.25y` | cycle-transition | 100% | varies |
| Halving transition | `1.5y→2.0y` to `2.0y→2.75y` | cycle-transition bearish | 100% | varies |
| DXY 30d return | `x < -0.023` | +75d to +90d | ~88.5% to ~90.4% | 470 |
| DXY 100d return | `x < -0.070` | +75d to +90d | ~100% | 111 |
| WR oversold composite | both WR deep in oversold | +20d | ~69% | varies |
| WR oversold composite | neither WR oversold | +55d | ~83% (momentum) | varies |
| WR pair-state | `short>=long \| short=ob \| long=deep_os` | +9d to +11d | ~77.8% | 54 |
| WR pair-state | `short<long \| short=ob \| long=ob` | +5d | ~79.5% | 78 |

The log-price residual is the strongest signal. Low residual = deeply undervalued vs long-run trend = strong mean reversion over 6–12 months. Ultra-low 30-day realized volatility also marks an extreme bullish historical pocket: `realized_vol_30 < 18.31` produces roughly `89%` to `99%` BTC win rates across a broad span of forward horizons. The halving cycle-transition view makes the cycle geometry explicit: early-cycle transitions into the `1.0y→1.5y` zone form a win cluster, while transitions from roughly `1.5y→2.0y` into `2.0y→2.75y` form a loss cluster. DXY weakness appears to be a meaningful medium-horizon regime condition, with the strongest predictive representation concentrated around `+75d` to `+90d`: `dxy_ret_30` is the broadest robust expression, while `dxy_ret_100` contains the most extreme historical pocket. Williams %R captures short-term dynamics best as pair-state regime structure: `short<long | short=ob | long=ob` is strongest over the next 3–5 days, while `short>=long | short=ob | long=deep_os` tends to peak later around days 9–11.
