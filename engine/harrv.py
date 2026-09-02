"""
HAR-RV volatility forecasting, benchmarked against naive persistence and implied vol.

Three forecasters compete on the same target — realized volatility over the next h bars:

  HAR-RV      Corsi (2009). Regresses future RV on realized volatility measured over
              three lookbacks: daily, weekly (5) and monthly (22). The cascade is the
              *input* structure, not the horizon; h is chosen independently. Fit in
              logs, where RV is far closer to Gaussian, then transformed back.

  RV (naive)  Today's trailing RV, carried forward unchanged. Volatility is strongly
              persistent, so this is a genuinely hard baseline — most of what any vol
              model knows is "yesterday's vol". A forecaster that cannot beat it has
              learned nothing.

  IV          The option market's own forecast (DVOL for BTC), in annualized percent.
              This is the benchmark that decides whether a forecast is *tradeable*
              rather than merely accurate: IV already contains the market's estimate
              plus a variance risk premium, so beating RV-persistence is table stakes
              while beating IV is the actual bar.

Scoring uses QLIKE alongside R². Realized volatility is a noisy proxy for latent
volatility, and under a noisy proxy squared-error loss can rank forecasts incorrectly;
QLIKE is one of the loss functions that stays consistent (Patton 2011). It is computed
in variance units and is asymmetric — under-forecasting a spike is penalised much more
heavily than over-forecasting a calm patch, which is the asymmetry a risk manager wants.
"""

import numpy as np
import pandas as pd

# Crypto trades continuously, so a year is 365 bars rather than the ~252 of an
# equity calendar. Using 252 here would overstate annualized vol by ~20%.
ANNUALIZE = 365.0

HAR_LAGS = {'rv_d': 1, 'rv_w': 5, 'rv_m': 22}

_MIN_TRAIN = 500    # bars before the first forecast is made
_REFIT     = 21     # refit cadence, in bars
_EPS       = 1e-8

# Rolling rather than expanding by default. BTC realized vol averaged ~66 over
# 2015-2020 and ~53 over 2021-2026; an expanding window keeps the high-vol early
# years in the training set forever and forecasts the later sample persistently
# high. A ~4-year window tracks the level while still spanning several regimes.
_TRAIN_WINDOW = 1000

# Below this many bars the forward RV target is one or two squared returns rather
# than an average of many, and is too noisy to forecast from daily closes at all.
# Genuine daily-frequency HAR-RV uses intraday (typically 5-minute) returns to
# build the RV proxy; with daily closes the shortest honest horizon is about a week.
MIN_HONEST_H = 5


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def realized_vol(ret: pd.Series, window: int) -> pd.Series:
    """
    Annualized realized volatility in percent over a trailing window.

    Uses the uncentred second moment (sqrt of mean squared return) rather than a
    sample standard deviation: at daily frequency the mean return is negligible
    against its dispersion, and not subtracting it keeps the estimator additive in
    the way the realized-variance literature assumes.
    """
    return np.sqrt((ret ** 2).rolling(window).mean() * ANNUALIZE) * 100.0


def forward_realized_vol(ret: pd.Series, h: int) -> pd.Series:
    """
    Annualized RV over the *next* h bars, aligned to the decision date t.

    Value at t covers returns t+1 .. t+h, so it is unknown at t — that is the
    quantity being forecast. The final h rows are NaN.
    """
    return realized_vol(ret, h).shift(-h)


def har_features(ret: pd.Series) -> pd.DataFrame:
    """The daily / weekly / monthly RV cascade, all known at t."""
    return pd.DataFrame({name: realized_vol(ret, w) for name, w in HAR_LAGS.items()})


# ── losses ────────────────────────────────────────────────────────────────────

def qlike(actual_vol: pd.Series | np.ndarray, pred_vol: pd.Series | np.ndarray) -> float:
    """
    QLIKE loss in variance units: mean of  s/f - log(s/f) - 1,  where s and f are
    actual and forecast *variance*. Zero for a perfect forecast, always positive
    otherwise, and robust to RV being a noisy proxy for latent volatility.
    """
    s = np.asarray(actual_vol, dtype=float) ** 2
    f = np.asarray(pred_vol,   dtype=float) ** 2
    # QLIKE divides by the forecast, so a forecast near zero produces an unbounded
    # loss from a single quiet bar. Floor it at 1 annualized vol point squared —
    # far below any real volatility, but enough to keep one flat day from deciding
    # the comparison. This mainly bites the naive forecaster at very short h.
    f = np.maximum(f, 1.0)
    m = np.isfinite(s) & np.isfinite(f) & (s > _EPS)
    if not m.any():
        return np.nan
    r = s[m] / f[m]
    return float(np.mean(r - np.log(r) - 1.0))


def r2_oos(actual: np.ndarray, pred: np.ndarray) -> float:
    """Out-of-sample R^2 against the mean of the evaluation sample."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    m = np.isfinite(a) & np.isfinite(p)
    if m.sum() < 2:
        return np.nan
    ss_res = np.sum((a[m] - p[m]) ** 2)
    ss_tot = np.sum((a[m] - a[m].mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept prepended; returns coefficients."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _apply(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


# ── walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(
    ret:          pd.Series,
    h:            int,
    iv:           pd.Series | None = None,
    min_train:    int = _MIN_TRAIN,
    refit:        int = _REFIT,
    train_window: int | None = _TRAIN_WINDOW,
) -> pd.DataFrame:
    """
    Walk-forward forecasts of h-bar-ahead realized volatility, one row per
    evaluation date.

    The training window ends h bars before the forecast date. Without that embargo
    the last h training labels would overlap the very window being predicted, which
    inflates every score — the same overlap problem the h-bar outcomes have in the
    conditional-winrate engine.

    train_window caps how far back training reaches (None for an expanding window).
    Rolling is the default because volatility levels shift across regimes and a
    model anchored to 2015 forecasts 2025 too high.

    Returns columns: actual, har, rv_naive, and iv where implied vol is available.
    """
    feats  = har_features(ret)
    target = forward_realized_vol(ret, h)
    naive  = realized_vol(ret, h)          # today's trailing RV over the same span

    frame = pd.concat([feats, target.rename('actual'), naive.rename('rv_naive')], axis=1)
    if iv is not None:
        frame['iv'] = iv.reindex(frame.index)
    frame = frame.dropna(subset=list(HAR_LAGS) + ['actual', 'rv_naive'])

    X_all = np.log(frame[list(HAR_LAGS)].to_numpy() + _EPS)
    y_all = np.log(frame['actual'].to_numpy() + _EPS)

    preds: dict[int, float] = {}
    beta = None
    for i in range(min_train, len(frame)):
        if (i - min_train) % refit == 0:
            tr = i - h                      # embargo: drop labels overlapping the forecast
            if tr <= 50:
                continue
            lo = max(0, tr - train_window) if train_window else 0
            Xt, yt = X_all[lo:tr], y_all[lo:tr]
            beta = _ols(Xt, yt)
            # Jensen correction: E[exp(z)] > exp(E[z]), so the median forecast that
            # exp(x'b) gives would sit below the mean. Add half the residual variance.
            resid = yt - _apply(beta, Xt)
            beta = (beta, float(np.var(resid)) / 2.0)
        if beta is None:
            continue
        b, half_var = beta
        preds[i] = float(np.exp(_apply(b, X_all[i:i + 1])[0] + half_var) - _EPS)

    if not preds:
        return pd.DataFrame()

    idx = frame.index[list(preds)]
    out = pd.DataFrame({
        'actual':   frame['actual'].iloc[list(preds)].to_numpy(),
        'har':      list(preds.values()),
        'rv_naive': frame['rv_naive'].iloc[list(preds)].to_numpy(),
    }, index=idx)
    if 'iv' in frame.columns:
        out['iv'] = frame['iv'].iloc[list(preds)].to_numpy()
    return out


def score(preds: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """
    Score each forecaster on a common sample.

    Restricting to rows where every model is available matters when IV starts later
    than price history: comparing HAR on 11 years against IV on 5 would compare
    different market regimes, not different models.
    """
    models = models or [c for c in ('har', 'rv_naive', 'iv') if c in preds.columns]
    common = preds.dropna(subset=['actual'] + models)
    if common.empty:
        return pd.DataFrame()

    a = common['actual'].to_numpy()
    rows = {}
    for m in models:
        f = common[m].to_numpy()
        rows[m] = {
            'n':        len(common),
            'qlike':    qlike(a, f),
            'r2_level': r2_oos(a, f),
            'r2_log':   r2_oos(np.log(a + _EPS), np.log(f + _EPS)),
            'rmse':     float(np.sqrt(np.mean((a - f) ** 2))),
            # Positive bias means the forecast sits above realized vol on average.
            # For IV this is the variance risk premium, not a modelling error.
            'bias':     float(np.mean(f - a)),
            'mean_fc':  float(np.mean(f)),
        }
    out = pd.DataFrame(rows).T
    out.index.name = 'model'
    return out


def encompassing(preds: pd.DataFrame) -> dict:
    """
    Does HAR add anything to IV, or IV to HAR?

    Regresses realized vol on both forecasts (in logs). If IV's coefficient absorbs
    everything, the model is redundant to the option market and there is no trade in
    it however good its R^2 looks in isolation.
    """
    need = {'actual', 'har', 'iv'}
    if not need.issubset(preds.columns):
        return {}
    d = preds.dropna(subset=list(need))
    if len(d) < 100:
        return {}
    X = np.log(d[['har', 'iv']].to_numpy() + _EPS)
    y = np.log(d['actual'].to_numpy() + _EPS)
    beta = _ols(X, y)
    return {
        'n':            int(len(d)),
        'const':        float(beta[0]),
        'beta_har':     float(beta[1]),
        'beta_iv':      float(beta[2]),
        'r2':           r2_oos(y, _apply(beta, X)),
        'r2_har_only':  r2_oos(y, _apply(_ols(X[:, [0]], y), X[:, [0]])),
        'r2_iv_only':   r2_oos(y, _apply(_ols(X[:, [1]], y), X[:, [1]])),
    }
