"""
Shuffle-null validation for node surfaces.

A node's headline number is the peak absolute deviation over ~30 threshold bins.
That statistic is a maximum over many noisy estimates, so it is biased upward even
when the feature carries no information: with n ≈ 50 per bin the standard error of
a rate is ~7pp, and the max over 30 bins lands near 10pp by construction. Comparing
a node's peak against a fixed threshold therefore cannot separate signal from noise.

This module builds the null distribution of that same statistic directly. The outcome
series is circularly shifted against the feature many times; each shift preserves the
autocorrelation of both series — which matters, because overlapping h-bar outcomes are
strongly serially correlated — while destroying any real alignment between them. The
node's real peak is then read as a quantile of that null.

    p-value = fraction of shifts whose peak deviation matched or exceeded the real one

A node is only evidence of structure if its peak clears the null, not if it clears a
hand-set pp threshold.
"""

import numpy as np
import pandas as pd

DEFAULT_SHIFTS = 200
DEFAULT_MIN_N  = 30
_EDGE_GUARD    = 200   # keep shifts away from near-zero / near-N wraps


def _bin_index(feature: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Assign each observation to a threshold interval (the PDF sheet's slices)."""
    return np.searchsorted(thresholds, feature)


def _peak_deviation(bin_idx: np.ndarray, y: np.ndarray, n_bins: int, min_n: int) -> float:
    """
    Max |P(event | X in bin) − P(event)| over bins with at least min_n observations,
    in percentage points. This is the same quantity the PDF sheet reports, computed
    directly rather than by differencing the CDF.
    """
    counts = np.bincount(bin_idx, minlength=n_bins)
    sums   = np.bincount(bin_idx, weights=y, minlength=n_bins)
    ok     = counts >= min_n
    if not ok.any():
        return 0.0
    rates = sums[ok] / counts[ok]
    return float(np.abs(rates - y.mean()).max() * 100.0)


def null_test(
    feature:    pd.Series,
    outcome:    pd.Series,
    thresholds: np.ndarray,
    min_n:      int = DEFAULT_MIN_N,
    n_shifts:   int = DEFAULT_SHIFTS,
    seed:       int = 0,
) -> dict:
    """
    Compare a node's peak deviation against its circular-shift null.

    Returns {real, null_median, null_p95, p_value, n_obs, n_shifts}. Deviations are
    in percentage points. p_value is the share of shifts reaching the real peak, so
    small is good; with n_shifts=200 the resolution floor is 0.005.
    """
    aligned = pd.concat([feature.rename('x'), outcome.rename('y')], axis=1).dropna()
    if len(aligned) < 2 * _EDGE_GUARD + 1:
        return {'real': np.nan, 'null_median': np.nan, 'null_p95': np.nan,
                'p_value': np.nan, 'n_obs': len(aligned), 'n_shifts': 0}

    x      = aligned['x'].to_numpy(dtype=float)
    y      = aligned['y'].to_numpy(dtype=float)
    n      = len(y)
    n_bins = len(thresholds) + 1
    idx    = _bin_index(x, thresholds)

    real = _peak_deviation(idx, y, n_bins, min_n)

    rng    = np.random.default_rng(seed)
    shifts = rng.integers(_EDGE_GUARD, n - _EDGE_GUARD, size=n_shifts)
    null   = np.array([_peak_deviation(idx, np.roll(y, int(s)), n_bins, min_n) for s in shifts])

    # Add-one (Davison & Hinkley) estimator: a permutation p-value must never be
    # exactly 0, since the observed statistic is itself one draw from the null. The
    # floor is 1/(n_shifts+1), and no p-value below that floor is resolvable -- which
    # is what `verdict` checks before awarding a corrected verdict.
    n_ge = int((null >= real).sum())
    return {
        'real':        round(real, 3),
        'null_median': round(float(np.median(null)), 3),
        'null_p95':    round(float(np.percentile(null, 95)), 3),
        'p_value':     round((1.0 + n_ge) / (1.0 + n_shifts), 5),
        'p_floor':     round(1.0 / (1.0 + n_shifts), 5),
        'n_obs':       int(n),
        'n_shifts':    int(n_shifts),
    }


def verdict(
    p_value:  float,
    alpha:    float = 0.05,
    n_tests:  int = 1,
    p_floor:  float = 0.0,
) -> str:
    """
    Translate a p-value into a verdict, Bonferroni-corrected for the size of the sweep.

      structure    - clears alpha/n_tests; the only claim that survives correction
      nominal      - clears alpha alone; what a single-node view would call an edge
      underpowered - sits at the resolution floor, so it *might* clear the corrected
                     alpha but this many shifts cannot show it. Re-run with more.
      noise        - indistinguishable from the null

    The underpowered case matters: with 200 shifts the smallest resolvable p-value is
    1/201, while a 195-test sweep needs 2.6e-4. Reporting such a node as 'structure'
    would claim a precision the resampling does not have.
    """
    if pd.isna(p_value):
        return 'insufficient'
    alpha_corr = alpha / max(n_tests, 1)
    if p_value <= alpha_corr:
        return 'structure'
    if p_value <= p_floor and p_floor > alpha_corr:
        return 'underpowered'
    if p_value <= alpha:
        return 'nominal'
    return 'noise'


def shifts_needed(alpha: float = 0.05, n_tests: int = 1, margin: int = 10) -> int:
    """Shifts required for the resolution floor to sit below the corrected alpha."""
    return int(np.ceil(margin * max(n_tests, 1) / alpha))
