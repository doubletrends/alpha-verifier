"""
Outcome registry — the binary event whose conditional probability the matrix measures.

The engine measures  P(outcome | feature condition) − P(outcome).  The outcome was
historically fixed to "price rose over the next h bars"; it is pluggable here so the
same conditional-counting machinery can be aimed at risk events, where the empirical
structure survives a null test that directional structure does not.

An outcome fn has signature  (data: pd.DataFrame, h: int, params: dict) -> pd.Series
of 1.0 / 0.0 / NaN aligned to data.index.  NaN marks bars whose outcome is not yet
realized (the trailing h bars); those rows are dropped from every count, exactly as
the directional outcome always dropped them.

Registration mirrors data/features.py, so a workspace plugin can add its own outcome
at import time with no root edits.
"""

from typing import Callable

import numpy as np
import pandas as pd

_registry: dict[str, dict] = {}


def register(
    name:  str,
    fn:    Callable,
    event: str,
    title: str,
    expr:  str,
) -> None:
    """
    Register an outcome.

      name  — key used in universe.json / --outcome
      fn    — (data, h, params) -> Series of 1.0/0.0/NaN
      event — very short symbol for sheet names, e.g. 'up', 'dd10'  (Excel caps
              sheet names at 31 chars, so keep this under ~6 characters)
      title — human title for the xlsx header block, e.g. 'Winrate'
      expr  — formula-style description of the event for the header block
    """
    _registry[name] = {'fn': fn, 'event': event, 'title': title, 'expr': expr}


def get(name: str) -> dict:
    if name not in _registry:
        raise KeyError(f"unknown outcome '{name}' (registered: {', '.join(sorted(_registry))})")
    return _registry[name]


def available() -> list[str]:
    return sorted(_registry)


def compute(data: pd.DataFrame, name: str, h: int, params: dict | None = None) -> pd.Series:
    """Evaluate a registered outcome at horizon h. Returns 1.0/0.0/NaN on data.index."""
    return get(name)['fn'](data, h, params or {})


def event_label(name: str, params: dict | None = None) -> str:
    """
    Short symbol for the event, used in xlsx sheet names and column headers.

    Threshold-parameterised outcomes fold the threshold into the label so that
    drawdown at 10% and at 20% do not produce identically-named sheets.
    """
    spec  = get(name)
    ev    = spec['event']
    thr   = (params or {}).get('threshold')
    if thr is not None:
        ev = f'{ev}{abs(float(thr)) * 100:g}'
    return ev


def describe(name: str, params: dict | None = None) -> str:
    """One-line description of the event with its parameters substituted in."""
    spec = get(name)
    expr = spec['expr']
    for k, v in (params or {}).items():
        expr = expr.replace(k, f'{v:g}' if isinstance(v, (int, float)) else str(v))
    return expr


# ── built-in outcomes ─────────────────────────────────────────────────────────

def _forward_min(close: pd.Series, h: int) -> pd.Series:
    """min(close[t+1] ... close[t+h]), NaN where the full window is not realized."""
    fwd = pd.concat([close.shift(-k) for k in range(1, h + 1)], axis=1)
    return fwd.min(axis=1).where(close.shift(-h).notna())


def _forward_max(close: pd.Series, h: int) -> pd.Series:
    fwd = pd.concat([close.shift(-k) for k in range(1, h + 1)], axis=1)
    return fwd.max(axis=1).where(close.shift(-h).notna())


def _up(data: pd.DataFrame, h: int, params: dict) -> pd.Series:
    """Price rose over the next h bars. The original outcome; still the default."""
    close   = data['close']
    shifted = close.shift(-h)
    return (shifted > close).where(shifted.notna()).astype(float)


def _drawdown(data: pd.DataFrame, h: int, params: dict) -> pd.Series:
    """
    Price fell more than `threshold` below its starting value at some point within
    the next h bars — i.e. a peak-to-trough loss from t, not just a lower close at t+h.
    """
    close = data['close']
    thr   = float(params.get('threshold', 0.10))
    trough = _forward_min(close, h) / close - 1.0
    return (trough < -abs(thr)).where(trough.notna()).astype(float)


def _runup(data: pd.DataFrame, h: int, params: dict) -> pd.Series:
    """Mirror of _drawdown: price gained more than `threshold` at some point within h bars."""
    close = data['close']
    thr   = float(params.get('threshold', 0.10))
    peak  = _forward_max(close, h) / close - 1.0
    return (peak > abs(thr)).where(peak.notna()).astype(float)


def _vol_high(data: pd.DataFrame, h: int, params: dict) -> pd.Series:
    """
    Realized volatility over the next h bars exceeded its own trailing median.

    The reference median is computed from *backward*-looking h-bar volatility only,
    so the comparison uses nothing that was unavailable at time t.
    """
    close    = data['close']
    lookback = int(params.get('lookback', 252))
    ret      = np.log(close).diff()
    rv_back  = ret.rolling(h).std()
    ref      = rv_back.rolling(lookback).median()
    rv_fwd   = rv_back.shift(-h)
    valid    = rv_fwd.notna() & ref.notna()
    return (rv_fwd > ref).where(valid).astype(float)


register('up',        _up,
         event='up',    title='Winrate',
         expr='price_h0+h > price_h0')

register('drawdown',  _drawdown,
         event='dd',    title='Drawdown Risk',
         expr='min(price_h0+1..h) / price_h0 - 1 < -threshold')

register('runup',     _runup,
         event='run',   title='Runup Chance',
         expr='max(price_h0+1..h) / price_h0 - 1 > +threshold')

register('vol_high',  _vol_high,
         event='volhi', title='Volatility Regime',
         expr='realized_vol(h0..h0+h) > trailing median realized_vol')
