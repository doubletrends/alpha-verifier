import json
import urllib.parse
import urllib.request

import pandas as pd
import yfinance as yf

_REGISTRY = {}


def _register(name):
    def dec(fn):
        _REGISTRY[name] = fn
        return fn
    return dec


def _yfinance_ohlcv(ticker: str, interval: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, interval=interval, auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = 'Date'
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)


@_register('ohlcv')
def _fetch_ohlcv(start: str) -> pd.DataFrame:
    return _yfinance_ohlcv('BTC-USD', '1d', start)


@_register('dxy')
def _fetch_dxy(start: str) -> pd.DataFrame:
    df = yf.download('DX-Y.NYB', start=start, interval='1d', auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = 'Date'
    return df[['Close']].rename(columns={'Close': 'dxy'}).ffill()


@_register('coinmetrics')
def _fetch_coinmetrics(start: str) -> pd.DataFrame:
    params = {
        'assets':    'btc',
        'metrics':   'CapMVRVCur,HashRate,AdrActCnt,TxCnt',
        'frequency': '1d',
        'format':    'json',
        'page_size': '10000',
    }
    url = (
        'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?'
        + urllib.parse.urlencode(params)
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.load(r)
    df = pd.DataFrame(payload['data'])
    df['Date'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    df = df.set_index('Date').sort_index()
    rename = {
        'CapMVRVCur': 'mvrv',
        'HashRate':   'hash_rate',
        'AdrActCnt':  'adr_act_cnt',
        'TxCnt':      'tx_cnt',
    }
    df = df[list(rename)].rename(columns=rename).apply(pd.to_numeric, errors='coerce')
    return df[df.index >= pd.Timestamp(start)]


def fetch(sources: list[str], start: str = '2015-01-01', asset: dict | None = None) -> pd.DataFrame:
    handlers = dict(_REGISTRY)
    if asset is not None and asset.get('provider') == 'yfinance':
        ticker, interval = asset['ticker'], asset['interval']
        handlers['ohlcv'] = lambda start: _yfinance_ohlcv(ticker, interval, start)
    frames = [handlers[s](start=start) for s in sources]
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how='left')
    return out
