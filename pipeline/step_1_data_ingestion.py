"""Stage 1 — data ingestion.

Fetches raw BTC on-chain data (CoinMetrics), OHLCV (yfinance) and macro series
(FRED and DXY), joins them on the CoinMetrics date spine, and writes the
operational data store (ODS).

    Output: config.ODS_CSV

Run:  python pipeline/step_1_data_ingestion.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

from config import ROOT, ODS_CSV, ensure_dirs

_YF_CACHE = Path(os.environ.get('TEMP', str(ROOT))) / 'yfinance-cache'
_YF_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(_YF_CACHE))


def download_coinmetrics() -> pd.DataFrame:
    base_url = 'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'
    params = {
        'assets': 'btc',
        'metrics': 'PriceUSD,HashRate,TxCnt,AdrActCnt,SplyCur,CapMVRVCur',
        'frequency': '1d',
        'format': 'json',
        'page_size': '10000',
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.load(r)

    df = pd.DataFrame(payload['data'])
    df['Date'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    df = df.set_index('Date').sort_index()
    df = df.iloc[365 * 2:]  # skip first 2 years of sparse data
    cols = ['PriceUSD', 'HashRate', 'TxCnt', 'AdrActCnt', 'SplyCur', 'CapMVRVCur']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df[cols].rename(columns={
        'PriceUSD':   'price_usd',
        'HashRate':   'hash_rate',
        'TxCnt':      'tx_cnt',
        'AdrActCnt':  'adr_act_cnt',
        'SplyCur':    'sply_cur',
        'CapMVRVCur': 'mvrv',
    })


def download_ohlcv() -> pd.DataFrame:
    df = yf.download('BTC-USD', start='2010-01-01', interval='1d', auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = 'Date'
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)


def download_dxy() -> pd.Series:
    df = yf.download('DX-Y.NYB', start='2010-01-01', interval='1d', auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = 'Date'
    return df['Close'].rename('dxy')


def download_fred(series_id: str, name: str) -> pd.Series:
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    with urllib.request.urlopen(url, timeout=60) as r:
        frame = pd.read_csv(r)
    frame['observation_date'] = pd.to_datetime(frame['observation_date'])
    values = pd.to_numeric(frame[series_id], errors='coerce')
    return pd.Series(values.to_numpy(), index=frame['observation_date'], name=name)


def build_ods() -> pd.DataFrame:
    print('Fetching CoinMetrics on-chain data...')
    cm = download_coinmetrics()

    print('Fetching yfinance BTC-USD OHLCV...')
    ohlcv = download_ohlcv()

    print('Fetching yfinance DXY...')
    dxy = download_dxy()

    print('Fetching FRED UNRATE...')
    unrate = download_fred('UNRATE', 'unrate')

    print('Fetching FRED CPIAUCSL...')
    cpi = download_fred('CPIAUCSL', 'cpi')

    df = cm.copy()
    df = df.join(ohlcv,  how='left')
    df = df.join(dxy,    how='left')
    df = df.join(unrate, how='left')
    df = df.join(cpi,    how='left')
    df['dxy']    = df['dxy'].ffill()
    df['unrate'] = df['unrate'].ffill()
    df['cpi']    = df['cpi'].ffill()
    return df


if __name__ == '__main__':
    ensure_dirs()
    data = build_ods()
    data.to_csv(ODS_CSV)
    print(f'\nODS saved → {ODS_CSV}')
    print(f'Shape      : {data.shape[0]} rows × {data.shape[1]} cols')
    print(f'Date range : {data.index[0].date()} → {data.index[-1].date()}')
