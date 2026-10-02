"""Price loading with point-in-time discipline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_TICKERS = (
    "AAPL", "AMZN", "GOOGL", "META", "NFLX", "MSFT", "NVDA", "JPM",
    "XOM", "JNJ", "WMT", "PG", "KO", "DIS", "BA", "CAT", "GE", "T",
    "PFE", "INTC",
)


def download(
    tickers=DEFAULT_TICKERS,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    cache: str | Path = "data/prices.pkl",
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV and cache it.

    The original study used 5 tickers over ~103 trading days -- roughly 500
    observations, which cannot support a directional-accuracy claim at any
    confidence. The default here is 20 tickers over 15 years (~75,000
    observations) so that per-fold test windows are large enough for the
    confidence intervals in `metrics` to be narrow enough to mean anything.
    """
    import yfinance as yf

    cache = Path(cache)
    if cache.exists():
        raw = pd.read_pickle(cache)
    else:
        raw = yf.download(
            list(tickers), start=start, end=end,
            auto_adjust=True, progress=False, group_by="ticker",
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_pickle(cache)

    out = {}
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        df = raw[t].dropna()
        if len(df) < 500:
            continue
        out[t] = df[["Open", "High", "Low", "Close", "Volume"]]
    if not out:
        raise RuntimeError("no usable tickers downloaded")
    return out


def load_legacy_csv(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read the original project's wide two-row-header CSV.

    Kept so the rebuilt pipeline can be run against the original data and the
    old numbers reproduced under correct validation -- the comparison is the
    point, not the data.
    """
    raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    fields = raw.columns.get_level_values(0).unique()
    tickers = raw.columns.get_level_values(1).unique()
    out = {}
    for t in tickers:
        cols = {f: raw[(f, t)] for f in fields if (f, t) in raw.columns}
        df = pd.DataFrame(cols).dropna()
        if len(df) >= 30:
            out[t] = df
    return out
