"""Daily bars from yfinance with a local CSV cache.

Conventions (same spirit as gapbot/data.py):
  * a daily frame is indexed by datetime.date (the exchange's own local date) with
    columns open, high, low, close, volume.
  * prices are split-adjusted but NOT dividend-adjusted (auto_adjust=False), so a gap is
    what a trader would actually have seen on the screen.

yfinance is an unofficial scraper. It can return stale or wrong rows, so bad rows are
dropped and logged here, and the cache lets you re-run without the network.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import time
from pathlib import Path

import pandas as pd

from .config import DataConfig

log = logging.getLogger("jpgap")
COLS = ["open", "high", "low", "close", "volume"]


def safe_name(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", ticker)


def clean_daily(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise a yfinance frame to the daily convention and drop unusable rows."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLS)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df = df[[c for c in COLS if c in df.columns]]
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)   # keep the exchange's local calendar date
    df.index = [d.date() for d in idx]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    n0 = len(df)
    df = df.dropna(subset=["open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    if ticker.endswith(".T") and "volume" in df:
        # a zero-volume row on a Tokyo stock is a placeholder for a day it did not trade
        df = df[df["volume"] > 0]
    if len(df) < n0:
        log.info("%s: dropped %d unusable rows", ticker, n0 - len(df))
    return df


def drop_glitches(df: pd.DataFrame, ticker: str, ratio: float, window: int) -> pd.DataFrame:
    """Drop rows whose open or close is wildly off the rolling median close. Yahoo sometimes
    returns a few days at 1/10 of the real price (a split applied and then undone). Real stocks
    do not move 3x inside a couple of weeks, so such rows are data errors, not trades."""
    if ratio <= 1 or len(df) < 3:
        return df
    med = df["close"].rolling(window, center=True, min_periods=3).median()
    bad = pd.Series(False, index=df.index)
    for col in ("open", "close"):
        r = df[col] / med
        bad |= (r > ratio) | (r < 1.0 / ratio)
    if bad.any():
        log.warning("%s: dropped %d row(s) with a price more than %gx away from the rolling median (bad data): %s",
                    ticker, int(bad.sum()), ratio, [d.isoformat() for d in df.index[bad]])
    return df[~bad]


def _fetch(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf   # imported lazily so tests do not need the network package

    raw = yf.Ticker(ticker).history(start=start.isoformat(), end=(end + dt.timedelta(days=1)).isoformat(),
                                    auto_adjust=False, actions=False)
    return clean_daily(raw, ticker)


def load_daily(ticker: str, start: dt.date, end: dt.date, cfg: DataConfig,
               refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    """Cached daily bars for one ticker between start and end (inclusive)."""
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{safe_name(ticker)}_{start.isoformat()}.csv"
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < cfg.cache_max_age_days * 86400
    df = None
    if path.exists() and (offline or (fresh and not refresh)):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = [d.date() for d in df.index]
    elif offline:
        raise RuntimeError(f"No cached data for {ticker} and --offline was given")
    else:
        try:
            df = _fetch(ticker, start, end)
            time.sleep(cfg.request_pause_s)
        except Exception as exc:   # network blocked, yfinance changed, etc.
            if not path.exists():
                raise RuntimeError(f"Download failed for {ticker}: {exc}") from exc
            log.warning("%s: download failed (%s); using the older cache", ticker, exc)
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = [d.date() for d in df.index]
        else:
            if df.empty:
                raise RuntimeError(f"yfinance returned no data for {ticker}")
            df.to_csv(path)
    df = drop_glitches(df, ticker, cfg.glitch_ratio, cfg.glitch_window)
    return df[(df.index >= start) & (df.index <= end)]
