"""Causal indicators: each value at row i uses only rows <= i.

That property is what lets the engine precompute indicators for a whole day once
and then read the row for 'now' without looking ahead. tests/test_no_lookahead.py
checks it.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample(minute: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute bars into N-minute bars aligned to the clock (09:30, 09:45 ...).

    Index is the bar START time. A bar is complete once start + N minutes has passed;
    the column `end` makes that easy to check.
    """
    if minute.empty:
        out = pd.DataFrame(columns=[*AGG, "end"])
        out.index = minute.index
        return out
    out = minute.resample(f"{minutes}min", label="left", closed="left").agg(AGG).dropna(subset=["open"])
    out["end"] = out.index + pd.Timedelta(minutes=minutes)
    return out


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()   # Wilder smoothing


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    return line - ema(line, signal)


def session_vwap(df: pd.DataFrame, session_start: dt.time = dt.time(9, 30)) -> pd.Series:
    """VWAP from the regular-session open. NaN for premarket rows."""
    in_session = df.index.time >= session_start
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = (typical * df["volume"]).where(in_session, 0.0).cumsum()
    vol = df["volume"].where(in_session, 0.0).cumsum()
    out = pv / vol.replace(0, np.nan)
    return out.where(in_session)


def last_completed(frame: pd.DataFrame, now: pd.Timestamp) -> int:
    """Position of the last bar whose end <= now, or -1 if none."""
    if frame.empty:
        return -1
    return int(frame["end"].searchsorted(now, side="right")) - 1
