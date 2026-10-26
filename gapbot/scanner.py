"""Daily gap scan. Uses only data known at the 09:30 open: prior daily bars and today's open."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .config import ScannerConfig

COLUMNS = ["symbol", "open", "prev_close", "gap_pct", "avg_volume", "avg_dollar_volume", "daily_atr"]


def scan_gappers(daily: pd.DataFrame, day: dt.date, cfg: ScannerConfig) -> pd.DataFrame:
    """Return today's gap-up candidates, best first.

    Columns: symbol, open, prev_close, gap_pct, avg_volume, avg_dollar_volume, daily_atr
    """
    hist = daily[daily["date"] < day].sort_values(["symbol", "date"])
    today = daily[daily["date"] == day][["symbol", "open"]]
    if hist.empty or today.empty:
        return _empty()

    last20 = hist.groupby("symbol", group_keys=False).tail(20).copy()
    last20["dollar_volume"] = last20["close"] * last20["volume"]
    stats = last20.groupby("symbol").agg(
        avg_volume=("volume", "mean"),
        avg_dollar_volume=("dollar_volume", "mean"),
        n_days=("volume", "size"),
    )
    prev_close = hist.groupby("symbol")["close"].last().rename("prev_close")

    # 14-day average true range from daily bars (simple mean), used to floor intraday stops
    pc = last20.groupby("symbol")["close"].shift(1)
    last20["tr"] = pd.concat([last20["high"] - last20["low"], (last20["high"] - pc).abs(),
                              (last20["low"] - pc).abs()], axis=1).max(axis=1)
    daily_atr = last20.groupby("symbol")["tr"].apply(lambda s: s.tail(14).mean()).rename("daily_atr")

    df = today.set_index("symbol").join(prev_close, how="inner").join(stats, how="inner").join(daily_atr, how="left")
    df = df[df["n_days"] >= 10]                       # need some history to judge liquidity
    df["gap_pct"] = (df["open"] / df["prev_close"] - 1.0) * 100.0

    df = df[
        (df["open"] >= cfg.min_price) & (df["open"] <= cfg.max_price)
        & (df["gap_pct"] >= cfg.min_gap_pct)
        & (df["avg_dollar_volume"] >= cfg.min_avg_dollar_volume)
        & (df["avg_volume"] >= cfg.min_avg_volume)
    ]
    df = df.sort_values("gap_pct", ascending=False).head(cfg.max_candidates)
    return df.reset_index()[COLUMNS]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)
