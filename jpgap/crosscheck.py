"""Is the SOX data believable? Three checks, each reported plainly.

1. Calendar: if SOX is missing a session that the S&P 500 has, the 'one-day' move on the next
   day really spans two sessions and looks too big.
2. Sanity: the close must lie between the day's low and high.
3. Second source: the biggest SOX days are compared with US chip ETFs (default SOXX and SMH).
   These are different baskets, not the same index, so a match within a couple of percentage
   points is a good sign and a miss is a reason to look closer, not proof of an error.
   (A truly independent copy of the SOX index, such as Stooq, blocks scripted downloads.)
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .config import CrossCheckConfig


def missing_sessions(sox_dates: list[dt.date], ref_dates: list[dt.date]) -> list[dt.date]:
    """Sessions the reference index has inside SOX's date range but SOX lacks."""
    if not sox_dates:
        return []
    lo, hi = min(sox_dates), max(sox_dates)
    have = set(sox_dates)
    return sorted(d for d in ref_dates if lo <= d <= hi and d not in have)


def bad_ohlc(df: pd.DataFrame) -> list[dt.date]:
    """Days where the close is outside the low to high range (a sign of a bad row)."""
    if not {"high", "low", "close"} <= set(df.columns):
        return []
    tol = 1e-6
    bad = (df["close"] > df["high"] * (1 + tol)) | (df["close"] < df["low"] * (1 - tol))
    return list(df.index[bad])


def compare_with_proxy(sox_close: pd.Series, proxy_close: pd.Series, cfg: CrossCheckConfig) -> pd.DataFrame:
    """For each big SOX day, the proxy's move on the same date and whether they match."""
    sox_ret = sox_close.dropna().sort_index().pct_change() * 100
    prx_ret = proxy_close.dropna().sort_index().pct_change() * 100
    big = sox_ret[sox_ret.abs() > cfg.min_sox_move_pct].dropna()
    rows = []
    for d, r in big.items():
        p = prx_ret.get(d, float("nan"))
        diff = p - r
        rows.append({"date": d, "sox_pct": r, "proxy_pct": p, "diff_pp": diff,
                     "match": bool(pd.notna(p) and (p > 0) == (r > 0) and abs(diff) <= cfg.tolerance_pp)})
    return pd.DataFrame(rows, columns=["date", "sox_pct", "proxy_pct", "diff_pp", "match"])
