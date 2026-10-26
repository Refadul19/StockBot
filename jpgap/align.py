"""Pairing each Tokyo day with the US session that ended just before its open.

This is the part most likely to hide a bug, so the rules are written out here.

Time facts (all by calendar date, so daylight saving time never enters the pairing):
  * A US session dated D closes at 16:00 New York time, which is D+1 at 05:00 JST (summer
    time) or 06:00 JST (winter time). Either way it is before Tokyo opens at 09:00 JST.
  * So a Tokyo day T can use any US session dated T-1 or earlier, and never one dated T
    (that session closes on T+1 Japan time, after Tokyo has closed).

The gap on Tokyo day T covers everything from the previous Tokyo close (day P, 15:00 JST)
to the open on T. The US sessions that closed inside that window are those dated from P
up to T-1. We keep the day only if exactly ONE US session closed in the window:
  * zero  -> the US was closed (holiday), there is no fresh overnight signal;
  * two+  -> Tokyo was closed on a weekday in between, so the gap mixes several US sessions
             and cannot be matched to one SOX move.
Normal case: Tue to Fri pair with the US session of the day before, Monday pairs with Friday.

The signal is close(that session) / close(the US session before it) - 1.
"""
from __future__ import annotations

import bisect
import datetime as dt
import logging
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger("jpgap")
NY = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")
US_CLOSE = dt.time(16, 0)    # regular close; early closes are earlier, so this is the cautious choice
TOKYO_OPEN = dt.time(9, 0)

PAIR_COLS = ["us_date", "us_prev_date", "signal", "gap", "intraday", "next_day"]


def us_close_utc(d: dt.date) -> dt.datetime:
    return dt.datetime.combine(d, US_CLOSE, tzinfo=NY).astimezone(dt.timezone.utc)


def tokyo_open_utc(d: dt.date) -> dt.datetime:
    return dt.datetime.combine(d, TOKYO_OPEN, tzinfo=TOKYO).astimezone(dt.timezone.utc)


def weekday_range(start: dt.date, end: dt.date) -> list[dt.date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def log_closed_days(jp_days: list[dt.date], us_days: list[dt.date]) -> list[dict]:
    """Weekdays on which one market had no bar. Informational: these days cannot be used."""
    if not jp_days or not us_days:
        return []
    out = []
    lo, hi = max(jp_days[0], us_days[0]), min(jp_days[-1], us_days[-1])
    jp, us = set(jp_days), set(us_days)
    for d in weekday_range(lo, hi):
        if d not in jp:
            out.append({"date": d, "market": "tokyo", "reason": "closed"})
        if d not in us:
            out.append({"date": d, "market": "us", "reason": "closed"})
    return out


def build_pairs(jp: pd.DataFrame, us_close: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """One row per usable Tokyo day.

    jp:        daily frame of one Japanese stock (index = date, needs open and close).
    us_close:  close of the US signal index (index = date).
    Returns (pairs indexed by Tokyo date, skipped) where skipped lists {date, reason} dicts.
    """
    jp = jp.sort_index()
    us_close = us_close.dropna().sort_index()
    us_days = list(us_close.index)
    jp_days = list(jp.index)
    rows: dict[dt.date, dict] = {}
    skipped: list[dict] = []
    one = dt.timedelta(days=1)

    for i, t in enumerate(jp_days):
        if i == 0:
            continue     # no previous Tokyo close yet
        p = jp_days[i - 1]
        lo = bisect.bisect_left(us_days, p)         # first US session dated >= P
        hi = bisect.bisect_right(us_days, t - one)  # one past the last US session dated <= T-1
        n = hi - lo
        if n == 0:
            skipped.append({"date": t, "reason": "no_us_session_in_window"})
            continue
        if n > 1:
            skipped.append({"date": t, "reason": "multiple_us_sessions_in_window"})
            continue
        j = hi - 1
        if j < 1:
            skipped.append({"date": t, "reason": "no_earlier_us_session"})
            continue
        u1, u0 = us_days[j], us_days[j - 1]
        # Safety net: the signal must be fully known before the Tokyo open.
        if not us_close_utc(u1) < tokyo_open_utc(t):
            raise AssertionError(f"look-ahead: US session {u1} closes after Tokyo open on {t}")
        o, c = jp["open"].iloc[i], jp["close"].iloc[i]
        pc = jp["close"].iloc[i - 1]
        if any(pd.isna(x) for x in (o, c, pc)):
            skipped.append({"date": t, "reason": "missing_price"})
            continue
        nxt = jp["close"].iloc[i + 1] / c - 1.0 if i + 1 < len(jp_days) else np.nan
        rows[t] = {
            "us_date": u1, "us_prev_date": u0,
            "signal": us_close.iloc[j] / us_close.iloc[j - 1] - 1.0,
            "gap": o / pc - 1.0,
            "intraday": c / o - 1.0,
            "next_day": nxt,   # close T to close T+1: an outcome, never used to decide anything
            "open": o, "close": c,
        }
    pairs = pd.DataFrame.from_dict(rows, orient="index")
    if pairs.empty:
        pairs = pd.DataFrame(columns=PAIR_COLS + ["open", "close"])
    return pairs, skipped


def add_context(pairs: pd.DataFrame, series: dict[str, pd.Series]) -> pd.DataFrame:
    """Add each context series' overnight return, measured on the same US date as the signal.

    The return on date u1 is close(u1) / its own previous close - 1. NaN when that series has
    no bar on u1 (for example USD/JPY has no bar on some US holidays).
    """
    out = pairs.copy()
    for name, s in series.items():
        ret = s.dropna().sort_index().pct_change()
        out[name] = [ret.get(u, np.nan) for u in out["us_date"]]
    return out
