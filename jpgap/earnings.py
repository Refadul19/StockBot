"""Earnings days: marking the days a stock moves on its own news.

Sources, in order of trust:
  1. a manual CSV (config/jp_earnings.csv): ticker, date, time (optional), note (ignored)
  2. yfinance Ticker.get_earnings_dates (unofficial, often incomplete for Japan)
A manual date replaces any yfinance date within a few days of it for the same ticker.

Which days get marked, per report (the stock's own trading calendar is used, so weekends and
holidays are skipped correctly):
  * after the close, or time unknown (the default assumption): the report day AND the next
    trading day, because the price reaction appears at the next open;
  * during market hours: the report day only.
"""
from __future__ import annotations

import bisect
import datetime as dt
import logging
import time as _time
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import EarningsConfig
from .data import safe_name

log = logging.getLogger("jpgap")
AFTER, DURING = "after_close", "during_hours"
_AFTER_WORDS = {"after_close", "after-close", "afterclose", "amc", "post_close"}
_DURING_WORDS = {"during_hours", "market_hours", "intraday", "before_open", "pre_market", "bmo"}


@dataclass(frozen=True)
class Event:
    ticker: str
    date: dt.date          # Japan date of the report
    time: str              # "" = unknown, else HH:MM or a label such as after_close
    source: str            # "manual" or "yfinance"


def _minutes(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


def classify(time_str: str, cfg: EarningsConfig) -> str:
    """AFTER or DURING for one report time. Unknown or unreadable times use cfg.unknown_time_as."""
    t = (time_str or "").strip().lower().replace(" ", "_")
    if t in _AFTER_WORDS:
        return AFTER
    if t in _DURING_WORDS:
        return DURING
    try:
        return AFTER if _minutes(t) >= _minutes(cfg.after_close_from) else DURING
    except (ValueError, AttributeError):
        return cfg.unknown_time_as


def mark_days(events: list[Event], trading_days: list[dt.date], cfg: EarningsConfig) -> set[dt.date]:
    """The trading days on which an earnings reaction may be mixed into the price."""
    days = sorted(trading_days)
    out: set[dt.date] = set()
    for e in events:
        i = bisect.bisect_left(days, e.date)
        on_day = i < len(days) and days[i] == e.date
        j = bisect.bisect_right(days, e.date)          # first trading day after the report date
        nxt = days[j] if j < len(days) else None
        if classify(e.time, cfg) == AFTER:
            if cfg.mark_announcement_day and on_day:
                out.add(e.date)
            if cfg.mark_next_day_after_close and nxt is not None:
                out.add(nxt)
        else:
            first = e.date if on_day else nxt          # a report on a closed day reacts at the next open
            if first is not None:
                out.add(first)
    return out


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def load_manual(path: str | Path) -> list[Event]:
    p = Path(path)
    if not p.exists():
        log.info("No manual earnings file at %s", p)
        return []
    df = pd.read_csv(p, comment="#", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"ticker", "date"} <= set(df.columns):
        raise ValueError(f"{p} needs the columns ticker,date (time and note are optional)")
    out = []
    for r in df.to_dict("records"):
        try:
            d = dt.date.fromisoformat(r["date"].strip())
        except ValueError:
            log.warning("%s: ignoring row with a bad date %r (use YYYY-MM-DD)", p, r["date"])
            continue
        out.append(Event(r["ticker"].strip().upper(), d, r.get("time", "").strip(), "manual"))
    return out


def _fetch_yf(ticker: str, cfg: EarningsConfig) -> pd.DataFrame:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.Ticker(ticker).get_earnings_dates(limit=cfg.yfinance_limit)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "time"])
    idx = pd.DatetimeIndex(raw.index)
    idx = idx.tz_convert("Asia/Tokyo") if idx.tz is not None else idx
    return pd.DataFrame({"date": [d.date().isoformat() for d in idx], "time": [d.strftime("%H:%M") for d in idx]})


def load_yfinance(ticker: str, cache_dir: str, cfg: EarningsConfig, refresh: bool = False,
                  offline: bool = False) -> list[Event]:
    """Report dates from yfinance, cached on disk. Dates are converted to Japan time."""
    path = Path(cache_dir) / f"earnings_{safe_name(ticker)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.exists() and (_time.time() - path.stat().st_mtime) < cfg.cache_max_age_days * 86400
    if path.exists() and (offline or (fresh and not refresh)):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif offline:
        return []
    else:
        try:
            df = _fetch_yf(ticker, cfg)
            df.to_csv(path, index=False)
        except Exception as exc:
            if not path.exists():
                log.warning("%s: yfinance earnings dates unavailable (%s)", ticker, exc)
                return []
            log.warning("%s: earnings download failed (%s); using the older cache", ticker, exc)
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return [Event(ticker, dt.date.fromisoformat(r["date"]), r["time"] if cfg.trust_yfinance_times else "", "yfinance")
            for r in df.to_dict("records")]


def merge_events(manual: list[Event], yf_events: list[Event], cfg: EarningsConfig) -> list[Event]:
    """Manual entries win: a yfinance date within override_window_days of a manual date for the
    same ticker is dropped, so a corrected date is not counted twice."""
    window = dt.timedelta(days=cfg.override_window_days)
    kept = [e for e in yf_events
            if not any(m.ticker == e.ticker and abs(m.date - e.date) <= window for m in manual)]
    return sorted(manual + kept, key=lambda e: (e.ticker, e.date))


def coverage(events: list[Event], first_day: dt.date, last_day: dt.date, max_gap_days: int) -> tuple[str, str]:
    """(status, detail) for one ticker. Companies report about four times a year, so a long
    silence means missing dates. Dates after last_day are upcoming and ignored here."""
    past = sorted({e.date for e in events if e.date <= last_day})
    if not past:
        return "none", "no report dates found; earnings days are NOT excluded for this stock"
    limit = dt.timedelta(days=max_gap_days)
    chain = [first_day] + [d for d in past if d >= first_day]
    gaps = [f"{a} to {b}" for a, b in zip(chain[:-1], chain[1:]) if b - a > limit]
    if not any(d >= first_day for d in past):
        gaps = [f"{first_day} to {last_day}"]
    elif last_day - past[-1] > limit and past[-1] >= first_day:
        gaps.append(f"{past[-1]} to {last_day}")
    if gaps:
        return "uncertain", f"{len(gaps)} stretch(es) longer than {max_gap_days} days without a report date: " + "; ".join(gaps[:4])
    return "ok", f"{len(past)} report dates"
