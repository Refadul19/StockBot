"""Reference levels for one Japanese stock, with an honest track record.

This is NOT a prediction engine. The overnight US move explains part of the opening gap (the
research in this repo shows that), so the expected open is a fair estimate. What happens after
the open is close to a coin flip, so the target and stop are simply how far this stock has
usually moved from the open on similar days. The same levels are then replayed on the stock's
own history, with costs, so the visitor sees whether they would have worked, in-sample and on
held-out days. Long only, like the rest of the project.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

from gapbot.metrics import split_days

from .align import add_context, build_pairs
from .analysis import assign_bucket, bucket_labels
from .config import JpConfig
from .costs import net_return
from .earnings import Event, mark_days

CODE = re.compile(r"^[0-9][0-9A-Z]{3}$")


def parse_ticker(text: str) -> str:
    """'6857', '6857.t' and ' 285a.T ' all become a Tokyo ticker; anything else is refused."""
    t = (text or "").strip().upper()
    if t.endswith(".T"):
        t = t[:-2]
    if not CODE.match(t):
        raise ValueError("Enter a Tokyo stock code such as 6857 or 6857.T (4 characters, e.g. 285A).")
    return f"{t}.T"


def latest_signal(sox_close: pd.Series) -> dict:
    c = sox_close.dropna().sort_index()
    if len(c) < 2:
        raise ValueError("Not enough SOX data")
    return {"date": c.index[-1], "prev_date": c.index[-2], "move": float(c.iloc[-1] / c.iloc[-2] - 1.0)}


def next_weekday(d: dt.date) -> dt.date:
    n = d + dt.timedelta(days=1)
    while n.weekday() >= 5:
        n += dt.timedelta(days=1)
    return n


def _levels(p: pd.DataFrame, cfg: JpConfig) -> dict:
    """Target, stop and typical close, as fractions of the open, from the given days."""
    up = (p["high"] / p["open"] - 1.0).clip(lower=0)
    down = (1.0 - p["low"] / p["open"]).clip(lower=0)
    return {"n": len(p), "target": float(np.percentile(up, cfg.plan.target_percentile)),
            "stop": float(np.percentile(down, cfg.plan.stop_percentile)),
            "median_close": float((p["close"] / p["open"] - 1.0).median())}


def levels_by_bucket(p_is: pd.DataFrame, cfg: JpConfig) -> dict[str, dict]:
    """Per SOX bucket from in-sample days; a bucket with too few days falls back to all days."""
    allday = _levels(p_is, cfg)
    b = assign_bucket(p_is["signal"], cfg.analysis.bucket_edges_pct)
    out = {}
    for lab in bucket_labels(cfg.analysis.bucket_edges_pct):
        sub = p_is[b == lab]
        ok = len(sub) >= cfg.analysis.min_bucket_days
        out[lab] = {**(_levels(sub, cfg) if ok else allday), "used": "this SOX bucket" if ok else "all days (bucket too small)"}
    return out


def simulate(p: pd.DataFrame, levels: dict[str, dict], cost_pct: float, cfg: JpConfig) -> pd.DataFrame:
    """Buy the open, stop and target set as fractions of the open, otherwise sell the close.
    Daily bars cannot tell which came first, so when both were touched the stop is assumed to
    have hit first (pessimistic, like the rest of the project)."""
    if p.empty:
        return pd.DataFrame(columns=["date", "bucket", "exit", "ret_net"])
    labs = assign_bucket(p["signal"], cfg.analysis.bucket_edges_pct).astype(str)
    rows = []
    for (d, r), lab in zip(p.iterrows(), labs):
        lv = levels[lab]
        stop_px, tgt_px = r["open"] * (1 - lv["stop"]), r["open"] * (1 + lv["target"])
        if r["low"] <= stop_px:
            kind, px = "stop", stop_px
        elif r["high"] >= tgt_px:
            kind, px = "target", tgt_px
        else:
            kind, px = "close", r["close"]
        rows.append({"date": d, "bucket": lab, "exit": kind, "ret_net": net_return(px / r["open"] - 1.0, cost_pct)})
    return pd.DataFrame(rows)


def _summ(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    n = len(t)
    sd = t["ret_net"].std(ddof=1) if n > 1 else np.nan
    return {"trades": n, "win_rate_pct": float((t["ret_net"] > 0).mean() * 100),
            "avg_net_per_trade_pct": float(t["ret_net"].mean() * 100),
            "t_stat": float(t["ret_net"].mean() / (sd / np.sqrt(n))) if n > 1 and sd and sd > 0 else np.nan,
            "hit_target_pct": float((t["exit"] == "target").mean() * 100),
            "hit_stop_pct": float((t["exit"] == "stop").mean() * 100)}


def verdict(oos_all: dict, oos_same: dict, cfg: JpConfig) -> str:
    n, m = oos_all.get("trades", 0), cfg.plan.min_trades_for_verdict
    if n < m:
        return f"Too few held-out trades ({n}) to judge these levels. Treat them as information only."
    avg, t = oos_all["avg_net_per_trade_pct"], oos_all["t_stat"]
    if avg <= 0:
        return (f"On held-out days these levels LOST {abs(avg):.2f}% per trade on average after costs. "
                "They are a description of this stock's usual range, not a trading signal.")
    if not np.isfinite(t) or t < 2:
        return (f"On held-out days these levels averaged +{avg:.2f}% per trade after costs, but that is within normal "
                "noise (t-stat below 2), so it is not evidence of an edge.")
    return (f"On held-out days these levels averaged +{avg:.2f}% per trade after costs (t-stat {t:.1f}). That is "
            "encouraging but one stock over a limited period can be luck. Check other periods before relying on it.")


def build_plan(ticker: str, jp: pd.DataFrame, sox_close: pd.Series, events: list[Event],
               cfg: JpConfig, today: dt.date | None = None) -> dict:
    """Everything the dashboard shows for one stock. Raises ValueError for unusable data."""
    today = today or dt.date.today()
    pairs, _ = build_pairs(jp, sox_close)
    marked = mark_days(events, list(jp.index), cfg.earnings)
    pairs["earnings"] = [d in marked for d in pairs.index]
    n_earn = int(pairs["earnings"].sum())
    if cfg.earnings.exclude:
        pairs = pairs[~pairs["earnings"]]
    pairs = pairs.dropna(subset=["open", "high", "low", "close", "signal", "gap"])
    if len(pairs) < cfg.plan.min_history_days:
        raise ValueError(f"Only {len(pairs)} usable days for {ticker}; need at least {cfg.plan.min_history_days}. "
                         "Try a longer history, or the stock may be newly listed.")
    days = list(pairs.index)
    is_days, oos_days = split_days(days, cfg.analysis.out_of_sample_fraction)
    p_is, p_oos = pairs.loc[is_days], pairs.loc[oos_days]
    slope, icpt = np.polyfit(p_is["signal"], p_is["gap"], 1)
    resid = float(np.std(p_is["gap"] - (icpt + slope * p_is["signal"]), ddof=1))
    corr = float(p_is["gap"].corr(p_is["signal"]))

    sig = latest_signal(sox_close)
    levels = levels_by_bucket(p_is, cfg)
    lab = str(assign_bucket(pd.Series([sig["move"]]), cfg.analysis.bucket_edges_pct).iloc[0])
    lv = levels[lab]
    prev_close = float(jp["close"].iloc[-1])
    exp_gap = float(icpt + slope * sig["move"])
    entry = prev_close * (1 + exp_gap)
    plan_levels = {"entry": entry, "target": entry * (1 + lv["target"]), "stop": entry * (1 - lv["stop"]),
                   "expected_close": entry * (1 + lv["median_close"]), "expected_gap_pct": exp_gap * 100,
                   "gap_error_pct": resid * 100, "target_pct": lv["target"] * 100, "stop_pct": lv["stop"] * 100,
                   "levels_from": lv["used"], "levels_days": lv["n"]}

    cost = cfg.costs.cost_per_side_pct
    t_is, t_oos = simulate(p_is, levels, cost, cfg), simulate(p_oos, levels, cost, cfg)
    track = {"in-sample, all days": _summ(t_is), "held-out, all days": _summ(t_oos),
             f"in-sample, days like today ({lab})": _summ(t_is[t_is["bucket"] == lab]),
             f"held-out, days like today ({lab})": _summ(t_oos[t_oos["bucket"] == lab])}
    hold = (p_oos["close"].iloc[-1] / p_oos["open"].iloc[0] - 1.0) if len(p_oos) else np.nan
    buy_hold_pct = float(net_return(hold, cost) * 100) if np.isfinite(hold) else np.nan

    warnings = []
    next_tokyo = next_weekday(sig["date"])
    if (today - sig["date"]).days > cfg.plan.stale_days:
        warnings.append(f"The latest SOX session in the data is {sig['date']}, which is old. Refresh before using this.")
    if jp.index[-1] >= next_tokyo:
        warnings.append(f"Tokyo has already traded on {jp.index[-1]}, after that SOX session. These levels describe a "
                        "session that is over; the new SOX signal will apply to the next one.")
    for e in sorted(events, key=lambda e: e.date):
        if today <= e.date <= today + dt.timedelta(days=cfg.plan.earnings_warn_days):
            warnings.append(f"{ticker} is due to report results around {e.date} ({e.source}). Earnings days are dominated by "
                            "the company's own news, so these levels do not apply on that day or the one after.")
    return {"ticker": ticker, "signal": sig, "bucket": lab, "next_tokyo": next_tokyo, "prev_close": prev_close,
            "slope": float(slope), "corr": corr, "levels": plan_levels, "track": track, "buy_hold_oos_pct": buy_hold_pct,
            "oos_days": len(oos_days), "is_days": len(is_days), "earnings_excluded": n_earn if cfg.earnings.exclude else 0,
            "verdict": verdict(track["held-out, all days"], track[f"held-out, days like today ({lab})"], cfg),
            "warnings": warnings, "pairs": pairs, "p_is": p_is}
