"""Bucket tables, simple long-only rules and the TOPIX buy-and-hold comparison.

Terms used in the output:
  gap          Tokyo open / previous Tokyo close - 1
  intraday     Tokyo close / Tokyo open - 1
  continued    the intraday move had the same sign as the gap (gap == 0 days are not counted)
  t-stat       average intraday move divided by its standard error; roughly, |t| below 2 means
               the average is indistinguishable from noise
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from gapbot.metrics import max_drawdown_pct, split_days

from .config import AnalysisConfig, CostConfig, RulesConfig
from .costs import net_return, trade_pnl_jpy

TABLE_COLS = ["days", "avg_gap_pct", "avg_intraday_pct", "avg_intraday_net_pct",
              "pct_continued", "t_intraday", "avg_next_day_pct", "note"]


UNRELIABLE = "too few days, not reliable"


def bucket_labels(edges_pct: list[float]) -> list[str]:
    e = sorted(edges_pct)
    labels = [f"below {e[0]:+g}%"]
    labels += [f"{a:+g}% to {b:+g}%" for a, b in zip(e[:-1], e[1:])]
    labels.append(f"above {e[-1]:+g}%")
    return labels


def assign_bucket(signal: pd.Series, edges_pct: list[float]) -> pd.Categorical:
    """Left-closed buckets: a move of exactly -2.0% falls in the '-2% to -1%' bucket."""
    e = sorted(edges_pct)
    bins = [-np.inf] + [x / 100.0 for x in e] + [np.inf]
    return pd.cut(signal, bins=bins, labels=bucket_labels(edges_pct), right=False)


def _row(g: pd.DataFrame, cost_pct: float, min_days: int) -> dict:
    n = len(g)
    if n == 0:
        return {"days": 0, **{c: np.nan for c in TABLE_COLS[1:-1]}, "note": UNRELIABLE}
    intr = g["intraday"]
    nz = g[g["gap"] != 0]
    cont = float((np.sign(nz["intraday"]) == np.sign(nz["gap"])).mean() * 100) if len(nz) else np.nan
    sd = intr.std(ddof=1) if n > 1 else np.nan
    t = float(intr.mean() / (sd / math.sqrt(n))) if n > 1 and sd and sd > 0 else np.nan
    return {
        "days": n,
        "avg_gap_pct": g["gap"].mean() * 100,
        "avg_intraday_pct": intr.mean() * 100,
        "avg_intraday_net_pct": intr.map(lambda x: net_return(x, cost_pct)).mean() * 100,
        "pct_continued": cont,
        "t_intraday": t,
        "avg_next_day_pct": g["next_day"].mean() * 100,
        "note": UNRELIABLE if n < min_days else "",
    }


def bucket_table(df: pd.DataFrame, acfg: AnalysisConfig, cost_pct: float) -> pd.DataFrame:
    """One row per SOX bucket. `df` needs columns signal, gap, intraday, next_day."""
    labels = bucket_labels(acfg.bucket_edges_pct)
    b = assign_bucket(df["signal"], acfg.bucket_edges_pct)
    rows = {lab: _row(df[b == lab], cost_pct, acfg.min_bucket_days) for lab in labels}
    all_row = _row(df, cost_pct, acfg.min_bucket_days)
    rows["all days"] = all_row
    return pd.DataFrame.from_dict(rows, orient="index")[TABLE_COLS]


def pool(pairs: dict[str, pd.DataFrame], tickers: list[str]) -> pd.DataFrame:
    """Stack several stocks into one frame (a 'stock-day' per row). Days are not independent
    across stocks, since all of them react to the same SOX move, so counts overstate certainty."""
    frames = [pairs[t].assign(ticker=t).rename_axis("date").reset_index() for t in tickers if t in pairs]
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "signal", "gap", "intraday", "next_day"])
    return pd.concat(frames, ignore_index=True)


def response_slopes(df: pd.DataFrame) -> dict:
    """How many % the stock moves per 1% SOX move: at the open (gap), during the day, next day.
    A laggard has a smaller gap slope and a larger intraday or next-day slope."""
    out = {}
    for col in ("gap", "intraday", "next_day"):
        d = df[["signal", col]].dropna()
        out[f"{col}_per_1pct_sox"] = float(np.polyfit(d["signal"], d[col], 1)[0]) if len(d) > 2 else np.nan
    out["stock_days"] = len(df)
    return out


def stock_ranking(pairs: dict[str, pd.DataFrame], names: dict[str, str], groups: dict[str, str],
                  min_days: int) -> pd.DataFrame:
    """One row per stock. 'Follows closely' = the opening gap tracks the overnight SOX move
    (high correlation and slope). 'Late' = the move arrives after the open: a large slope of
    the open-to-close or next-day move on SOX means the stock keeps catching up."""
    rows = []
    for t, p in pairs.items():
        if len(p) < min_days:
            continue
        sl = response_slopes(p)
        rows.append({
            "ticker": t, "name": names.get(t, ""), "group": groups.get(t, ""), "days": len(p),
            "corr_gap_vs_sox": float(p["gap"].corr(p["signal"])),
            "gap_per_1pct_sox": sl["gap_per_1pct_sox"],
            "intraday_per_1pct_sox": sl["intraday_per_1pct_sox"],
            "next_day_per_1pct_sox": sl["next_day_per_1pct_sox"],
        })
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    r["late_catchup_score"] = r["intraday_per_1pct_sox"] + r["next_day_per_1pct_sox"]
    r["rank_follow_open"] = r["corr_gap_vs_sox"].rank(ascending=False, method="min").astype(int)
    r["rank_late"] = r["late_catchup_score"].rank(ascending=False, method="min").astype(int)
    return r.sort_values("rank_follow_open").reset_index(drop=True)


def split_frame(df: pd.DataFrame, oos_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """In-sample first, out-of-sample last, split by calendar day like gapbot does."""
    days = sorted(df["date"].unique())
    ins, oos = split_days(days, oos_fraction)
    return df[df["date"].isin(set(ins))], df[df["date"].isin(set(oos))]


def daily_average(df: pd.DataFrame) -> pd.DataFrame:
    """Average all stocks within each day, so each calendar day counts once. The SOX move is the
    same for every stock on a day, so stock-day counts overstate how much independent evidence
    there is. Returns one row per day with columns signal, gap, intraday, next_day, stocks."""
    g = df.groupby("date")
    out = g[["signal", "gap", "intraday", "next_day"]].mean()
    out["stocks"] = g.size()
    return out.reset_index()


def earnings_compare(normal: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """How different earnings days are. Both frames are stock-days (columns signal, gap, intraday).
    A weaker link between SOX and the gap on earnings days shows the stock was moved by its own news."""
    rows = {}
    for name, d in (("normal days", normal), ("earnings days", earnings)):
        if d.empty:
            rows[name] = {"stock_days": 0}
            continue
        sl = response_slopes(d)
        rows[name] = {
            "stock_days": len(d),
            "avg_abs_gap_pct": d["gap"].abs().mean() * 100,
            "avg_abs_intraday_pct": d["intraday"].abs().mean() * 100,
            "avg_intraday_pct": d["intraday"].mean() * 100,
            "corr_gap_vs_sox": float(d["gap"].corr(d["signal"])) if len(d) > 2 else np.nan,
            "gap_per_1pct_sox": sl["gap_per_1pct_sox"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def mark_extreme(df: pd.DataFrame, extreme_pct: float) -> pd.DataFrame:
    return df.assign(extreme=df["signal"].abs() > extreme_pct / 100.0)


def yearly_tables(df: pd.DataFrame, acfg: AnalysisConfig, cost_pct: float) -> dict[str, pd.DataFrame]:
    """Year by year, one column per SOX bucket: days, average open-to-close move, % continued.
    `df` should be day-averaged. Shows whether a pattern repeats or lives in one year only."""
    years = sorted({d.year for d in df["date"]})
    labels = bucket_labels(acfg.bucket_edges_pct)
    cols = {"days": {}, "avg_intraday_pct": {}, "pct_continued": {}}
    for y in years:
        t = bucket_table(df[[d.year == y for d in df["date"]]], acfg, cost_pct)
        for k in cols:
            cols[k][y] = t.loc[labels, k]
    return {k: pd.DataFrame(v).T[labels] for k, v in cols.items()}


# ---------------------------------------------------------------------------
# Rules, main view: percent per trade, no lots, no budget
# ---------------------------------------------------------------------------
def rule_mask(df: pd.DataFrame, rule: str, thr_pct: float) -> pd.Series:
    t = thr_pct / 100.0
    if rule == "follow":
        return df["signal"] >= t
    if rule == "dip":
        return df["signal"] <= -t
    if rule == "always":
        return pd.Series(True, index=df.index)
    raise ValueError(rule)


def rule_returns(pairs: dict[str, pd.DataFrame], tickers: list[str], rule: str, thr_pct: float,
                 cost_pct: float) -> pd.DataFrame:
    """Buy the open, sell the close, on days the rule selects. One row per trade with the
    return after paying the cost on both sides. No lots, no budget: nothing is skipped."""
    df = pool(pairs, tickers)
    sel = df[rule_mask(df, rule, thr_pct)]
    return pd.DataFrame({"date": sel["date"].values, "ticker": sel["ticker"].values,
                         "ret_net": sel["intraday"].map(lambda x: net_return(x, cost_pct)).values},
                        columns=["date", "ticker", "ret_net"])


def pct_curve(trades: pd.DataFrame, days: list[dt.date]) -> pd.Series:
    """Equal weight across the stocks selected on a day, cash on days with no trade. Starts at 100."""
    daily = trades.groupby("date")["ret_net"].mean() if len(trades) else pd.Series(dtype=float)
    return (1.0 + daily.reindex(days, fill_value=0.0)).cumprod() * 100.0


def pct_summary(trades: pd.DataFrame, days: list[dt.date]) -> dict:
    curve = pct_curve(trades, days) if days else pd.Series(dtype=float)
    return {
        "days": len(days),
        "trades": len(trades),
        "days_with_a_trade": int(trades["date"].nunique()) if len(trades) else 0,
        "win_rate_pct": float((trades["ret_net"] > 0).mean() * 100) if len(trades) else np.nan,
        "avg_net_per_trade_pct": float(trades["ret_net"].mean() * 100) if len(trades) else np.nan,
        "total_return_pct": float(curve.iloc[-1] - 100.0) if len(curve) else np.nan,
        "max_drawdown_pct": max_drawdown_pct(curve) if len(curve) else np.nan,
    }


def buy_hold_curve(bench: pd.DataFrame, days: list[dt.date], capital: float, cost_pct: float) -> pd.Series:
    """Buy the benchmark at the open of the first day; value = what selling at that day's close
    would bring in after cost, so the one round trip of costs is charged."""
    b = bench.reindex(days)
    c = cost_pct / 100.0
    units = capital / (b["open"].iloc[0] * (1 + c))
    return units * b["close"].ffill() * (1 - c)


def bench_summary(bench: pd.DataFrame, days: list[dt.date], capital: float, cost_pct: float) -> dict:
    if not days:
        return {}
    curve = buy_hold_curve(bench, days, capital, cost_pct)
    return {"days": len(days), "total_return_pct": float((curve.iloc[-1] / capital - 1) * 100),
            "max_drawdown_pct": max_drawdown_pct(curve)}


# ---------------------------------------------------------------------------
# Rules, feasibility view: what a real account of this size could buy
# ---------------------------------------------------------------------------
def feasibility_table(pairs: dict[str, pd.DataFrame], names: dict[str, str], rcfg: RulesConfig,
                      lot_size: int) -> pd.DataFrame:
    """Cost of one lot at the latest open in the period, and whether it fits the per-trade budget."""
    rows = []
    for t, p in pairs.items():
        if p.empty:
            continue
        px = float(p["open"].iloc[-1])
        rows.append({"ticker": t, "name": names.get(t, ""), "latest_open": px,
                     "one_lot_yen": px * lot_size, "buyable": px * lot_size <= rcfg.budget_per_trade_yen})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def rule_trades_yen(pairs: dict[str, pd.DataFrame], tickers: list[str], rule: str, rcfg: RulesConfig,
                    ccfg: CostConfig) -> tuple[pd.DataFrame, list[dt.date], list[dt.date]]:
    """Feasibility view: whole lots only, at most capital // budget trades per day (first in the
    config's stock order, a fixed order chosen without looking at results).
    Returns (trades, dates of trades skipped because one lot did not fit the budget that day,
    dates of trades dropped by the position cap)."""
    df = pool(pairs, tickers)
    sel = df[rule_mask(df, rule, rcfg.threshold_pct)]
    max_pos = max(1, int(rcfg.capital_yen // rcfg.budget_per_trade_yen))
    recs, unaffordable, capped = [], [], []
    for d, day in sel.groupby("date", sort=True):
        taken = 0
        for r in day.itertuples():
            shares, pnl = trade_pnl_jpy(r.open, r.close, rcfg.budget_per_trade_yen, ccfg.cost_per_side_pct, ccfg.lot_size)
            if shares == 0:
                unaffordable.append(d)
            elif taken >= max_pos:
                capped.append(d)
            else:
                taken += 1
                recs.append({"date": d, "ticker": r.ticker, "shares": shares, "pnl_jpy": pnl,
                             "ret_net": pnl / (shares * r.open)})
    cols = ["date", "ticker", "shares", "pnl_jpy", "ret_net"]
    return pd.DataFrame(recs, columns=cols), unaffordable, capped


def equity_curve(trades: pd.DataFrame, days: list[dt.date], capital: float) -> pd.Series:
    daily = trades.groupby("date")["pnl_jpy"].sum() if len(trades) else pd.Series(dtype=float)
    return capital + daily.reindex(days, fill_value=0.0).cumsum()


def summarize_yen(trades: pd.DataFrame, unaffordable: int, capped: int, days: list[dt.date], capital: float) -> dict:
    curve = equity_curve(trades, days, capital) if days else pd.Series(dtype=float)
    return {
        "days": len(days), "trades": len(trades), "no_lot_fits": unaffordable, "over_position_cap": capped,
        "win_rate_pct": float((trades["ret_net"] > 0).mean() * 100) if len(trades) else np.nan,
        "avg_net_per_trade_pct": float(trades["ret_net"].mean() * 100) if len(trades) else np.nan,
        "profit_yen": float(curve.iloc[-1] - capital) if len(curve) else np.nan,
        "lowest_equity_yen": float(curve.min()) if len(curve) else np.nan,
        "total_return_pct": float((curve.iloc[-1] / capital - 1) * 100) if len(curve) else np.nan,
        "max_drawdown_pct": max_drawdown_pct(curve) if len(curve) else np.nan,
    }
