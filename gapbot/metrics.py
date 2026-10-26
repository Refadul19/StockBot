"""Performance statistics and the in-sample / out-of-sample split."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd


def max_drawdown_pct(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    return float(((curve / peak) - 1.0).min() * 100.0)


def summarize(trades: pd.DataFrame, daily: pd.DataFrame, curve: pd.Series) -> dict:
    out: dict = {}
    if daily.empty:
        return {"days": 0, "trades": 0}
    start_eq = float(daily["start_equity"].iloc[0])
    end_eq = float(daily["end_equity"].iloc[-1])
    rets = daily["return_pct"] / 100.0
    out["days"] = int(len(daily))
    out["start_equity"] = round(start_eq, 2)
    out["end_equity"] = round(end_eq, 2)
    out["total_return_pct"] = round((end_eq / start_eq - 1) * 100, 2)
    out["green_days_pct"] = round(float((daily["pnl"] > 0).mean() * 100), 1)
    out["best_day_pct"] = round(float(rets.max() * 100), 2)
    out["worst_day_pct"] = round(float(rets.min() * 100), 2)
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    out["sharpe_annualized"] = round(float(rets.mean() / sd * math.sqrt(252)), 2) if sd > 0 else None
    out["max_drawdown_pct"] = round(max_drawdown_pct(curve), 2)
    out["trades"] = int(len(trades))
    if trades.empty:
        return out
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    out["win_rate_pct"] = round(len(wins) / len(trades) * 100, 1)
    out["avg_win"] = round(float(wins["pnl"].mean()), 2) if len(wins) else 0.0
    out["avg_loss"] = round(float(losses["pnl"].mean()), 2) if len(losses) else 0.0
    gross_loss = -float(losses["pnl"].sum())
    out["profit_factor"] = round(float(wins["pnl"].sum()) / gross_loss, 2) if gross_loss > 0 else None
    out["expectancy_r"] = round(float(trades["r_multiple"].mean()), 3)
    out["avg_minutes_held"] = round(float(trades["minutes_held"].mean()), 1)
    out["total_fees"] = round(float(trades["fees"].sum()), 2)
    out["exit_reasons"] = trades["final_exit"].value_counts().to_dict()
    # t-statistic of the mean trade: below ~2 the edge is not distinguishable from luck
    r = trades["r_multiple"]
    out["t_stat_mean_r"] = round(float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else None
    return out


def slippage_cost(fills: pd.DataFrame) -> float:
    """Money lost versus the reference price (bar open, stop level or limit)."""
    if fills.empty:
        return 0.0
    sign = np.where(fills["side"] == "buy", 1.0, -1.0)
    return float(((fills["price"] - fills["reference"]) * sign * fills["shares"]).sum())


def split_days(days: list[dt.date], oos_fraction: float) -> tuple[list[dt.date], list[dt.date]]:
    n_oos = int(round(len(days) * oos_fraction))
    if n_oos <= 0:
        return days, []
    return days[:-n_oos], days[-n_oos:]


def benchmark_daily_returns(bench: pd.DataFrame | None) -> pd.DataFrame:
    """Closes (symbol, date, close) -> wide frame of daily % returns, one column per symbol."""
    if bench is None or bench.empty:
        return pd.DataFrame()
    wide = bench.pivot(index="date", columns="symbol", values="close").sort_index()
    return wide.pct_change() * 100.0


def benchmark_stats(bench_rets: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Buy-and-hold return of each benchmark over the same days, and how closely the
    strategy's daily returns move with it (correlation near 1 = mostly market exposure)."""
    out: dict = {}
    if bench_rets.empty or daily.empty:
        return out
    strat = daily.set_index("date")["return_pct"]
    for sym in bench_rets.columns:
        r = bench_rets[sym].reindex(strat.index)
        if r.isna().all():
            continue
        out[f"{sym}_return_pct"] = round(float(((1 + r.fillna(0) / 100).prod() - 1) * 100), 2)
        both = pd.concat([strat, r], axis=1).dropna()
        if len(both) > 2 and both.iloc[:, 0].std() > 0 and both.iloc[:, 1].std() > 0:
            out[f"corr_{sym}"] = round(float(both.iloc[:, 0].corr(both.iloc[:, 1])), 2)
    return out


def summarize_split(result, oos_fraction: float, bench: pd.DataFrame | None = None) -> dict:
    days = list(result.daily["date"]) if not result.daily.empty else []
    is_days, oos_days = split_days(days, oos_fraction)
    bench_rets = benchmark_daily_returns(bench)
    out = {"all": _subset(result, days, bench_rets)}
    if oos_days:
        out["in_sample"] = _subset(result, is_days, bench_rets)
        out["out_of_sample"] = _subset(result, oos_days, bench_rets)
    return out


def _subset(result, days: list[dt.date], bench_rets: pd.DataFrame | None = None) -> dict:
    if not days:
        return {"days": 0, "trades": 0}
    dset = set(days)
    daily = result.daily[result.daily["date"].isin(dset)]
    trades = result.trades
    if not trades.empty:
        trades = trades[trades["entry_time"].map(lambda x: x.date()).isin(dset)]
    curve = result.equity_curve
    if not curve.empty:
        curve = curve[[ts.date() in dset for ts in curve.index]]
    s = summarize(trades, daily, curve)
    if bench_rets is not None:
        s.update(benchmark_stats(bench_rets, daily))
    s["period"] = f"{days[0]} to {days[-1]}"
    return s
