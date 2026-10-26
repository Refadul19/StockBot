"""One command: download, cache, analyse, write tables and charts.

  python -m jpgap research
  python -m jpgap research --months 12 --cost-pct 0.15
  python -m jpgap research --config config.jp.yaml --offline      (cache only, no network)
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from gapbot.metrics import split_days

from .align import add_context, build_pairs, log_closed_days
from .analysis import (UNRELIABLE, bench_summary, bucket_labels, bucket_table, buy_hold_curve, daily_average,
                       earnings_compare, equity_curve, feasibility_table, mark_extreme, pct_curve, pct_summary, pool,
                       response_slopes, rule_returns, rule_trades_yen, stock_ranking, summarize_yen, yearly_tables)
from .charts import bucket_chart, equity_chart, group_compare_chart, scatter_chart
from .config import JpConfig
from .crosscheck import bad_ohlc, compare_with_proxy, missing_sessions
from .data import load_daily, safe_name
from .earnings import coverage, load_manual, load_yfinance, mark_days, merge_events
from .summary import what_this_means

log = logging.getLogger("jpgap")
RULES = {"follow": "Follow: buy open, sell close after a big SOX rise",
         "dip": "Dip: buy open, sell close after a big SOX fall",
         "always": "Every day: buy open, sell close (baseline)"}


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:,.2f}", na_rep="-")


def _setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(out / "research.log", mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)


def _load_all(cfg: JpConfig, start, end, fetch_start, refresh, offline):
    """Download (or read from cache) every series. Only SOX is essential; everything else that
    fails is logged clearly and skipped."""
    def get(t: str):
        return load_daily(t, fetch_start, end, cfg.data, refresh, offline)

    sox_df = get(cfg.universe.signal)
    context, proxies = {}, {}
    for t in cfg.universe.context_signals:
        try:
            context[t] = get(t)["close"]
        except Exception as exc:
            log.warning("Context series %s unavailable, continuing without it: %s", t, exc)
    for t in cfg.crosscheck.proxies:
        try:
            proxies[t] = get(t)["close"]
        except Exception as exc:
            log.warning("Cross-check ETF %s unavailable: %s", t, exc)
    names, group_of, daily, failed = {}, {}, {}, []
    for label, members in cfg.universe.groups.items():
        for t, nm in members.items():
            names[t], group_of[t] = nm, label
            try:
                daily[t] = get(t)
            except Exception as exc:
                failed.append(t)
                log.error("FAILED to download %s (%s): %s. Skipping it.", t, nm, exc)
    bench = None
    try:
        bench = get(cfg.universe.benchmark)
    except Exception as exc:
        log.error("FAILED to download benchmark %s: %s. No buy-and-hold comparison.", cfg.universe.benchmark, exc)
    return sox_df, context, proxies, names, group_of, daily, failed, bench


def _data_checks(cfg, sox_df, context, proxies, start) -> list[str]:
    out = ["=" * 20 + " DATA CHECKS " + "=" * 20]
    sox = sox_df["close"]
    ref = context.get("^GSPC")
    if ref is not None:
        miss = [d for d in missing_sessions(list(sox.index), list(ref.index)) if d >= start]
        out.append(f"SOX is missing {len(miss)} session(s) that the S&P 500 has" + (f": {miss}. The move on the next day spans two sessions." if miss else ". Good."))
    bad = [d for d in bad_ohlc(sox_df) if d >= start]
    out.append(f"SOX rows with the close outside the day's low to high: {len(bad)}" + (f" {bad}" if bad else ". Good."))
    for t, p in proxies.items():
        cmp = compare_with_proxy(sox, p, cfg.crosscheck)
        cmp = cmp[[d >= start for d in cmp["date"]]]
        n, ok = len(cmp), int(cmp["match"].sum())
        out.append(f"Big SOX days (move over {cfg.crosscheck.min_sox_move_pct:g}%) vs {t}: {ok} of {n} match "
                   f"(same direction, within {cfg.crosscheck.tolerance_pp:g} percentage points).")
        if n:
            out.append(_fmt(cmp.set_index("date")))
    out.append("The ETFs hold different baskets from the SOX index, so they are a sanity check, not an exact copy. "
               "Free copies of the SOX index itself were not reachable by script.")
    out.append("")
    return out


def run(cfg: JpConfig, refresh: bool = False, offline: bool = False) -> Path:
    out = Path(cfg.output.out_dir)
    _setup_logging(out)
    end = dt.date.fromisoformat(cfg.data.end) if cfg.data.end else dt.date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(months=cfg.data.months)).date()
    fetch_start = start - dt.timedelta(days=cfg.data.lookback_buffer_days)
    log.info("Period %s to %s (%d months). Costs %.2f%% per side.", start, end, cfg.data.months, cfg.costs.cost_per_side_pct)
    sox_df, context, proxies, names, group_of, daily, failed, bench = _load_all(cfg, start, end, fetch_start, refresh, offline)
    sox = sox_df["close"]
    if not daily:
        raise RuntimeError("No Japanese stock data could be downloaded")
    acfg, ccfg, rcfg = cfg.analysis, cfg.costs, cfg.rules
    cost = ccfg.cost_per_side_pct

    # -- pair days
    pairs, skip_rows = {}, []
    for t, df in daily.items():
        p, skipped = build_pairs(df, sox)
        p = p[[d >= start for d in p.index]]          # drop the buffer days
        p = mark_extreme(add_context(p, context), acfg.extreme_sox_pct)
        big = acfg.outlier_warn_pct / 100.0
        for d, r in p[(p["gap"].abs() > big) | (p["intraday"].abs() > big)].iterrows():
            log.warning("%s %s: gap %.1f%% intraday %.1f%% looks like a data error, kept in the data", t, d, r["gap"] * 100, r["intraday"] * 100)
        pairs[t] = p
        skip_rows += [{"ticker": t, **s} for s in skipped if s["date"] >= start]
        if len(p) < acfg.min_days_for_ranking:
            log.warning("%s has only %d usable days (new listing or missing data)", t, len(p))
    skips = pd.DataFrame(skip_rows, columns=["ticker", "date", "reason"])
    skips.to_csv(out / "skipped_days.csv", index=False)
    jp_ref = list(daily[next(iter(daily))].index)
    closed = pd.DataFrame(log_closed_days([d for d in jp_ref if d >= start], [d for d in sox.index if d >= start]))
    for r in closed.itertuples():
        log.info("Skipping weekday %s: %s market closed", r.date, r.market)
    log.info("Days excluded from pairing (all stocks, by reason): %s", skips["reason"].value_counts().to_dict() if len(skips) else {})

    # -- earnings days: mark them, then leave them out of the main results unless asked not to
    ecfg = cfg.earnings
    manual = load_manual(ecfg.manual_file)
    ev_rows, upcoming = [], []
    pairs_all = pairs
    for t, p in pairs_all.items():
        yf_ev = load_yfinance(t, cfg.data.cache_dir, ecfg, refresh, offline) if ecfg.use_yfinance else []
        events = merge_events([e for e in manual if e.ticker == t], yf_ev, ecfg)
        marked = mark_days(events, list(daily[t].index), ecfg)
        p["earnings"] = [d in marked for d in p.index]
        status, detail = coverage(events, p.index[0] if len(p) else start, end, ecfg.max_gap_days)
        if status != "ok":
            log.warning("%s earnings dates %s: %s", t, status.upper(), detail)
        ev_rows.append({"ticker": t, "name": names[t], "dates_found": sum(e.date <= end for e in events),
                        "from_manual": sum(e.source == "manual" for e in events), "status": status,
                        "days_marked": int(p["earnings"].sum()), "note": detail})
        upcoming += [(e.date, t, names[t], e.source, e.time or "time unknown") for e in events if e.date >= end]
    earn_tbl = pd.DataFrame(ev_rows).set_index("ticker")
    earn_tbl.to_csv(out / "earnings_coverage.csv")
    normal_pool = pool({t: p[~p["earnings"]] for t, p in pairs_all.items()}, list(pairs_all))
    earn_pool = pool({t: p[p["earnings"]] for t, p in pairs_all.items()}, list(pairs_all))
    pairs = {t: (p[~p["earnings"]] if ecfg.exclude else p) for t, p in pairs_all.items()}
    n_excluded = int(earn_tbl["days_marked"].sum()) if ecfg.exclude else 0
    log.info("Earnings: %d stock-days marked; %s", int(earn_tbl["days_marked"].sum()),
             "excluded from the main results" if ecfg.exclude else "kept in the results (--include-earnings)")

    all_names = list(pairs)
    all_pool = pool(pairs, all_names)
    all_pool = mark_extreme(all_pool, acfg.extreme_sox_pct)
    day_all = mark_extreme(daily_average(all_pool), acfg.extreme_sox_pct)
    days = sorted(day_all["date"])
    is_days, oos_days = split_days(days, acfg.out_of_sample_fraction)
    labels = bucket_labels(acfg.bucket_edges_pct)
    up_lab, dn_lab = labels[-1], labels[0]

    report: list[str] = []
    P = report.append
    P(f"JAPAN GAP RESEARCH  {start} to {end}  ({cfg.data.months} months, {len(days)} Tokyo days with a usable SOX pairing)")
    P(f"Costs {cost:.2f}% per side. Buckets under {acfg.min_bucket_days} days are marked '{UNRELIABLE}'.")
    if failed:
        P(f"DOWNLOAD FAILED for: {', '.join(failed)}")
    P("")
    report += _data_checks(cfg, sox_df, context, proxies, start)

    # -- earnings section
    P("=" * 20 + " EARNINGS DAYS " + "=" * 20)
    if ecfg.exclude:
        P(f"Earnings days are EXCLUDED from every main result below ({n_excluded} stock-days). Use --include-earnings to keep them.")
    else:
        P("Earnings days are INCLUDED in every result below (--include-earnings). They are still marked and shown separately here.")
    P(f"A report after the close or at an unknown time marks the report day and the next trading day; a report during market hours marks that day only.")
    P(f"Unknown times are assumed {ecfg.unknown_time_as.replace('_', ' ')}. Yahoo times are {'used' if ecfg.trust_yfinance_times else 'ignored (treated as unknown)'}. Manual file: {ecfg.manual_file}.\n")
    P("Days marked per stock in this period, and how complete the date list looks:")
    P(_fmt(earn_tbl[["name", "dates_found", "from_manual", "days_marked", "status", "note"]]))
    short = [t for t, r in earn_tbl.iterrows() if r["status"] != "ok"]
    if short:
        P(f"\nMISSING OR UNCERTAIN earnings dates for: {', '.join(short)}. Earnings days near the gaps are not marked, so those stocks may still carry some earnings noise.")
    P(f"\nEarnings days only ({len(earn_pool)} stock-days), by SOX bucket. Compare with the normal-day tables below:")
    P(_fmt(bucket_table(earn_pool, acfg, cost)))
    P("\nHow different are earnings days? (stock-days)")
    P(_fmt(earnings_compare(normal_pool, earn_pool)))
    if upcoming:
        P("\nUpcoming report dates (not in the data yet):")
        for d, t, nm, src, tm in sorted(upcoming):
            P(f"  {d}  {t} {nm}  [{src}, {tm}]")
    P("")

    # -- rules, main view: percent per trade, no lots, no budget
    P("=" * 20 + " SIMPLE RULES VS BUY-AND-HOLD TOPIX ETF (percent, no lot or budget limit) " + "=" * 20)
    P(f"Rules fixed in the config before looking at results; nothing was tuned. 'follow' = buy open, sell close after SOX rose "
      f"{rcfg.threshold_pct:g}% or more; 'dip' = same after SOX fell {rcfg.threshold_pct:g}% or more; 'always' = every day.")
    P("Equal weight across the stocks selected that day, cash on other days. Returns are after costs. OUT-OF-SAMPLE FIRST.\n")
    trades = {r: rule_returns(pairs, all_names, r, rcfg.threshold_pct, cost) for r in RULES}
    res = {}
    for title, dset in (("OUT-OF-SAMPLE", oos_days), ("IN-SAMPLE", is_days)):
        rows = {r: pct_summary(t[t["date"].isin(set(dset))], dset) for r, t in trades.items()}
        if bench is not None and dset:
            rows[f"TOPIX ETF {cfg.universe.benchmark} buy&hold"] = bench_summary(bench, dset, 100.0, cost)
        res[title] = rows
        P(f"{title} ({len(dset)} days)")
        P(_fmt(pd.DataFrame(rows).T))
        P("")
    curves = {RULES[r]: pct_curve(t, days) for r, t in trades.items()}
    if bench is not None:
        curves[f"TOPIX ETF {cfg.universe.benchmark} buy & hold"] = buy_hold_curve(bench, days, 100.0, cost)

    # -- main tables: each day counts once
    P("=" * 20 + " ALL STOCKS, EACH DAY COUNTED ONCE (average of all stocks per day) " + "=" * 20)
    P("'days' is real calendar days here. Gap and move are the average across the stocks that traded that day.\n")
    t_all = bucket_table(day_all, acfg, cost)
    t_noext = bucket_table(day_all[~day_all["extreme"]], acfg, cost)
    ext = day_all[day_all["extreme"]]
    P("ALL DAYS")
    P(_fmt(t_all))
    P(f"\nWITHOUT EXTREME DAYS (SOX moved more than {acfg.extreme_sox_pct:g}% either way; {len(ext)} days removed)")
    P(_fmt(t_noext))
    P(f"\nEXTREME DAYS ONLY ({len(ext)} days)")
    P(_fmt(ext.set_index("date")[["signal", "gap", "intraday", "next_day", "stocks"]].mul([100, 100, 100, 100, 1]).rename(columns={"signal": "sox_pct", "gap": "gap_pct", "intraday": "intraday_pct", "next_day": "next_day_pct"})) if len(ext) else "none")
    ins_df, oos_df = day_all[day_all["date"].isin(set(is_days))], day_all[day_all["date"].isin(set(oos_days))]
    P(f"\nOUT-OF-SAMPLE PART ONLY ({len(oos_days)} days)")
    P(_fmt(bucket_table(oos_df, acfg, cost)))
    P(f"\nIN-SAMPLE PART ONLY ({len(is_days)} days)")
    P(_fmt(bucket_table(ins_df, acfg, cost)))
    t_all.to_csv(out / "table_all_days_dayavg.csv")
    t_noext.to_csv(out / "table_all_days_dayavg_no_extreme.csv")

    # -- year by year
    yt = yearly_tables(day_all, acfg, cost)
    P("\n" + "=" * 20 + " YEAR BY YEAR (each day counted once) " + "=" * 20)
    P("If the same sign shows up in most years, the pattern is steady. If it flips, it is probably noise.")
    for key, ttl in (("days", "Days in each bucket"), ("avg_intraday_pct", "Average open-to-close move, percent (before costs)"),
                     ("pct_continued", "Percent of days the move continued the gap")):
        P(f"\n{ttl}")
        P(_fmt(yt[key]))
        yt[key].to_csv(out / f"yearly_{key}.csv")
    fade = total = 0
    for y in yt["days"].index:
        for lab, sign in ((up_lab, -1), (dn_lab, 1)):
            n_d, v = yt["days"].loc[y, lab], yt["avg_intraday_pct"].loc[y, lab]
            if n_d >= acfg.min_bucket_days and pd.notna(v):
                total += 1
                fade += int(v * sign > 0)

    # -- by group and by stock (stock-days and day-averaged)
    group_tables = {}
    P("\n" + "=" * 20 + " BY GROUP " + "=" * 20)
    P("Each group is shown twice: averaged per day (each day once), then as stock-days (more rows, but not independent).")
    for label, members in cfg.universe.groups.items():
        have = [t for t in members if t in pairs]
        if not have:
            continue
        gp = pool(pairs, have)
        td = bucket_table(daily_average(gp), acfg, cost)
        ts = bucket_table(gp, acfg, cost)
        group_tables[label] = td
        P(f"\n{label}  ({', '.join(have)})\n-- each day counted once")
        P(_fmt(td))
        P("-- stock-days")
        P(_fmt(ts))
        td.to_csv(out / f"table_group_{safe_name(label)}.csv")
        ts.to_csv(out / f"table_group_{safe_name(label)}_stockdays.csv")
    P("\n" + "=" * 20 + " BY STOCK " + "=" * 20)
    for t, p in pairs.items():
        tb = bucket_table(p, acfg, cost)
        P(f"\n{t} {names[t]}  [{group_of[t]}]")
        P(_fmt(tb))
        tb.to_csv(out / f"table_stock_{safe_name(t)}.csv")

    # -- ranking
    rank = stock_ranking(pairs, names, group_of, acfg.min_days_for_ranking)
    rank.to_csv(out / "ranking.csv", index=False)
    P("\n" + "=" * 20 + " RANKING " + "=" * 20)
    P("rank_follow_open 1 = the opening gap tracks the overnight SOX move most closely; the highest number reacts least.")
    P("rank_late 1 = the biggest catch-up AFTER the open (open-to-close plus next-day move per 1% of SOX).")
    P("Slopes are % per 1% SOX move. Stocks with short history have noisy ranks.")
    if len(rank):
        P(_fmt(rank.set_index("ticker")))
        P("\nBy group (average of its stocks):")
        P(_fmt(rank.groupby("group")[["corr_gap_vs_sox", "gap_per_1pct_sox", "intraday_per_1pct_sox", "next_day_per_1pct_sox"]].mean()))

    # -- feasibility: a separate view
    P("\n" + "=" * 20 + " FEASIBILITY (separate view, not the main result) " + "=" * 20)
    P(f"What an account of {rcfg.capital_yen:,.0f} yen could do with at most {rcfg.budget_per_trade_yen:,.0f} yen per trade, "
      f"whole lots of {ccfg.lot_size}. Only stocks whose one-lot price fits the budget are used.")
    P("Trade size stays fixed in yen (no compounding). If lowest_equity_yen is zero or below, the account would have run out of money,")
    P("and the loss figures after that point are what the fixed-size rule WOULD have lost, not what a real account could lose.")
    ft = feasibility_table(pairs, names, rcfg, ccfg.lot_size)
    P(_fmt(ft))
    ft.to_csv(out / "feasibility_stocks.csv")
    buyable = [t for t in ft.index if ft.loc[t, "buyable"]] if len(ft) else []
    yen_curves = {}
    if not buyable:
        P("\nNo stock fits the budget, so there is nothing to simulate. Raise rules.budget_per_trade_yen.")
    else:
        P(f"\nBuyable now ({len(buyable)}): {', '.join(buyable)}. Prices change, so a stock can be unaffordable on some earlier days; those trades are counted below.")
        yen = {r: rule_trades_yen(pairs, buyable, r, rcfg, ccfg) for r in RULES}
        for title, dset in (("OUT-OF-SAMPLE", oos_days), ("IN-SAMPLE", is_days)):
            ds = set(dset)
            rows = {}
            for r, (tr, un, cp) in yen.items():
                sub = tr[tr["date"].isin(ds)] if len(tr) else tr
                rows[r] = summarize_yen(sub, sum(d in ds for d in un), sum(d in ds for d in cp), dset, rcfg.capital_yen)
            if bench is not None and dset:
                b = bench_summary(bench, dset, rcfg.capital_yen, cost)
                rows[f"TOPIX ETF buy&hold"] = {**b, "profit_yen": b["total_return_pct"] / 100 * rcfg.capital_yen}
            P(f"\n{title} ({len(dset)} days)")
            P(_fmt(pd.DataFrame(rows).T))
        yen_curves = {RULES[r]: equity_curve(tr, days, rcfg.capital_yen) for r, (tr, _, _) in yen.items()}
        if bench is not None:
            yen_curves["TOPIX ETF buy & hold"] = buy_hold_curve(bench, days, rcfg.capital_yen, cost)

    # -- charts
    for label, tb in group_tables.items():
        have = [t for t in cfg.universe.groups[label] if t in pairs]
        bucket_chart(tb, f"{label}: after an overnight SOX move (each day counted once)", out / f"buckets_{safe_name(label)}.png")
        scatter_chart(pool(pairs, have), f"{label}: overnight SOX move vs Tokyo open gap", out / f"scatter_{safe_name(label)}.png")
    scatter_chart(all_pool, "All stocks: overnight SOX move vs Tokyo open gap", out / "scatter_all.png")
    bucket_chart(t_all, "All stocks: after an overnight SOX move (each day counted once)", out / "buckets_all.png")
    bucket_chart(t_noext, f"All stocks without extreme SOX days (over {acfg.extreme_sox_pct:g}%)", out / "buckets_all_no_extreme.png")
    if len(group_tables) > 1:
        group_compare_chart(group_tables, out / "groups_compare.png")
    oos_start = oos_days[0] if oos_days else None
    equity_chart(curves, oos_start, out / "equity_rules_vs_topix.png", "Simple rules vs the TOPIX ETF (percent per trade, after costs)")
    if yen_curves:
        equity_chart(yen_curves, oos_start, out / "equity_feasibility.png", "Feasibility view: whole lots, position cap (after costs)")

    # -- plain-English summary on top
    def row(t, lab):
        return {k: float(v) for k, v in t.loc[lab].items() if k != "note"}
    sl = response_slopes(day_all)
    topix_oos = res["OUT-OF-SAMPLE"].get(f"TOPIX ETF {cfg.universe.benchmark} buy&hold", {}).get("total_return_pct", float("nan"))
    nums = {
        "days": len(days), "start": start, "end": end, "stocks": len(pairs),
        "gap_slope": sl["gap_per_1pct_sox"], "up": row(t_all, up_lab), "down": row(t_all, dn_lab),
        "extreme_days": len(ext), "extreme_pct": acfg.extreme_sox_pct,
        "up_ex": t_noext.loc[up_lab, "avg_intraday_pct"], "down_ex": t_noext.loc[dn_lab, "avg_intraday_pct"],
        "years_fade": fade, "years_total": total, "oos_days": len(oos_days), "thr": rcfg.threshold_pct,
        "oos_dip": res["OUT-OF-SAMPLE"]["dip"]["total_return_pct"], "oos_follow": res["OUT-OF-SAMPLE"]["follow"]["total_return_pct"],
        "earnings_note": (f"Days when a company reported its own results ({int(earn_tbl['days_marked'].sum())} stock-days) are left out."
                          if ecfg.exclude else "Earnings days are included."),
        "oos_topix": topix_oos, "is_dip": res["IN-SAMPLE"]["dip"]["total_return_pct"],
        "is_follow": res["IN-SAMPLE"]["follow"]["total_return_pct"],
        "is_topix": res["IN-SAMPLE"].get(f"TOPIX ETF {cfg.universe.benchmark} buy&hold", {}).get("total_return_pct", float("nan")), "min_days": acfg.min_bucket_days, "cost": cost,
    }
    text = what_this_means(nums) + "\n" + "\n".join(report)
    (out / "report.txt").write_text(text)
    cfg.save(out / "config_used.yaml")
    print("\n" + text)
    log.info("Done. Tables, charts and report.txt are in %s", out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jpgap", description="Japan post-open research (no trading)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("research", help="download, cache, analyse, write tables and charts")
    r.add_argument("--config", help="YAML file, see config.jp.example.yaml")
    r.add_argument("--months", type=int, help="history length in months (default 6)")
    r.add_argument("--end", help="last date, YYYY-MM-DD (default today)")
    r.add_argument("--cost-pct", type=float, help="cost per side in percent (default 0.1)")
    r.add_argument("--refresh", action="store_true", help="ignore the cache and download again")
    r.add_argument("--offline", action="store_true", help="use cached data only")
    r.add_argument("--include-earnings", action="store_true", help="keep earnings days in the main results (default: exclude)")
    a = ap.parse_args(argv)
    cfg = JpConfig.load(a.config)
    if a.months:
        cfg.data.months = a.months
    if a.end:
        cfg.data.end = a.end
    if a.include_earnings:
        cfg.earnings.exclude = False
    if a.cost_pct is not None:
        cfg.costs.cost_per_side_pct = a.cost_pct
    run(cfg, refresh=a.refresh, offline=a.offline)
