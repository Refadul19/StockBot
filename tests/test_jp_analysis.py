import datetime as dt

import numpy as np
import pandas as pd
import pytest

from jpgap.analysis import (UNRELIABLE, assign_bucket, bucket_labels, bucket_table, daily_average,
                            feasibility_table, mark_extreme, pct_curve, pct_summary, pool, rule_returns,
                            rule_trades_yen, stock_ranking, yearly_tables)
from jpgap.config import AnalysisConfig, CostConfig, CrossCheckConfig, JpConfig, RulesConfig
from jpgap.crosscheck import bad_ohlc, compare_with_proxy, missing_sessions
from jpgap.summary import what_this_means
from jpgap.costs import lot_shares, net_return, trade_pnl_jpy

D = dt.date


def frame(signal, gap, intraday, next_day=None):
    n = len(signal)
    idx = [D(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    return pd.DataFrame({"signal": signal, "gap": gap, "intraday": intraday,
                         "next_day": next_day if next_day is not None else [0.0] * n,
                         "open": [1000.0] * n, "close": [1000.0 * (1 + i) for i in intraday]}, index=idx)


def test_bucket_edges_are_left_closed():
    e = [-2.0, -1.0, 1.0, 2.0]
    s = pd.Series([-0.03, -0.02, -0.0101, -0.01, 0.0, 0.0099, 0.01, 0.02, 0.05])
    got = list(assign_bucket(s, e).astype(str))
    lab = bucket_labels(e)
    assert got == [lab[0], lab[1], lab[1], lab[2], lab[2], lab[2], lab[3], lab[4], lab[4]]


def test_bucket_table_counts_and_continuation():
    df = frame([0.03, 0.03, 0.03, 0.0], [0.02, 0.02, -0.01, 0.01], [0.01, -0.01, -0.02, 0.01])
    t = bucket_table(df, AnalysisConfig(min_bucket_days=3), 0.1)
    top = t.loc["above +2%"]
    assert top["days"] == 3
    assert top["pct_continued"] == pytest.approx(2 / 3 * 100)   # day 1 up/up and day 3 down/down continued; day 2 reversed
    assert top["avg_gap_pct"] == pytest.approx((0.02 + 0.02 - 0.01) / 3 * 100)
    assert t.loc["all days", "days"] == 4


def test_small_buckets_are_marked_unreliable():
    df = frame([0.03] * 9 + [0.0] * 20, [0.01] * 29, [0.01] * 29)
    t = bucket_table(df, AnalysisConfig(min_bucket_days=10), 0.1)
    assert t.loc["above +2%", "note"] == UNRELIABLE          # 9 days
    assert t.loc["-1% to +1%", "note"] == ""                  # 20 days
    assert t.loc["below -2%", "note"] == UNRELIABLE           # empty bucket
    assert t.loc["below -2%", "days"] == 0


def test_zero_gap_days_not_counted_as_continuation():
    df = frame([0.0, 0.0], [0.0, 0.01], [0.01, 0.01])
    t = bucket_table(df, AnalysisConfig(), 0.1)
    assert t.loc["all days", "pct_continued"] == 100.0


def test_net_return_charges_both_sides():
    assert net_return(0.0, 0.1) < -0.0019 and net_return(0.0, 0.1) > -0.0021
    assert net_return(0.05, 0.0) == pytest.approx(0.05)


def test_lot_rounding():
    assert lot_shares(1000, 1_000_000, 100) == 1000
    assert lot_shares(1500, 1_000_000, 100) == 600           # 666 shares -> 6 lots
    assert lot_shares(15000, 1_000_000, 100) == 0            # one lot is 1.5M, unaffordable
    assert lot_shares(float("nan"), 1_000_000, 100) == 0


def test_trade_pnl_includes_costs_and_unaffordable_is_zero():
    sh, pnl = trade_pnl_jpy(1000.0, 1000.0, 1_000_000, 0.1, 100)
    assert sh == 1000 and pnl == pytest.approx(-1000 * 1000 * 0.002, rel=0.01)
    assert trade_pnl_jpy(20000.0, 21000.0, 1_000_000, 0.1, 100) == (0, 0.0)


def test_rule_returns_use_percent_and_skip_nothing_for_expensive_stocks():
    a = frame([0.02, 0.0, -0.02], [0.0] * 3, [0.01, 0.01, 0.01])
    b = frame([0.02, 0.0, -0.02], [0.0] * 3, [0.01, 0.01, 0.01])
    b["open"], b["close"] = 20000.0, 20200.0          # one lot would cost 2M yen
    r = rule_returns({"A": a, "B": b}, ["A", "B"], "follow", 1.0, 0.1)
    assert sorted(r["ticker"]) == ["A", "B"]           # the expensive stock is NOT skipped
    assert r["ret_net"].iloc[0] == pytest.approx(net_return(0.01, 0.1))
    assert len(rule_returns({"A": a}, ["A"], "dip", 1.0, 0.1)) == 1
    assert len(rule_returns({"A": a}, ["A"], "always", 1.0, 0.1)) == 3


def test_pct_curve_is_cash_on_days_without_trades():
    days = [D(2025, 1, 1), D(2025, 1, 2), D(2025, 1, 3)]
    tr = pd.DataFrame({"date": [days[0], days[0]], "ticker": ["A", "B"], "ret_net": [0.10, 0.00]})
    c = pct_curve(tr, days)
    assert list(c.round(4)) == [105.0, 105.0, 105.0]     # day 1 averages the two stocks
    s = pct_summary(tr, days)
    assert s["trades"] == 2 and s["days_with_a_trade"] == 1 and s["total_return_pct"] == pytest.approx(5.0)


def test_feasibility_lists_buyable_and_caps_positions():
    cheap = frame([0.02] * 3, [0.0] * 3, [0.01] * 3)          # open 1000 -> one lot 100,000 yen
    dear = frame([0.02] * 3, [0.0] * 3, [0.01] * 3)
    dear["open"], dear["close"] = 20000.0, 20200.0            # one lot 2,000,000 yen
    rc = RulesConfig(threshold_pct=1.0, budget_per_trade_yen=500_000, capital_yen=1_000_000)   # 2 positions max
    ft = feasibility_table({"C": cheap, "D": dear}, {}, rc, 100)
    assert bool(ft.loc["C", "buyable"]) and not bool(ft.loc["D", "buyable"])
    tr, un, capped = rule_trades_yen({"C": cheap, "D": dear}, ["C", "D"], "follow", rc, CostConfig())
    assert set(tr["ticker"]) == {"C"} and len(un) == 3 and not capped
    many = {f"S{i}": cheap for i in range(4)}
    tr, un, capped = rule_trades_yen(many, list(many), "follow", rc, CostConfig())
    assert tr.groupby("date").size().max() == 2 and len(capped) == 6   # 2 of 4 dropped on each of 3 days


def test_daily_average_counts_each_day_once():
    a = frame([0.03, 0.0], [0.02, 0.0], [0.01, 0.0])
    b = frame([0.03, 0.0], [0.04, 0.0], [0.03, 0.0])
    d = daily_average(pool({"A": a, "B": b}, ["A", "B"]))
    assert len(d) == 2 and d.loc[0, "stocks"] == 2
    assert d.loc[0, "gap"] == pytest.approx(0.03) and d.loc[0, "intraday"] == pytest.approx(0.02)
    t = bucket_table(d, AnalysisConfig(), 0.1)
    assert t.loc["all days", "days"] == 2          # not 4 stock-days


def test_extreme_marker_is_strictly_greater_than_threshold():
    df = frame([0.04, 0.0401, -0.05, 0.0], [0.0] * 4, [0.0] * 4)
    assert list(mark_extreme(df, 4.0)["extreme"]) == [False, True, True, False]


def test_yearly_tables_split_by_calendar_year():
    idx = [D(2024, 6, 3), D(2024, 6, 4), D(2025, 6, 3), D(2025, 6, 4)]
    df = pd.DataFrame({"date": idx, "signal": [0.03, 0.03, 0.03, -0.03], "gap": [0.02] * 4,
                       "intraday": [-0.01, -0.01, 0.01, 0.01], "next_day": [0.0] * 4})
    y = yearly_tables(df, AnalysisConfig(min_bucket_days=1), 0.1)
    up, dn = bucket_labels(AnalysisConfig().bucket_edges_pct)[-1], bucket_labels(AnalysisConfig().bucket_edges_pct)[0]
    assert y["days"].loc[2024, up] == 2 and y["days"].loc[2025, up] == 1 and y["days"].loc[2025, dn] == 1
    assert y["avg_intraday_pct"].loc[2024, up] == pytest.approx(-1.0)
    assert y["avg_intraday_pct"].loc[2025, up] == pytest.approx(1.0)


def test_extreme_days_can_be_excluded_without_touching_others():
    df = frame([0.10, 0.03, 0.03], [0.05, 0.02, 0.02], [-0.05, -0.01, -0.01])
    df = mark_extreme(df, 4.0)
    full = bucket_table(df, AnalysisConfig(), 0.1).loc["above +2%", "avg_intraday_pct"]
    clean = bucket_table(df[~df["extreme"]], AnalysisConfig(), 0.1).loc["above +2%", "avg_intraday_pct"]
    assert full == pytest.approx((-5 - 1 - 1) / 3) and clean == pytest.approx(-1.0)


def test_crosscheck_missing_sessions_and_bad_rows():
    sox = [D(2025, 6, 2), D(2025, 6, 4)]
    ref = [D(2025, 6, 2), D(2025, 6, 3), D(2025, 6, 4)]
    assert missing_sessions(sox, ref) == [D(2025, 6, 3)]
    df = pd.DataFrame({"high": [10.0, 10.0], "low": [9.0, 9.0], "close": [9.5, 11.0]}, index=[D(2025, 6, 2), D(2025, 6, 3)])
    assert bad_ohlc(df) == [D(2025, 6, 3)]


def test_crosscheck_compare_with_proxy_flags_matches_and_misses():
    idx = [D(2025, 6, d) for d in (2, 3, 4, 5)]
    sox = pd.Series([100.0, 90.0, 99.0, 99.0], index=idx)          # -10%, +10%, 0%
    good = pd.Series([100.0, 91.0, 100.1, 100.1], index=idx)         # -9%, +10%
    bad = pd.Series([100.0, 101.0, 100.0, 100.0], index=idx)         # +1%, -1%
    cfg = CrossCheckConfig(min_sox_move_pct=5.0, tolerance_pp=2.0)
    g = compare_with_proxy(sox, good, cfg)
    assert len(g) == 2 and g["match"].all()
    b = compare_with_proxy(sox, bad, cfg)
    assert not b["match"].any()


def test_what_this_means_is_short_and_has_no_em_dash():
    row = {"days": 20.0, "avg_gap_pct": 2.5, "avg_intraday_pct": -0.8, "t_intraday": -3.0}
    n = {"days": 500, "start": "a", "end": "b", "stocks": 17, "gap_slope": 0.7, "up": row,
         "down": {**row, "avg_gap_pct": -2.5, "avg_intraday_pct": 0.4, "t_intraday": 1.0}, "extreme_days": 6,
         "extreme_pct": 4.0, "up_ex": -0.5, "down_ex": 0.3, "years_fade": 6, "years_total": 8, "oos_days": 150,
         "thr": 1.0, "is_dip": -3.0, "is_follow": -9.0, "is_topix": 8.0, "oos_dip": 2.0, "oos_follow": -1.0, "oos_topix": 3.0, "min_days": 10, "cost": 0.1}
    text = what_this_means(n)
    body = text.splitlines()[1]
    assert 5 <= body.count(". ") + 1 <= 8 and "\u2014" not in text
    assert "partial reversal" in body and "too small" in body       # |t| 3 is a reversal, |t| 1 is noise


def test_ranking_orders_by_correlation_and_skips_short_history():
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 0.02, 60)
    tight = frame(sig, sig * 1.0 + rng.normal(0, 0.001, 60), rng.normal(0, 0.01, 60))
    loose = frame(sig, rng.normal(0, 0.01, 60), rng.normal(0, 0.01, 60))
    short = frame(sig[:5], sig[:5], sig[:5])
    r = stock_ranking({"T": tight, "L": loose, "S": short}, {}, {}, min_days=30)
    assert list(r["ticker"]) == ["T", "L"]
    assert r.loc[0, "rank_follow_open"] == 1


def test_pool_adds_ticker_column():
    a = frame([0.01], [0.0], [0.0])
    p = pool({"A": a, "B": a}, ["A", "B", "missing"])
    assert sorted(p["ticker"]) == ["A", "B"]


def test_config_defaults_and_validation():
    cfg = JpConfig()
    assert cfg.data.months == 6 and cfg.analysis.min_bucket_days == 10 and cfg.analysis.extreme_sox_pct == 4.0
    assert cfg.rules.budget_per_trade_yen > 0 and cfg.rules.capital_yen >= cfg.rules.budget_per_trade_yen
    assert cfg.universe.benchmark == "1306.T"
    labels = list(cfg.universe.groups)
    assert labels[0] == "Semiconductors & chip equipment" and "9984.T" in cfg.universe.groups["AI investment"]
    assert JpConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    with pytest.raises(ValueError):
        JpConfig.from_dict({"data": {"years": 5}})
    with pytest.raises(ValueError):
        JpConfig.from_dict({"nope": {}})


def test_drop_glitches_removes_tenfold_price_errors_but_keeps_real_moves():
    from jpgap.data import drop_glitches
    idx = [D(2026, 3, 1) + dt.timedelta(days=i) for i in range(20)]
    px = [380.0] * 20
    px[10], px[11] = 37.0, 37.1                      # two days at one tenth of the price
    df = pd.DataFrame({"open": px, "close": px}, index=idx)
    out = drop_glitches(df, "X", 3.0, 11)
    assert idx[10] not in out.index and idx[11] not in out.index and len(out) == 18
    real = pd.DataFrame({"open": [100.0] * 10 + [135.0] * 10, "close": [100.0] * 10 + [135.0] * 10}, index=idx)
    assert len(drop_glitches(real, "X", 3.0, 11)) == 20       # a 35% jump is real, not a glitch
