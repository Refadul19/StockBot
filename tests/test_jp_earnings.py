"""Earnings-day marking: which days get flagged, and how missing data is reported."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from jpgap.analysis import bucket_table, earnings_compare
from jpgap.config import AnalysisConfig, EarningsConfig, JpConfig
from jpgap.earnings import AFTER, DURING, Event, classify, coverage, load_manual, mark_days, merge_events

D = dt.date
CFG = EarningsConfig()


def tokyo_days(start, end, holidays=()):
    return [d.date() for d in pd.bdate_range(start, end) if d.date() not in holidays]


def ev(day, time="", ticker="X"):
    return Event(ticker, day, time, "manual")


def test_after_close_report_marks_report_day_and_next_day():
    days = tokyo_days(D(2026, 10, 19), D(2026, 11, 6))
    assert mark_days([ev(D(2026, 10, 28), "after_close")], days, CFG) == {D(2026, 10, 28), D(2026, 10, 29)}
    assert mark_days([ev(D(2026, 10, 28), "16:00")], days, CFG) == {D(2026, 10, 28), D(2026, 10, 29)}


def test_unknown_time_is_treated_as_after_close():
    days = tokyo_days(D(2026, 10, 19), D(2026, 11, 6))
    assert mark_days([ev(D(2026, 10, 28), "")], days, CFG) == {D(2026, 10, 28), D(2026, 10, 29)}
    assert classify("not a time", CFG) == AFTER
    # the assumption is configurable
    during = EarningsConfig(unknown_time_as="during_hours")
    assert mark_days([ev(D(2026, 10, 28), "")], days, during) == {D(2026, 10, 28)}


def test_during_hours_report_marks_only_the_report_day():
    days = tokyo_days(D(2026, 10, 19), D(2026, 11, 6))
    for t in ("11:00", "during_hours", "09:30", "before_open"):
        assert classify(t, CFG) == DURING
        assert mark_days([ev(D(2026, 10, 28), t)], days, CFG) == {D(2026, 10, 28)}


def test_report_before_a_weekend_marks_the_following_monday():
    days = tokyo_days(D(2026, 10, 19), D(2026, 11, 6))
    fri = D(2026, 10, 30)
    assert fri.weekday() == 4
    assert mark_days([ev(fri, "after_close")], days, CFG) == {fri, D(2026, 11, 2)}   # Monday


def test_report_before_a_holiday_skips_the_holiday():
    hol = {D(2026, 11, 3)}                                            # Culture Day, a Tuesday here
    days = tokyo_days(D(2026, 10, 26), D(2026, 11, 13), hol)
    assert mark_days([ev(D(2026, 11, 2), "after_close")], days, CFG) == {D(2026, 11, 2), D(2026, 11, 4)}


def test_report_dated_on_a_closed_day_reacts_at_the_next_open():
    days = tokyo_days(D(2026, 10, 19), D(2026, 11, 6))
    sat = D(2026, 10, 31)
    assert mark_days([ev(sat, "after_close")], days, CFG) == {D(2026, 11, 2)}
    assert mark_days([ev(sat, "11:00")], days, CFG) == {D(2026, 11, 2)}


def test_report_on_the_last_data_day_does_not_invent_a_next_day():
    days = tokyo_days(D(2026, 10, 19), D(2026, 10, 26))
    assert mark_days([ev(D(2026, 10, 26), "after_close")], days, CFG) == {D(2026, 10, 26)}


def test_marking_rules_are_configurable():
    days = tokyo_days(D(2026, 10, 19), D(2026, 11, 6))
    only_next = EarningsConfig(mark_announcement_day=False)
    assert mark_days([ev(D(2026, 10, 28))], days, only_next) == {D(2026, 10, 29)}
    only_day = EarningsConfig(mark_next_day_after_close=False)
    assert mark_days([ev(D(2026, 10, 28))], days, only_day) == {D(2026, 10, 28)}
    with pytest.raises(ValueError):
        EarningsConfig(unknown_time_as="whenever")


def test_missing_earnings_data_marks_nothing_and_is_reported():
    days = tokyo_days(D(2026, 10, 19), D(2026, 11, 6))
    assert mark_days([], days, CFG) == set()
    status, detail = coverage([], D(2026, 5, 1), D(2026, 10, 26), 120)
    assert status == "none" and "NOT excluded" in detail


def test_coverage_flags_long_silences_and_accepts_quarterly_dates():
    first, last = D(2025, 11, 1), D(2026, 10, 26)
    quarterly = [ev(d) for d in (D(2025, 11, 10), D(2026, 2, 10), D(2026, 5, 12), D(2026, 8, 5))]
    assert coverage(quarterly, first, last, 120)[0] == "ok"
    patchy = [ev(D(2025, 11, 10)), ev(D(2026, 8, 5))]               # nine months with nothing
    status, detail = coverage(patchy, first, last, 120)
    assert status == "uncertain" and "2025-11-10 to 2026-08-05" in detail
    late_start = [ev(D(2026, 8, 5))]                                 # nothing before August
    assert coverage(late_start, first, last, 120)[0] == "uncertain"
    future_only = [ev(D(2026, 10, 30))]
    assert coverage(future_only, first, last, 120)[0] == "none"


def test_manual_file_overrides_yfinance_within_the_window(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("# comment\nticker,date,time,note\n6857.T,2026-10-28,15:30,to verify\n6857.T,notadate,,x\n")
    manual = load_manual(f)
    assert len(manual) == 1 and manual[0].time == "15:30" and manual[0].source == "manual"
    yf = [Event("6857.T", D(2026, 10, 29), "", "yfinance"), Event("6857.T", D(2026, 7, 29), "", "yfinance"),
          Event("8035.T", D(2026, 10, 29), "", "yfinance")]
    merged = merge_events(manual, yf, CFG)
    dates = [(e.ticker, e.date, e.source) for e in merged]
    assert ("6857.T", D(2026, 10, 29), "yfinance") not in dates          # replaced by the manual date
    assert ("6857.T", D(2026, 10, 28), "manual") in dates
    assert ("6857.T", D(2026, 7, 29), "yfinance") in dates               # a different quarter is kept
    assert ("8035.T", D(2026, 10, 29), "yfinance") in dates              # another ticker is untouched


def test_missing_manual_file_is_not_an_error(tmp_path):
    assert load_manual(tmp_path / "nope.csv") == []


def test_prefilled_csv_in_the_repo_is_valid_and_marked_to_verify():
    manual = load_manual("config/jp_earnings.csv")
    got = {(e.ticker, e.date) for e in manual}
    assert ("6857.T", D(2026, 10, 28)) in got and ("4062.T", D(2026, 10, 29)) in got
    assert {t for t, d in got if d == D(2026, 10, 30)} == {"8035.T", "6981.T", "6920.T", "7735.T", "285A.T"}
    assert len(manual) == 7 and "to verify" in open("config/jp_earnings.csv").read()


def test_earnings_compare_and_empty_earnings_table():
    rng = np.random.default_rng(1)
    sig = rng.normal(0, 0.02, 80)
    normal = pd.DataFrame({"signal": sig, "gap": sig * 0.6 + rng.normal(0, 0.003, 80), "intraday": rng.normal(0, 0.01, 80), "next_day": 0.0})
    earn = pd.DataFrame({"signal": sig[:20], "gap": rng.normal(0, 0.04, 20), "intraday": rng.normal(0, 0.03, 20), "next_day": 0.0})
    t = earnings_compare(normal, earn)
    assert t.loc["normal days", "corr_gap_vs_sox"] > t.loc["earnings days", "corr_gap_vs_sox"]
    assert t.loc["earnings days", "avg_abs_gap_pct"] > t.loc["normal days", "avg_abs_gap_pct"]
    empty = earnings_compare(normal, normal.iloc[0:0])
    assert empty.loc["earnings days", "stock_days"] == 0
    bt = bucket_table(pd.DataFrame(columns=["signal", "gap", "intraday", "next_day"], dtype=float), AnalysisConfig(), 0.1)
    assert (bt["days"] == 0).all()


def test_config_has_earnings_defaults():
    c = JpConfig()
    assert c.earnings.exclude and c.earnings.unknown_time_as == "after_close"
    assert JpConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()
