"""Date alignment between US sessions and Tokyo days. This is where bugs hide."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from jpgap.align import (add_context, build_pairs, log_closed_days, tokyo_open_utc, us_close_utc,
                         weekday_range)

D = dt.date


def us_series(start, end, holidays=()):
    """US closes where each session's close equals 100 + its ordinal, so the signal is unique."""
    days = [d for d in weekday_range(start, end) if d not in holidays]
    return pd.Series([100.0 + d.toordinal() % 1000 for d in days], index=days, dtype=float)


def jp_frame(start, end, holidays=()):
    days = [d for d in weekday_range(start, end) if d not in holidays]
    n = len(days)
    return pd.DataFrame({"open": np.linspace(100, 110, n), "close": np.linspace(101, 112, n)}, index=days)


def test_normal_week_pairs_with_previous_us_day():
    us = us_series(D(2025, 6, 1), D(2025, 6, 20))
    jp = jp_frame(D(2025, 6, 2), D(2025, 6, 20))
    p, _ = build_pairs(jp, us)
    assert p.loc[D(2025, 6, 10), "us_date"] == D(2025, 6, 9)       # Tue pairs with Mon
    assert p.loc[D(2025, 6, 10), "us_prev_date"] == D(2025, 6, 6)  # the session before is Friday
    assert p.loc[D(2025, 6, 11), "us_date"] == D(2025, 6, 10)


def test_monday_pairs_with_friday():
    us = us_series(D(2025, 6, 1), D(2025, 6, 20))
    jp = jp_frame(D(2025, 6, 2), D(2025, 6, 20))
    p, _ = build_pairs(jp, us)
    assert p.loc[D(2025, 6, 9), "us_date"] == D(2025, 6, 6)        # never Sunday, never the Monday session
    assert p.loc[D(2025, 6, 9), "us_prev_date"] == D(2025, 6, 5)


def test_never_uses_us_session_of_same_date():
    """The US session dated T closes after Tokyo T is over. It must not be the signal for T."""
    us = us_series(D(2025, 6, 1), D(2025, 6, 30))
    jp = jp_frame(D(2025, 6, 2), D(2025, 6, 30))
    p, _ = build_pairs(jp, us)
    assert (pd.Series(p.index, index=p.index) > pd.Series(p["us_date"], index=p.index)).all()


def test_us_holiday_skips_next_tokyo_day():
    # US closed Thu 2025-07-03? use Fri 4 July: Tokyo Mon 7 July has no fresh US session
    hol = {D(2025, 7, 4)}
    us = us_series(D(2025, 6, 20), D(2025, 7, 20), hol)
    jp = jp_frame(D(2025, 6, 23), D(2025, 7, 20))
    p, skipped = build_pairs(jp, us)
    assert D(2025, 7, 7) not in p.index
    assert {"date": D(2025, 7, 7), "reason": "no_us_session_in_window"} in skipped
    assert D(2025, 7, 8) in p.index                                 # Tue pairs with Mon 7 July


def test_us_holiday_on_monday_skips_tuesday_only():
    hol = {D(2025, 9, 1)}                                           # Labor Day
    us = us_series(D(2025, 8, 20), D(2025, 9, 30), hol)
    jp = jp_frame(D(2025, 8, 21), D(2025, 9, 30))
    p, skipped = build_pairs(jp, us)
    assert D(2025, 9, 2) not in p.index
    assert D(2025, 9, 3) in p.index and D(2025, 9, 1) in p.index    # Mon pairs with Friday 29 Aug


def test_tokyo_holiday_skips_next_day_with_two_us_sessions():
    jp_hol = {D(2025, 7, 21)}                                       # Marine Day, Monday
    us = us_series(D(2025, 7, 1), D(2025, 8, 15))
    jp = jp_frame(D(2025, 7, 2), D(2025, 8, 15), jp_hol)
    p, skipped = build_pairs(jp, us)
    assert D(2025, 7, 21) not in p.index
    assert D(2025, 7, 22) not in p.index                            # gap covers Fri and Mon US sessions
    assert {"date": D(2025, 7, 22), "reason": "multiple_us_sessions_in_window"} in skipped
    assert D(2025, 7, 23) in p.index


def test_golden_week_and_new_year_streaks():
    jp_hol = {D(2025, 4, 29), D(2025, 5, 5), D(2025, 5, 6), D(2025, 5, 2)}
    us = us_series(D(2025, 4, 15), D(2025, 5, 31))
    jp = jp_frame(D(2025, 4, 16), D(2025, 5, 30), jp_hol)
    p, _ = build_pairs(jp, us)
    for d in (D(2025, 4, 30), D(2025, 5, 7), D(2025, 5, 3)):
        assert d not in p.index or p.loc[d, "us_date"] < d
    assert D(2025, 4, 30) not in p.index and D(2025, 5, 7) not in p.index
    assert D(2025, 5, 8) in p.index
    # every row that survives uses exactly one session, and it is the latest one before T
    for t, r in p.iterrows():
        assert r["us_date"] == max(d for d in us.index if d < t)


@pytest.mark.parametrize("t", [D(2025, 3, 6), D(2025, 3, 10), D(2025, 3, 11), D(2025, 11, 3), D(2025, 11, 4), D(2024, 3, 11)])
def test_us_close_is_before_tokyo_open_around_dst_changes(t):
    """US DST starts the 2nd Sunday in March and ends the 1st Sunday in November. Tokyo has none."""
    prev = t - dt.timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= dt.timedelta(days=1)
    assert us_close_utc(prev) < tokyo_open_utc(t)


def test_us_close_in_jst_shifts_by_one_hour_with_dst():
    jst = lambda d: us_close_utc(d).astimezone(dt.timezone(dt.timedelta(hours=9)))
    assert jst(D(2025, 7, 1)).hour == 5      # summer: 05:00 JST
    assert jst(D(2025, 12, 1)).hour == 6     # winter: 06:00 JST


def test_signal_does_not_change_when_later_data_is_removed():
    """No look-ahead: cutting every series at T leaves T's signal, gap and intraday unchanged."""
    us = us_series(D(2025, 5, 1), D(2025, 6, 30))
    jp = jp_frame(D(2025, 5, 2), D(2025, 6, 30))
    full, _ = build_pairs(jp, us)
    t = D(2025, 6, 11)
    cut_us = us[[d for d in us.index if d < t]]          # only sessions that finished before T
    cut_jp = jp[[d <= t for d in jp.index]]
    part, _ = build_pairs(cut_jp, cut_us)
    for col in ("signal", "gap", "intraday", "us_date"):
        assert part.loc[t, col] == full.loc[t, col]


def test_gap_and_intraday_definitions():
    jp = pd.DataFrame({"open": [100.0, 110.0, 99.0], "close": [105.0, 121.0, 99.0]},
                      index=[D(2025, 6, 3), D(2025, 6, 4), D(2025, 6, 5)])
    us = pd.Series([100.0, 98.0, 99.0], index=[D(2025, 6, 2), D(2025, 6, 3), D(2025, 6, 4)])
    p, _ = build_pairs(jp, us)
    r = p.loc[D(2025, 6, 4)]
    assert r["gap"] == pytest.approx(110 / 105 - 1)
    assert r["intraday"] == pytest.approx(121 / 110 - 1)
    assert r["signal"] == pytest.approx(98 / 100 - 1)         # session of 3 June vs 2 June
    assert r["next_day"] == pytest.approx(99 / 121 - 1)


def test_missing_prices_are_skipped_and_logged():
    jp = pd.DataFrame({"open": [100.0, np.nan, 99.0], "close": [105.0, 121.0, 99.0]},
                      index=[D(2025, 6, 3), D(2025, 6, 4), D(2025, 6, 5)])
    us = us_series(D(2025, 6, 1), D(2025, 6, 10))
    p, skipped = build_pairs(jp, us)
    assert D(2025, 6, 4) not in p.index
    assert {"date": D(2025, 6, 4), "reason": "missing_price"} in skipped


def test_first_tokyo_day_and_missing_history_are_skipped():
    us = us_series(D(2025, 6, 3), D(2025, 6, 10))
    jp = jp_frame(D(2025, 6, 3), D(2025, 6, 10))
    p, skipped = build_pairs(jp, us)
    assert D(2025, 6, 3) not in p.index
    assert D(2025, 6, 4) not in p.index        # only one US close exists before it
    assert {"date": D(2025, 6, 4), "reason": "no_earlier_us_session"} in skipped


def test_log_closed_days_lists_each_market():
    us = us_series(D(2025, 7, 1), D(2025, 7, 11), {D(2025, 7, 4)})
    jp = jp_frame(D(2025, 7, 1), D(2025, 7, 11), {D(2025, 7, 9)})
    rows = log_closed_days(list(jp.index), list(us.index))
    assert {"date": D(2025, 7, 4), "market": "us", "reason": "closed"} in rows
    assert {"date": D(2025, 7, 9), "market": "tokyo", "reason": "closed"} in rows


def test_context_series_uses_same_us_date():
    us = us_series(D(2025, 6, 1), D(2025, 6, 20))
    jp = jp_frame(D(2025, 6, 2), D(2025, 6, 20))
    p, _ = build_pairs(jp, us)
    fx = pd.Series([100.0, 101.0, 102.0], index=[D(2025, 6, 9), D(2025, 6, 10), D(2025, 6, 11)])
    q = add_context(p, {"fx": fx})
    assert q.loc[D(2025, 6, 11), "fx"] == pytest.approx(101.0 / 100.0 - 1)   # US date is 10 June
    assert np.isnan(q.loc[D(2025, 6, 17), "fx"])                              # no FX bar on 16 June
