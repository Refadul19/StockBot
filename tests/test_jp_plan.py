"""Reference levels for the web dashboard: expected open, target/stop, honest track record."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from jpgap.config import JpConfig
from jpgap.earnings import Event
from jpgap.plan import (_levels, build_plan, latest_signal, levels_by_bucket, next_weekday, parse_ticker,
                        simulate, verdict)

D = dt.date


def synth(n=400, seed=0, beta=0.5, open_edge=0.0):
    """US closes on weekdays and a Tokyo stock whose gap is beta x the SOX move plus noise."""
    rng = np.random.default_rng(seed)
    days = [d.date() for d in pd.bdate_range("2024-01-02", periods=n)]
    ret = rng.normal(0, 0.02, n)
    sox = pd.Series(1000.0 * np.cumprod(1 + ret), index=days)
    jp_open, jp_close, hi, lo = [], [], [], []
    prev = 1000.0
    for i, d in enumerate(days):
        sig = ret[i - 1] if i else 0.0                       # previous US session's move
        o = prev * (1 + beta * sig + rng.normal(0, 0.003))
        c = o * (1 + open_edge + rng.normal(0, 0.01))
        jp_open.append(o); jp_close.append(c)
        hi.append(max(o, c) * (1 + abs(rng.normal(0, 0.004))))
        lo.append(min(o, c) * (1 - abs(rng.normal(0, 0.004))))
        prev = c
    jp = pd.DataFrame({"open": jp_open, "high": hi, "low": lo, "close": jp_close, "volume": 1}, index=days)
    return jp, sox


def test_parse_ticker_accepts_codes_and_refuses_junk():
    assert parse_ticker("6857") == "6857.T" and parse_ticker(" 285a.t ") == "285A.T" and parse_ticker("8035.T") == "8035.T"
    for bad in ("AAPL", "", "68", "6857.US", "../etc", "6857;rm"):
        with pytest.raises(ValueError):
            parse_ticker(bad)


def test_latest_signal_and_next_weekday():
    s = pd.Series([100.0, 110.0], index=[D(2026, 10, 22), D(2026, 10, 23)])      # Thursday, Friday
    sig = latest_signal(s)
    assert sig["move"] == pytest.approx(0.10) and sig["date"] == D(2026, 10, 23)
    assert next_weekday(D(2026, 10, 23)) == D(2026, 10, 26)                       # Friday -> Monday
    assert next_weekday(D(2026, 10, 21)) == D(2026, 10, 22)


def test_expected_open_follows_gap_beta():
    jp, sox = synth(beta=0.5)
    # make today's SOX move +4% so the expected gap is about +2%
    sox = pd.concat([sox, pd.Series([sox.iloc[-1] * 1.04], index=[D(2026, 1, 5)])])
    plan = build_plan("1111.T", jp, sox, [], JpConfig(), today=D(2026, 1, 6))
    assert plan["slope"] == pytest.approx(0.5, abs=0.06)
    lv = plan["levels"]
    assert lv["expected_gap_pct"] == pytest.approx(2.0, abs=0.4)
    assert lv["entry"] == pytest.approx(plan["prev_close"] * (1 + lv["expected_gap_pct"] / 100))
    assert lv["target"] > lv["entry"] > lv["stop"] > 0


def test_levels_use_only_in_sample_days_and_fall_back_when_a_bucket_is_small():
    jp, sox = synth()
    cfg = JpConfig()
    plan = build_plan("1111.T", jp, sox, [], cfg, today=D(2026, 1, 6))
    p_is = plan["p_is"]
    assert len(p_is) == plan["is_days"] and plan["is_days"] > plan["oos_days"] * 2
    lv = levels_by_bucket(p_is, cfg)
    assert any(v["used"].startswith("all days") for v in lv.values()) or all(v["n"] >= 10 for v in lv.values())
    # percentile settings really change the levels
    wide = JpConfig.from_dict({"plan": {"target_percentile": 90.0, "stop_percentile": 90.0}})
    assert _levels(p_is, wide)["target"] > _levels(p_is, cfg)["target"]


def test_simulate_assumes_stop_first_when_both_levels_are_touched():
    p = pd.DataFrame({"open": [100.0], "high": [110.0], "low": [90.0], "close": [105.0], "signal": [0.0]}, index=[D(2025, 1, 6)])
    lev = {lab: {"target": 0.05, "stop": 0.05, "n": 50, "median_close": 0.0, "used": "x"}
           for lab in ["below -2%", "-2% to -1%", "-1% to +1%", "+1% to +2%", "above +2%"]}
    t = simulate(p, lev, 0.1, JpConfig())
    assert t["exit"].iloc[0] == "stop" and t["ret_net"].iloc[0] < -0.05          # worse than -5% because of costs
    p["low"] = 99.0
    assert simulate(p, lev, 0.1, JpConfig())["exit"].iloc[0] == "target"
    p["high"] = 101.0
    assert simulate(p, lev, 0.1, JpConfig())["exit"].iloc[0] == "close"


def test_track_record_has_all_four_rows_and_flat_data_has_no_edge():
    jp, sox = synth(n=800, open_edge=0.0)
    plan = build_plan("1111.T", jp, sox, [], JpConfig(), today=D(2026, 1, 6))
    assert len(plan["track"]) == 4
    oos = plan["track"]["held-out, all days"]
    assert oos["trades"] == plan["oos_days"] and oos["avg_net_per_trade_pct"] < 0.15   # no edge was built in
    assert "LOST" in plan["verdict"] or "noise" in plan["verdict"] or "Too few" in plan["verdict"]


def test_verdict_wording_depends_on_numbers():
    cfg = JpConfig()
    assert "Too few" in verdict({"trades": 5}, {"trades": 0}, cfg)
    assert "LOST" in verdict({"trades": 60, "avg_net_per_trade_pct": -0.3, "t_stat": -2.0}, {}, cfg)
    assert "noise" in verdict({"trades": 60, "avg_net_per_trade_pct": 0.1, "t_stat": 1.0}, {}, cfg)
    assert "encouraging" in verdict({"trades": 60, "avg_net_per_trade_pct": 0.4, "t_stat": 3.0}, {}, cfg)


def test_warnings_for_stale_sox_session_upcoming_earnings_and_already_traded():
    jp, sox = synth()
    today = D(2026, 6, 1)
    ev = [Event("1111.T", D(2026, 6, 2), "", "manual")]
    plan = build_plan("1111.T", jp, sox, ev, JpConfig(), today=today)
    text = " ".join(plan["warnings"])
    assert "old" in text and "report results" in text
    assert "already traded" in text or jp.index[-1] < next_weekday(sox.index[-1])


def test_earnings_days_are_excluded_from_the_history_by_default():
    jp, sox = synth()
    ev = [Event("1111.T", jp.index[200], "", "manual"), Event("1111.T", jp.index[300], "", "manual")]
    with_ex = build_plan("1111.T", jp, sox, ev, JpConfig(), today=D(2026, 1, 6))
    assert with_ex["earnings_excluded"] == 4
    keep = JpConfig.from_dict({"earnings": {"exclude": False}})
    assert build_plan("1111.T", jp, sox, ev, keep, today=D(2026, 1, 6))["earnings_excluded"] == 0


def test_short_history_is_refused_with_a_clear_message():
    jp, sox = synth(n=80)
    with pytest.raises(ValueError, match="usable days"):
        build_plan("1111.T", jp, sox, [], JpConfig(), today=D(2026, 1, 6))
