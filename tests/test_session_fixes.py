"""Regression tests for the premarket bugs: indicators must use regular-session bars only,
the 5m ATR has a daily floor, and the volume check ignores premarket bars."""
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from gapbot.config import EntryConfig, ExitConfig, ScannerConfig
from gapbot.data import NY
from gapbot.metrics import benchmark_daily_returns, benchmark_stats
from gapbot.scanner import scan_gappers
from gapbot.strategy import SymbolDay

from conftest import DAY, make_daily, trending_minutes


def _with_premarket(m: pd.DataFrame, price: float, volume: float) -> pd.DataFrame:
    """Replace the fixture's premarket with wild bars of the given price and volume."""
    reg = m[m.index.time >= dt.time(9, 30)]
    pre_idx = pd.date_range(pd.Timestamp(f"{DAY} 04:00").tz_localize(NY), periods=330, freq="1min")
    pre = pd.DataFrame({"open": price, "high": price * 1.2, "low": price * 0.8, "close": price,
                        "volume": volume}, index=pre_idx)
    out = pd.concat([pre, reg])
    out.index.name = "timestamp"
    return out


@pytest.mark.parametrize("hhmm", ["10:00", "10:05", "10:30", "11:45", "14:00"])
def test_premarket_bars_do_not_change_any_signal(hhmm):
    base = trending_minutes(volume=10_000, noise=0.0005, seed=2)
    a = SymbolDay("A", _with_premarket(base, 5.0, 10.0), 9.0, 2e6, 11.0)
    b = SymbolDay("B", _with_premarket(base, 50.0, 1e7), 9.0, 2e6, 11.0)
    now = pd.Timestamp(f"{DAY} {hhmm}").tz_localize(NY)
    sa, sb = a.trend_score(now), b.trend_score(now)
    assert (math.isnan(sa) and math.isnan(sb)) or sa == sb
    assert a.confirm_5m(now, EntryConfig()) == b.confirm_5m(now, EntryConfig())
    assert a.atr5(now) == pytest.approx(b.atr5(now))
    assert a.rank_score(now) == pytest.approx(b.rank_score(now))


def test_first_15m_bar_is_the_session_open():
    sd = SymbolDay("X", trending_minutes(), 9.0, 2e6, 11.0)
    assert sd.m15.index[0].time() == dt.time(9, 30)
    assert sd.m5.index[0].time() == dt.time(9, 30)


def test_trend_score_first_available_at_10():
    sd = SymbolDay("X", trending_minutes(), 9.0, 2e6, 11.0)
    assert math.isnan(sd.trend_score(pd.Timestamp(f"{DAY} 09:59").tz_localize(NY)))
    assert not math.isnan(sd.trend_score(pd.Timestamp(f"{DAY} 10:00").tz_localize(NY)))


def test_atr_floor_from_daily_range():
    m = trending_minutes(noise=0.0)               # very smooth: tiny intraday ATR
    now = pd.Timestamp(f"{DAY} 10:00").tz_localize(NY)
    raw = SymbolDay("X", m, 9.0, 2e6, 11.0, daily_atr=0.0)
    floored = SymbolDay("X", m, 9.0, 2e6, 11.0, daily_atr=0.884)
    assert floored.atr5(now) == pytest.approx(max(raw.atr5(now), 0.884 / math.sqrt(78)))
    assert floored.atr5(now) > raw.atr5(now)


def test_atr_floor_can_be_switched_off():
    m = trending_minutes(noise=0.0)
    now = pd.Timestamp(f"{DAY} 10:00").tz_localize(NY)
    off = SymbolDay("X", m, 9.0, 2e6, 11.0, daily_atr=5.0, exit_cfg=ExitConfig(atr_floor_from_daily=False))
    plain = SymbolDay("X", m, 9.0, 2e6, 11.0)
    assert off.atr5(now) == pytest.approx(plain.atr5(now))


def test_volume_check_needs_enough_session_bars():
    sd = SymbolDay("X", trending_minutes(), 9.0, 2e6, 11.0)
    cfg = EntryConfig(min_vol_bars_5m=3, min_vol_ratio_5m=0.0, max_extension_atr=100)
    # at 09:45 the 09:30, 09:35 and 09:40 bars are complete: only two bars before the latest
    ok, info = sd.confirm_5m(pd.Timestamp(f"{DAY} 09:45").tz_localize(NY), cfg)
    assert not ok and "not enough" in info["reason"]
    _, info = sd.confirm_5m(pd.Timestamp(f"{DAY} 09:50").tz_localize(NY), cfg)
    assert "vol_ratio" in info


def test_volume_ratio_uses_session_bars_only():
    m = trending_minutes(volume=10_000)
    sd = SymbolDay("X", m, 9.0, 2e6, 11.0)
    now = pd.Timestamp(f"{DAY} 10:00").tz_localize(NY)
    _, info = sd.confirm_5m(now, EntryConfig(min_vol_ratio_5m=0.0, max_extension_atr=100))
    reg = m[m.index.time >= dt.time(9, 30)]
    v5 = reg["volume"].resample("5min").sum()
    expected = v5.iloc[5] / v5.iloc[:5].mean()
    assert info["vol_ratio"] == pytest.approx(expected, abs=0.01)


def test_scanner_reports_daily_atr():
    out = scan_gappers(make_daily("GAPR", prev_close=10.0, today_open=11.0), DAY, ScannerConfig())
    # fixture daily bars: high 10.1, low 9.9, flat closes -> true range 0.2
    assert out["daily_atr"].iloc[0] == pytest.approx(0.2)


def test_benchmark_stats():
    days = [dt.date(2025, 1, d) for d in (2, 3, 6, 7)]
    bench = pd.DataFrame({"symbol": "SPY", "date": [dt.date(2024, 12, 31), *days],
                          "close": [100.0, 101.0, 102.01, 101.0, 102.0]})
    daily = pd.DataFrame({"date": days, "return_pct": [2.0, 2.0, -2.0, 2.0]})
    s = benchmark_stats(benchmark_daily_returns(bench), daily)
    assert s["SPY_return_pct"] == pytest.approx(2.0, abs=0.01)
    assert s["corr_SPY"] > 0.9
    sub = benchmark_stats(benchmark_daily_returns(bench), daily.iloc[:2])
    assert sub["SPY_return_pct"] == pytest.approx(2.01, abs=0.01)
