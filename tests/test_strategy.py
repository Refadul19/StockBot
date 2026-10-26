import math

import numpy as np
import pandas as pd
import pytest

from gapbot.config import EntryConfig, RiskConfig, ScannerConfig
from gapbot.data import NY, SyntheticSource
from gapbot.scanner import scan_gappers
from gapbot.strategy import SymbolDay, position_size

from conftest import DAY, make_daily, trending_minutes


# -- sizing -------------------------------------------------------------------
def test_size_is_smallest_cap_risk():
    risk = RiskConfig(risk_per_trade_pct=0.5, max_position_pct=100, max_participation=1.0)
    shares, info = position_size(100_000, 10.0, 9.0, recent_volume=1e9, risk=risk)
    assert shares == 500 and info["binding"] == "risk"


def test_size_is_capped_by_volume():
    risk = RiskConfig(risk_per_trade_pct=5, max_position_pct=100, max_participation=0.05)
    shares, info = position_size(1_000_000, 10.0, 9.9, recent_volume=20_000, risk=risk)
    assert shares == 1000 and info["binding"] == "volume"


def test_size_is_capped_by_equity():
    risk = RiskConfig(risk_per_trade_pct=5, max_position_pct=10, max_participation=1.0)
    shares, info = position_size(100_000, 10.0, 9.99, recent_volume=1e9, risk=risk)
    assert shares == 1000 and info["binding"] == "equity"


def test_stop_above_entry_gives_zero():
    assert position_size(100_000, 10.0, 10.5, 1e6, RiskConfig())[0] == 0


# -- scanner ------------------------------------------------------------------
def _universe():
    return pd.concat([
        make_daily("GAPR", prev_close=10.0, today_open=11.0),                    # +10% gap, liquid
        make_daily("FLAT", prev_close=10.0, today_open=10.1),                    # +1% gap
        make_daily("THIN", prev_close=10.0, today_open=12.0, avg_volume=50_000),  # illiquid
        make_daily("PENY", prev_close=1.0, today_open=1.5, avg_volume=1e8),       # under $2
        make_daily("NEWB", prev_close=10.0, today_open=13.0, n_days=5),            # too little history
    ], ignore_index=True)


def test_scanner_filters_and_ranks():
    out = scan_gappers(_universe(), DAY, ScannerConfig())
    assert out["symbol"].tolist() == ["GAPR"]
    assert out["gap_pct"].iloc[0] == pytest.approx(10.0)


def test_scanner_ignores_future_rows():
    d = _universe()
    future = d[d["date"] == DAY].copy()
    future["date"] = pd.Timestamp(DAY).date() + pd.Timedelta(days=1)
    future["close"] = 999.0
    a = scan_gappers(d, DAY, ScannerConfig())
    b = scan_gappers(pd.concat([d, future]), DAY, ScannerConfig())
    pd.testing.assert_frame_equal(a, b)


# -- signals ------------------------------------------------------------------
def test_strong_uptrend_scores_high_and_confirms():
    m = trending_minutes(volume=10_000)
    sd = SymbolDay("UP", m, prev_close=9.0, avg_volume=2e6, gap_pct=11.0)
    now = pd.Timestamp(f"{DAY} 11:00").tz_localize(NY)
    assert sd.trend_score(now) >= 80
    ok, checks = sd.confirm_5m(now, EntryConfig(min_vol_ratio_5m=1.0, max_extension_atr=50))
    assert checks["above_vwap"] and checks["above_ema9"]
    assert checks["vol_ratio"] >= 1.0          # volume rises through the day in this fixture


def test_downtrend_scores_low():
    m = trending_minutes(slope=-0.0008)
    sd = SymbolDay("DN", m, prev_close=9.0, avg_volume=2e6, gap_pct=11.0)
    now = pd.Timestamp(f"{DAY} 11:00").tz_localize(NY)
    assert sd.trend_score(now) <= 20


def test_no_trend_score_before_two_15m_bars_complete():
    sd = SymbolDay("X", trending_minutes(), 9.0, 2e6, 11.0)
    # premarket bars exist, so give it a frame without premarket
    m = trending_minutes()
    m = m[m.index.time >= pd.Timestamp("09:30").time()]
    sd = SymbolDay("X", m, 9.0, 2e6, 11.0)
    assert math.isnan(sd.trend_score(pd.Timestamp(f"{DAY} 09:44").tz_localize(NY)))


def test_recent_volume_window():
    m = trending_minutes(volume=1000)
    sd = SymbolDay("X", m, 9.0, 2e6, 11.0)
    now = pd.Timestamp(f"{DAY} 10:00").tz_localize(NY)
    reg = m.loc[pd.Timestamp(f"{DAY} 09:55").tz_localize(NY): pd.Timestamp(f"{DAY} 09:59").tz_localize(NY)]
    assert sd.recent_volume(now, 5) == pytest.approx(reg["volume"].sum())
