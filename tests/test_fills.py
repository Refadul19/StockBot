import pytest

from gapbot import fills
from gapbot.config import FillConfig


def test_market_buy_pays_slippage_spread_and_impact():
    cfg = FillConfig(slippage_bps=5, half_spread_bps=5, impact_bps_per_1pct_participation=2)
    # 100 shares of a 10,000 share bar = 1% participation -> 5 + 5 + 2 = 12 bps
    assert fills.market_buy(100.0, 100, 10_000, cfg) == pytest.approx(100.12)


def test_market_sell_is_symmetric():
    cfg = FillConfig(slippage_bps=5, half_spread_bps=5, impact_bps_per_1pct_participation=2)
    assert fills.market_sell(100.0, 100, 10_000, cfg) == pytest.approx(99.88)


def test_zero_volume_bar_counts_as_full_participation():
    assert fills.participation_pct(10, 0) == 100.0


def test_stop_fills_at_stop_minus_costs_when_traded_through():
    cfg = FillConfig(slippage_bps=0, half_spread_bps=0, impact_bps_per_1pct_participation=0, stop_extra_slippage_bps=10)
    assert fills.stop_sell(50.0, 51.0, 100, 1e9, cfg) == pytest.approx(50.0 * (1 - 0.001))


def test_stop_gap_through_fills_from_the_open():
    cfg = FillConfig(slippage_bps=0, half_spread_bps=0, impact_bps_per_1pct_participation=0, stop_extra_slippage_bps=0)
    assert fills.stop_sell(50.0, 47.0, 100, 1e9, cfg) == pytest.approx(47.0)


def test_limit_touch_is_not_a_fill_by_default():
    assert fills.limit_sell_fills(10.0, 9.9, 10.0) is None
    assert fills.limit_sell_fills(10.0, 9.9, 10.01) == 10.0


def test_limit_touch_fills_in_paper_mode():
    assert fills.limit_sell_fills(10.0, 9.9, 10.0, on_touch=True) == 10.0


def test_limit_gap_above_fills_at_open():
    assert fills.limit_sell_fills(10.0, 10.5, 10.6) == 10.5


def test_fees_only_on_sells_and_taf_is_capped():
    cfg = FillConfig()
    assert fills.fees("buy", 1000, 10.0, cfg) == 0.0
    big = fills.fees("sell", 1_000_000, 1.0, cfg)
    assert big == pytest.approx(cfg.finra_taf_cap + cfg.sec_fee_rate * 1_000_000)


def test_optimistic_profile_has_no_costs():
    o = FillConfig.optimistic()
    assert fills.market_buy(10.0, 1e6, 1, o) == 10.0
    assert o.limit_fills_on_touch
