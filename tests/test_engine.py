import datetime as dt

import pandas as pd
import pytest

from gapbot.config import Config, FillConfig
from gapbot.data import NY
from gapbot.engine import Backtester, Position

from conftest import DAY, StubSource, make_daily, trending_minutes

ZERO = FillConfig(slippage_bps=0, half_spread_bps=0, impact_bps_per_1pct_participation=0,
                  stop_extra_slippage_bps=0, sec_fee_rate=0, finra_taf_per_share=0)


def ts(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{DAY} {hhmm}").tz_localize(NY)


def _bt(fill: FillConfig = ZERO) -> Backtester:
    cfg = Config.from_dict({**Config().to_dict(), "fill": fill.__dict__})
    bt = Backtester(cfg, StubSource({}, make_daily("X", 10, 11)))
    bt.cash, bt.trades, bt.fill_log = 0.0, [], []
    return bt


def _pos(**kw) -> Position:
    base = dict(symbol="X", entry_time=ts("10:00"), entry_price=10.0, initial_shares=100, shares=100,
                initial_stop=9.5, stop=9.5, risk_per_share=0.5, target=11.0, highest=10.0,
                entry_fees=0.0, info={})
    base.update(kw)
    return Position(**base)


def test_stop_checked_before_target_in_same_bar():
    bt, p = _bt(), _pos()
    bt._check_resting(p, (10.0, 11.5, 9.0, 10.0, 1e6), ts("10:05"))
    assert p.shares == 0 and p.exits[0]["reason"] == "stop" and p.exits[0]["price"] == pytest.approx(9.5)


def test_target_sells_three_quarters():
    bt, p = _bt(), _pos()
    bt._check_resting(p, (10.5, 11.2, 10.4, 11.1, 1e6), ts("10:05"))
    assert p.target_done and p.shares == 25
    assert p.exits[0]["shares"] == 75 and p.exits[0]["price"] == pytest.approx(11.0)


def test_target_touch_does_not_fill():
    bt, p = _bt(), _pos()
    bt._check_resting(p, (10.5, 11.0, 10.4, 10.9, 1e6), ts("10:05"))
    assert not p.target_done and p.shares == 100


def test_gap_through_stop_fills_at_open():
    bt, p = _bt(), _pos()
    bt._check_resting(p, (9.0, 9.2, 8.8, 9.1, 1e6), ts("10:05"))
    assert p.exits[0]["price"] == pytest.approx(9.0)


def test_pnl_accounting_with_partial_exits():
    bt, p = _bt(), _pos()
    positions, cooldown = {"X": p}, {}
    bt._sell(p, 75, 11.0, ts("10:10"), "target", 11.0)
    bt._sell(p, 25, 12.0, ts("10:30"), "trail", 12.0)
    bt._close(p, positions, cooldown, ts("10:30"))
    tr = bt.trades[0]
    assert tr["pnl"] == pytest.approx(75 * 1.0 + 25 * 2.0)
    assert tr["r_multiple"] == pytest.approx(125 / 50)
    assert tr["exit_reasons"] == "target+trail"
    assert "X" not in positions and cooldown["X"] == ts("10:30") + pd.Timedelta(minutes=30)


def _run_trend_day(**overrides):
    minute = {"UP": trending_minutes(start_price=11.0, volume=40_000, noise=0.0004, seed=3)}
    daily = make_daily("UP", prev_close=10.0, today_open=11.0)
    cfg = Config().with_overrides({"entry.min_vol_ratio_5m": 0.0, "entry.max_extension_atr": 100.0, **overrides})
    return Backtester(cfg, StubSource(minute, daily)).run(DAY, DAY)


def test_end_to_end_entry_fills_on_bar_after_decision():
    r = _run_trend_day()
    assert len(r.trades) >= 1
    t = r.trades.iloc[0]
    decided = pd.Timestamp(t["sig_decided_at"])
    assert t["entry_time"] == decided          # fill bar starts at the decision moment, i.e. the NEXT bar
    entry_bar_open = trending_minutes(start_price=11.0, volume=40_000, noise=0.0004, seed=3).loc[t["entry_time"], "open"]
    assert t["entry_price"] >= entry_bar_open  # paid at least the open (plus costs)


def test_end_to_end_everything_closed_by_the_bell():
    r = _run_trend_day()
    assert (r.trades["exit_time"].map(lambda x: x.time()) <= dt.time(15, 59)).all()
    assert r.daily["end_equity"].iloc[0] == pytest.approx(r.equity_curve.iloc[-1], rel=1e-3)


def test_no_entries_before_window():
    r = _run_trend_day(**{"entry.no_entry_before": "14:00"})
    assert (r.trades["entry_time"].map(lambda x: x.time()) > dt.time(14, 0)).all()


def test_time_stop_exits_flat_trade():
    # flat market: nothing makes progress, so trades should end on the time stop or a stop, never a target
    minute = {"FLAT": trending_minutes(start_price=11.0, slope=0.00001, volume=40_000, noise=0.0003, seed=4)}
    daily = make_daily("FLAT", prev_close=10.0, today_open=11.0)
    cfg = Config().with_overrides({"entry.min_vol_ratio_5m": 0.0, "entry.max_extension_atr": 100.0,
                                   "entry.min_trend_score": 0.0})
    r = Backtester(cfg, StubSource(minute, daily)).run(DAY, DAY)
    assert len(r.trades) >= 1
    assert "target" not in set(r.trades["final_exit"])
    assert "time_stop" in set(r.trades["final_exit"])


def test_cash_is_conserved_without_costs():
    minute = {"UP": trending_minutes(start_price=11.0, volume=40_000, noise=0.0004, seed=3)}
    daily = make_daily("UP", prev_close=10.0, today_open=11.0)
    cfg = Config.from_dict({**Config().to_dict(), "fill": ZERO.__dict__}).with_overrides(
        {"entry.min_vol_ratio_5m": 0.0, "entry.max_extension_atr": 100.0})
    r = Backtester(cfg, StubSource(minute, daily)).run(DAY, DAY)
    assert r.daily["pnl"].iloc[0] == pytest.approx(r.trades["pnl"].sum(), abs=0.05)


def test_only_days_restricts_the_run():
    from gapbot.data import SyntheticSource
    src = SyntheticSource(n_symbols=10, seed=1)
    start, end = dt.date(2025, 3, 3), dt.date(2025, 3, 14)
    days = src.trading_days(start, end)
    r = Backtester(Config(), src).run(start, end, only_days=days[-3:])
    assert list(r.daily["date"]) == days[-3:]
