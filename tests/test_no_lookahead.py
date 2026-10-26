"""The most important tests: a decision at time `now` must not change if we delete
every bar after `now`. If one of these fails, backtest results cannot be trusted."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gapbot.config import EntryConfig
from gapbot.data import NY, synth_minutes
from gapbot.strategy import SymbolDay

DAY = dt.date(2025, 6, 3)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_signals_identical_with_future_removed(seed):
    full = synth_minutes(DAY, 20.0, 22.0, 3e6, seed=seed)
    sd_full = SymbolDay("S", full, prev_close=18.0, avg_volume=2e6, gap_pct=11.0)
    cfg = EntryConfig()
    for hhmm in ["09:50", "10:00", "10:35", "11:15", "13:40", "15:30"]:
        now = pd.Timestamp(f"{DAY} {hhmm}").tz_localize(NY)
        cut = full[full.index < now]          # only bars that have finished by `now`
        sd_cut = SymbolDay("S", cut, prev_close=18.0, avg_volume=2e6, gap_pct=11.0)
        a, b = sd_full.trend_score(now), sd_cut.trend_score(now)
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b), hhmm
        ok_a, ca = sd_full.confirm_5m(now, cfg)
        ok_b, cb = sd_cut.confirm_5m(now, cfg)
        assert ok_a == ok_b and ca == cb, hhmm
        assert sd_full.atr5(now) == pytest.approx(sd_cut.atr5(now)), hhmm
        assert sd_full.rank_score(now) == pytest.approx(sd_cut.rank_score(now)), hhmm
        assert sd_full.last_price(now) == pytest.approx(sd_cut.last_price(now)), hhmm


def test_partial_bar_is_not_used():
    """At 10:02 the 10:00-10:05 bar is still forming; the last completed 5m bar is 09:55."""
    full = synth_minutes(DAY, 20.0, 22.0, 3e6, seed=5)
    sd = SymbolDay("S", full, 18.0, 2e6, 11.0)
    now = pd.Timestamp(f"{DAY} 10:02").tz_localize(NY)
    from gapbot.indicators import last_completed
    i = last_completed(sd.m5, now)
    assert sd.m5.index[i] == pd.Timestamp(f"{DAY} 09:55").tz_localize(NY)
