import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gapbot.config import Config  # noqa: E402
from gapbot.data import NY, OHLCV  # noqa: E402

DAY = dt.date(2025, 6, 2)   # a Monday


class StubSource:
    """Daily history for a handful of symbols plus explicit minute bars for DAY."""

    def __init__(self, minute: dict[str, pd.DataFrame], daily: pd.DataFrame):
        self._minute = minute
        self._daily = daily

    def daily(self, start, end):
        return self._daily

    def minute_many(self, symbols, day):
        return {s: self._minute[s] for s in symbols if s in self._minute}

    def trading_days(self, start, end):
        return sorted(d for d in set(self._daily["date"]) if start <= d <= end)


def make_daily(symbol: str, prev_close: float, today_open: float, avg_volume: float = 2e6,
               n_days: int = 25, day: dt.date = DAY) -> pd.DataFrame:
    days = [d.date() for d in pd.bdate_range(end=day - dt.timedelta(days=1), periods=n_days)]
    rows = [(symbol, d, prev_close, prev_close * 1.01, prev_close * 0.99, prev_close, avg_volume) for d in days]
    rows.append((symbol, day, today_open, today_open * 1.1, today_open * 0.95, today_open * 1.05, avg_volume * 3))
    return pd.DataFrame(rows, columns=["symbol", "date", *OHLCV])


def trending_minutes(day: dt.date = DAY, start_price: float = 10.0, slope: float = 0.0008,
                     volume: float = 20_000, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Steady uptrend from 09:30 to 16:00 with rising volume, plus a quiet premarket."""
    rng = np.random.default_rng(seed)
    pre = pd.date_range(pd.Timestamp(f"{day} 08:00").tz_localize(NY), periods=90, freq="1min")
    reg = pd.date_range(pd.Timestamp(f"{day} 09:30").tz_localize(NY), periods=390, freq="1min")
    closes = start_price * np.exp(slope * np.arange(1, 391) + rng.normal(0, noise, 390))
    opens = np.concatenate([[start_price], closes[:-1]])
    highs = np.maximum(opens, closes) * 1.0005
    lows = np.minimum(opens, closes) * 0.9995
    vols = volume * (1 + np.arange(390) / 100)
    reg_df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=reg)
    pre_df = pd.DataFrame({"open": start_price, "high": start_price, "low": start_price, "close": start_price,
                           "volume": 100.0}, index=pre)
    df = pd.concat([pre_df, reg_df])
    df.index.name = "timestamp"
    return df


@pytest.fixture
def cfg() -> Config:
    return Config()
