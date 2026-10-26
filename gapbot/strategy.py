"""Entry rules, intraday ranking and position sizing.

Everything here is a pure function of data up to `now`, the moment a decision is
made (the close of the last completed 1-minute bar). Orders decided at `now` fill
on the next bar, never on a bar the decision has already seen.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import EntryConfig, ExitConfig, RiskConfig
from . import indicators as ind

ONE_MIN = pd.Timedelta(minutes=1)
SESSION_OPEN = dt.time(9, 30)


@dataclass
class SymbolDay:
    """One candidate stock for one day, with indicators precomputed causally."""
    symbol: str
    minute: pd.DataFrame
    prev_close: float
    avg_volume: float
    gap_pct: float
    daily_atr: float = math.nan            # 14-day ATR from daily bars, floors the 5m ATR
    exit_cfg: ExitConfig = field(default_factory=ExitConfig)

    def __post_init__(self) -> None:
        # Indicators use regular-session bars only (09:30 onward). Premarket bars are thin
        # and quiet, and mixing them in shrinks the ATR (stops too tight, size too big),
        # makes the first session bars look like volume surges, and compares the first
        # 15m bar's low with a premarket low.
        self.raw_minute = self.minute
        if not self.minute.empty:
            self.minute = self.minute[self.minute.index.time >= SESSION_OPEN]
        m = self.minute
        m5 = ind.resample(m, 5)
        m15 = ind.resample(m, 15)
        if not m5.empty:
            m5["ema9"] = ind.ema(m5["close"], 9)
            m5["atr"] = ind.atr(m5, self.exit_cfg.atr_period)
            m5["macd_hist"] = ind.macd_hist(m5["close"])
            m5["vwap"] = ind.session_vwap(m5)
        if not m15.empty:
            m15["ema9"] = ind.ema(m15["close"], 9)
            m15["ema20"] = ind.ema(m15["close"], 20)
            m15["vwap"] = ind.session_vwap(m15)
        self.m5, self.m15 = m5, m15
        self._session_vol = m["volume"].cumsum()
        self._atr_floor = (self.daily_atr / math.sqrt(78)
                           if self.exit_cfg.atr_floor_from_daily and self.daily_atr == self.daily_atr
                           and self.daily_atr > 0 else 0.0)

    # -- helpers -----------------------------------------------------------
    def minute_pos(self, now: pd.Timestamp) -> int:
        """Index of the last 1-minute bar completed by `now`."""
        return int(self.minute.index.searchsorted(now - ONE_MIN, side="right")) - 1

    def last_price(self, now: pd.Timestamp) -> float:
        i = self.minute_pos(now)
        return float(self.minute["close"].iloc[i]) if i >= 0 else math.nan

    def recent_volume(self, now: pd.Timestamp, minutes: int) -> float:
        i = self.minute_pos(now)
        if i < 0:
            return 0.0
        window = self.minute.iloc[max(0, i - minutes + 1): i + 1]
        window = window[window.index >= now - pd.Timedelta(minutes=minutes)]
        return float(window["volume"].sum())

    def atr5(self, now: pd.Timestamp) -> float:
        """5-minute ATR at `now`, never below the daily-ATR floor (if enabled)."""
        i = ind.last_completed(self.m5, now)
        if i < 0:
            return math.nan
        return max(float(self.m5["atr"].iloc[i]), self._atr_floor)

    # -- signals -----------------------------------------------------------
    def trend_score(self, now: pd.Timestamp) -> float:
        """0..100 from the last completed 15-minute bar. 20 points per condition."""
        i = ind.last_completed(self.m15, now)
        if i < 1:
            return math.nan
        b, p = self.m15.iloc[i], self.m15.iloc[i - 1]
        conds = [
            b["close"] > b["ema9"],
            b["ema9"] > b["ema20"],
            b["ema9"] > p["ema9"],
            not math.isnan(b["vwap"]) and b["close"] > b["vwap"],
            b["low"] > p["low"],
        ]
        return 20.0 * sum(bool(c) for c in conds)

    def confirm_5m(self, now: pd.Timestamp, cfg: EntryConfig) -> tuple[bool, dict]:
        """5-minute confirmation: volume, momentum not turning down, not extended."""
        i = ind.last_completed(self.m5, now)
        if i < max(1, cfg.min_vol_bars_5m):
            return False, {"reason": "not enough 5m session bars"}
        b, p = self.m5.iloc[i], self.m5.iloc[i - 1]
        prior = self.m5["volume"].iloc[max(0, i - cfg.vol_lookback_5m): i]   # session bars only
        avg_vol = prior.mean()
        vol_ratio = b["volume"] / avg_vol if avg_vol > 0 else 0.0
        a = self.atr5(now)
        extension = (b["close"] - b["ema9"]) / a if a > 0 else math.inf
        checks = {
            "vol_ratio": round(float(vol_ratio), 2),
            "momentum_up": bool(b["macd_hist"] >= p["macd_hist"]),
            "above_ema9": bool(b["close"] > b["ema9"]),
            "above_vwap": bool(not math.isnan(b["vwap"]) and b["close"] > b["vwap"]),
            "extension_atr": round(float(extension), 2),
        }
        ok = (
            vol_ratio >= cfg.min_vol_ratio_5m
            and checks["momentum_up"] and checks["above_ema9"] and checks["above_vwap"]
            and extension <= cfg.max_extension_atr
        )
        return ok, checks

    def rank_score(self, now: pd.Timestamp) -> float:
        """Intraday re-rank: trend + relative volume + holding the gap."""
        i = self.minute_pos(now)
        if i < 0:
            return -math.inf
        price = float(self.minute["close"].iloc[i])
        minutes_in = max(1.0, (now - now.normalize() - pd.Timedelta(hours=9, minutes=30)).total_seconds() / 60)
        expected = self.avg_volume * min(1.0, minutes_in / 390.0)
        rvol = float(self._session_vol.iloc[i]) / expected if expected > 0 else 0.0
        trend = self.trend_score(now)
        trend = 0.0 if math.isnan(trend) else trend
        holding = (price / self.prev_close - 1.0) * 100.0
        return trend + 10.0 * min(rvol, 5.0) + max(min(holding, 20.0), -20.0)


def position_size(equity: float, entry_price: float, stop_price: float, recent_volume: float,
                  risk: RiskConfig) -> tuple[int, dict]:
    """Shares to buy: the smallest of the risk-based, equity-based and volume-based caps."""
    per_share_risk = entry_price - stop_price
    if per_share_risk <= 0 or entry_price <= 0:
        return 0, {"reason": "bad stop"}
    by_risk = equity * risk.risk_per_trade_pct / 100.0 / per_share_risk
    by_equity = equity * risk.max_position_pct / 100.0 / entry_price
    by_volume = recent_volume * risk.max_participation
    shares = int(max(0.0, min(by_risk, by_equity, by_volume)))
    binding = min((by_risk, "risk"), (by_equity, "equity"), (by_volume, "volume"))[1]
    return shares, {"by_risk": int(by_risk), "by_equity": int(by_equity), "by_volume": int(by_volume),
                    "binding": binding}
