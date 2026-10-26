"""Fill prices and fees. Deliberately pessimistic compared with a paper account."""
from __future__ import annotations

from .config import FillConfig


def participation_pct(shares: float, bar_volume: float) -> float:
    if bar_volume <= 0:
        return 100.0
    return min(100.0, shares / bar_volume * 100.0)


def _cost_bps(cfg: FillConfig, part_pct: float, stop: bool = False) -> float:
    bps = cfg.slippage_bps + cfg.half_spread_bps + cfg.impact_bps_per_1pct_participation * part_pct
    if stop:
        bps += cfg.stop_extra_slippage_bps
    return bps


def market_buy(price: float, shares: float, bar_volume: float, cfg: FillConfig) -> float:
    return price * (1 + _cost_bps(cfg, participation_pct(shares, bar_volume)) / 1e4)


def market_sell(price: float, shares: float, bar_volume: float, cfg: FillConfig) -> float:
    return price * (1 - _cost_bps(cfg, participation_pct(shares, bar_volume)) / 1e4)


def stop_sell(trigger: float, bar_open: float, shares: float, bar_volume: float, cfg: FillConfig) -> float:
    """A sell stop becomes a market order once triggered. If the bar opens below the
    stop (a gap through it) the order starts from the open, not from the stop."""
    base = min(trigger, bar_open)
    return base * (1 - _cost_bps(cfg, participation_pct(shares, bar_volume), stop=True) / 1e4)


def limit_sell_fills(limit: float, bar_open: float, bar_high: float, on_touch: bool = False) -> float | None:
    """A sell limit fills only if price trades THROUGH it (a touch is not a fill, because
    you may be behind other orders in the queue). Returns the fill price or None."""
    if bar_open >= limit if on_touch else bar_open > limit:
        return bar_open
    if bar_high >= limit if on_touch else bar_high > limit:
        return limit
    return None


def fees(side: str, shares: float, price: float, cfg: FillConfig) -> float:
    total = cfg.commission_per_share * shares
    if side == "sell":
        total += min(cfg.finra_taf_per_share * shares, cfg.finra_taf_cap)
        total += cfg.sec_fee_rate * shares * price
    return total
