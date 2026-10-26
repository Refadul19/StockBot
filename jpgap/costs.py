"""Trading costs and Japan's 100-share unit.

A trade here is: buy at the open, sell at the close of the same Tokyo day. The cost
percentage is paid on each side, as a worse price (buy higher, sell lower).
"""
from __future__ import annotations

import math


def net_return(gross: float, cost_per_side_pct: float) -> float:
    """Round-trip return after paying the cost on both sides. `gross` is close/open - 1."""
    c = cost_per_side_pct / 100.0
    return (1.0 + gross) * (1.0 - c) / (1.0 + c) - 1.0


def lot_shares(price: float, budget_jpy: float, lot_size: int) -> int:
    """Whole lots only. Returns 0 when one lot costs more than the budget."""
    if not (price > 0) or math.isnan(price):
        return 0
    return int(budget_jpy // (price * lot_size)) * lot_size


def trade_pnl_jpy(open_px: float, close_px: float, budget_jpy: float, cost_pct: float, lot_size: int) -> tuple[int, float]:
    """(shares, profit in yen after costs). Shares is 0 when a single lot is unaffordable."""
    shares = lot_shares(open_px, budget_jpy, lot_size)
    if shares == 0:
        return 0, 0.0
    c = cost_pct / 100.0
    pnl = shares * (close_px * (1.0 - c) - open_px * (1.0 + c))
    return shares, pnl
