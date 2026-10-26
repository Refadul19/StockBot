"""Minute-by-minute portfolio backtest.

Timing model (the most important part of any backtest):
  * For each 1-minute bar starting at t, the decision moment is now = t + 1 minute.
  * Market orders decided at `now` fill at the OPEN of the next bar (the one starting at now).
  * Resting stop and target orders are checked against each bar's high/low.
    If one bar touches both the stop and the target, the stop is assumed to fill first.
  * Stops raised after a bar (profit locks, trailing) only apply from the following bar.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import pandas as pd

from . import fills
from .config import Config
from .data import NY, DataSource
from .scanner import scan_gappers
from .strategy import SymbolDay, position_size

ONE_MIN = pd.Timedelta(minutes=1)


def _t(hhmm: str) -> dt.time:
    h, m = hhmm.split(":")
    return dt.time(int(h), int(m))


@dataclass
class PendingEntry:
    symbol: str
    decided_at: pd.Timestamp
    signal_price: float
    limit_price: float
    stop: float
    shares: int
    info: dict


@dataclass
class Position:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    initial_shares: int
    shares: int
    initial_stop: float
    stop: float
    risk_per_share: float
    target: float
    highest: float
    entry_fees: float
    info: dict
    stop_kind: str = "stop"
    target_done: bool = False
    time_checked: bool = False
    pending_exit: str | None = None
    exits: list[dict] = field(default_factory=list)

    def r_multiple(self, price: float) -> float:
        return (price - self.entry_price) / self.risk_per_share


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    fills: pd.DataFrame
    daily: pd.DataFrame
    equity_curve: pd.Series
    config: Config
    skipped: dict


class Backtester:
    def __init__(self, cfg: Config, source: DataSource):
        self.cfg = cfg
        self.source = source

    # ------------------------------------------------------------------
    def run(self, start: dt.date, end: dt.date, progress: bool = False,
            only_days: list[dt.date] | None = None) -> BacktestResult:
        """Backtest every trading day in [start, end], or only `only_days` within it."""
        cfg = self.cfg
        daily = self.source.daily(start, end)
        days = self.source.trading_days(start, end)
        if only_days is not None:
            keep = set(only_days)
            days = [d for d in days if d in keep]
        self.cash = cfg.backtest.initial_equity
        self.trades: list[dict] = []
        self.fill_log: list[dict] = []
        self.daily_rows: list[dict] = []
        self.curve: list[tuple[pd.Timestamp, float]] = []
        self.skipped = {"limit_missed": 0, "no_cash": 0, "volume_cap_zero": 0, "halted_days": 0}
        for n, day in enumerate(days, 1):
            self._run_day(daily, day)
            if progress:
                print(f"  {day}  equity {self.cash:,.0f}  ({n}/{len(days)})", flush=True)
        return BacktestResult(
            trades=pd.DataFrame(self.trades),
            fills=pd.DataFrame(self.fill_log),
            daily=pd.DataFrame(self.daily_rows),
            equity_curve=pd.Series({t: v for t, v in self.curve}, dtype=float),
            config=cfg,
            skipped=self.skipped,
        )

    # ------------------------------------------------------------------
    def _run_day(self, daily: pd.DataFrame, day: dt.date) -> None:
        cfg = self.cfg
        start_equity = self.cash
        cands = scan_gappers(daily, day, cfg.scanner)
        minute = self.source.minute_many(cands["symbol"].tolist(), day) if not cands.empty else {}
        sds: dict[str, SymbolDay] = {}
        bars: dict[str, dict[pd.Timestamp, tuple]] = {}
        for r in cands.itertuples():
            m = minute.get(r.symbol)
            if m is None or m.empty:
                continue
            sds[r.symbol] = SymbolDay(r.symbol, m, prev_close=r.prev_close, avg_volume=r.avg_volume,
                                      gap_pct=r.gap_pct, daily_atr=r.daily_atr, exit_cfg=cfg.exit)
            reg = m[m.index.time >= dt.time(9, 30)]
            bars[r.symbol] = {ts: row for ts, row in zip(reg.index, reg[["open", "high", "low", "close", "volume"]].itertuples(index=False, name=None))}

        positions: dict[str, Position] = {}
        pending: dict[str, PendingEntry] = {}
        entries_today: dict[str, int] = {}
        cooldown_until: dict[str, pd.Timestamp] = {}
        last_close: dict[str, float] = {}
        watchlist: list[str] = []
        halted = False
        trades_before = len(self.trades)

        open_ts = pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)
        flatten_t = _t(cfg.exit.flatten_at)
        entry_from, entry_to = _t(cfg.entry.no_entry_before), _t(cfg.entry.no_entry_after)

        for k in range(390):
            t = open_ts + k * ONE_MIN
            now = t + ONE_MIN

            # a) market orders decided last minute fill at this bar's open
            for sym in list(pending):
                pe = pending[sym]
                bar = bars[sym].get(t)
                if bar is None:
                    if t - pe.decided_at >= pd.Timedelta(minutes=3):
                        del pending[sym]
                    continue
                del pending[sym]
                self._fill_entry(pe, bar, t, positions, entries_today)

            for sym, pos in list(positions.items()):
                bar = bars[sym].get(t)
                if bar is None:
                    continue
                if pos.pending_exit:
                    px = fills.market_sell(bar[0], pos.shares, bar[4], cfg.fill)
                    self._sell(pos, pos.shares, px, t, pos.pending_exit, ref=bar[0])
                else:
                    self._check_resting(pos, bar, t)
                if pos.shares == 0:
                    self._close(pos, positions, cooldown_until, t)

            # b) update state with the bar that just completed
            for sym, pos in positions.items():
                bar = bars[sym].get(t)
                if bar is None:
                    continue
                pos.highest = max(pos.highest, bar[1])
                best_r = pos.r_multiple(pos.highest)
                for at_r, lock_r in cfg.exit.profit_locks:
                    new = pos.entry_price + lock_r * pos.risk_per_share
                    if best_r >= at_r and new > pos.stop:
                        pos.stop, pos.stop_kind = new, "profit_lock"
                if pos.target_done:
                    a = sds[sym].atr5(now)
                    if not math.isnan(a):
                        trail = pos.highest - cfg.exit.trail_atr_mult * a
                        if trail > pos.stop:
                            pos.stop, pos.stop_kind = trail, "trail"
            for sym, b in bars.items():
                if t in b:
                    last_close[sym] = b[t][3]

            equity = self.cash + sum(p.shares * last_close.get(s, p.entry_price) for s, p in positions.items())
            self.curve.append((now, equity))
            if not halted and equity <= start_equity * (1 - cfg.risk.daily_loss_limit_pct / 100):
                halted = True
                self.skipped["halted_days"] += 1

            # c) decisions at `now`
            if now.time() >= flatten_t:
                pending.clear()
                for pos in positions.values():
                    pos.pending_exit = pos.pending_exit or "eod"
                continue

            for sym, pos in positions.items():
                if pos.pending_exit or pos.target_done or pos.time_checked:
                    continue
                if now - pos.entry_time >= pd.Timedelta(minutes=cfg.exit.time_stop_min):
                    pos.time_checked = True
                    if pos.r_multiple(last_close[sym]) < cfg.exit.time_stop_min_progress_r:
                        pos.pending_exit = "time_stop"

            minutes_in = k + 1
            if minutes_in % cfg.scanner.rescan_minutes == 0:
                ranked = sorted(sds, key=lambda s: sds[s].rank_score(now), reverse=True)
                watchlist = ranked[: cfg.scanner.active_watchlist]

            if halted or now.minute % 5 != 0 or not (entry_from <= now.time() <= entry_to):
                continue
            for sym in watchlist:
                if len(positions) + len(pending) >= cfg.risk.max_open_positions:
                    break
                if sym in positions or sym in pending:
                    continue
                if entries_today.get(sym, 0) >= cfg.entry.max_entries_per_symbol:
                    continue
                if sym in cooldown_until and now < cooldown_until[sym]:
                    continue
                sd = sds[sym]
                score = sd.trend_score(now)
                if math.isnan(score) or score < cfg.entry.min_trend_score:
                    continue
                ok, checks = sd.confirm_5m(now, cfg.entry)
                if not ok:
                    continue
                price, a = sd.last_price(now), sd.atr5(now)
                if math.isnan(price) or math.isnan(a) or a <= 0:
                    continue
                stop = price - cfg.exit.stop_atr_mult * a
                shares, size_info = position_size(equity, price, stop,
                                                  sd.recent_volume(now, cfg.risk.participation_window_min), cfg.risk)
                if shares < 1:
                    self.skipped["volume_cap_zero"] += 1
                    continue
                pending[sym] = PendingEntry(
                    sym, now, price, price * (1 + cfg.entry.entry_limit_offset_pct / 100), stop, shares,
                    {"trend_score": score, **checks, **size_info, "rank": watchlist.index(sym) + 1,
                     "gap_pct": round(sd.gap_pct, 2)},
                )

        # anything still open (no bar after the flatten time): exit at the last price, stop-style costs
        for sym, pos in list(positions.items()):
            px = fills.stop_sell(last_close[sym], last_close[sym], pos.shares, 0, self.cfg.fill)
            self._sell(pos, pos.shares, px, open_ts + 390 * ONE_MIN, "eod_forced", ref=last_close[sym])
            self._close(pos, positions, cooldown_until, open_ts + 390 * ONE_MIN)

        day_trades = self.trades[trades_before:]
        self.daily_rows.append({
            "date": day, "start_equity": start_equity, "end_equity": self.cash,
            "pnl": self.cash - start_equity, "return_pct": (self.cash / start_equity - 1) * 100,
            "trades": len(day_trades), "wins": sum(1 for x in day_trades if x["pnl"] > 0),
            "candidates": len(sds), "halted": halted,
        })

    # ------------------------------------------------------------------
    def _fill_entry(self, pe: PendingEntry, bar: tuple, t: pd.Timestamp,
                    positions: dict, entries_today: dict) -> None:
        cfg = self.cfg
        o, v = bar[0], bar[4]
        if o > pe.limit_price:
            self.skipped["limit_missed"] += 1
            return
        price = min(fills.market_buy(o, pe.shares, v, cfg.fill), pe.limit_price)
        shares = pe.shares
        fee = fills.fees("buy", shares, price, cfg.fill)
        if shares * price + fee > self.cash:
            shares = int((self.cash - fee) / price)
            if shares < 1:
                self.skipped["no_cash"] += 1
                return
        r = price - pe.stop
        if r <= 0:                       # opened at or below the planned stop: skip
            return
        self.cash -= shares * price + fee
        positions[pe.symbol] = Position(
            symbol=pe.symbol, entry_time=t, entry_price=price, initial_shares=shares, shares=shares,
            initial_stop=pe.stop, stop=pe.stop, risk_per_share=r,
            target=price + self.cfg.exit.target_r * r, highest=price, entry_fees=fee,
            info={**pe.info, "signal_price": round(pe.signal_price, 4), "decided_at": str(pe.decided_at)},
        )
        entries_today[pe.symbol] = entries_today.get(pe.symbol, 0) + 1
        self.fill_log.append({"time": t, "symbol": pe.symbol, "side": "buy", "shares": shares, "price": price,
                              "reference": o, "fees": fee, "reason": "entry"})

    def _check_resting(self, pos: Position, bar: tuple, t: pd.Timestamp) -> None:
        o, h, low, c, v = bar
        if low <= pos.stop or o <= pos.stop:
            px = fills.stop_sell(pos.stop, o, pos.shares, v, self.cfg.fill)
            self._sell(pos, pos.shares, px, t, pos.stop_kind, ref=min(pos.stop, o))
            return
        if not pos.target_done:
            px = fills.limit_sell_fills(pos.target, o, h, self.cfg.fill.limit_fills_on_touch)
            if px is not None:
                qty = min(pos.shares, max(1, round(pos.initial_shares * self.cfg.exit.target_fraction)))
                self._sell(pos, qty, px, t, "target", ref=px)
                pos.target_done = True

    def _sell(self, pos: Position, qty: int, price: float, t: pd.Timestamp, reason: str, ref: float) -> None:
        fee = fills.fees("sell", qty, price, self.cfg.fill)
        self.cash += qty * price - fee
        pos.shares -= qty
        pos.exits.append({"time": t, "shares": qty, "price": price, "fees": fee, "reason": reason})
        self.fill_log.append({"time": t, "symbol": pos.symbol, "side": "sell", "shares": qty, "price": price,
                              "reference": ref, "fees": fee, "reason": reason})

    def _close(self, pos: Position, positions: dict, cooldown_until: dict, t: pd.Timestamp) -> None:
        proceeds = sum(e["shares"] * e["price"] for e in pos.exits)
        exit_fees = sum(e["fees"] for e in pos.exits)
        cost = pos.initial_shares * pos.entry_price
        pnl = proceeds - cost - exit_fees - pos.entry_fees
        avg_exit = proceeds / pos.initial_shares
        last = pos.exits[-1]
        self.trades.append({
            "symbol": pos.symbol, "entry_time": pos.entry_time, "exit_time": last["time"],
            "shares": pos.initial_shares, "entry_price": round(pos.entry_price, 4),
            "avg_exit_price": round(avg_exit, 4), "initial_stop": round(pos.initial_stop, 4),
            "target": round(pos.target, 4), "pnl": round(pnl, 2),
            "r_multiple": round(pnl / (pos.risk_per_share * pos.initial_shares), 3),
            "fees": round(exit_fees + pos.entry_fees, 2),
            "minutes_held": int((last["time"] - pos.entry_time).total_seconds() // 60) + 1,
            "exit_reasons": "+".join(e["reason"] for e in pos.exits),
            "final_exit": last["reason"],
            **{f"sig_{k}": v for k, v in pos.info.items()},
        })
        del positions[pos.symbol]
        wait = self.cfg.entry.cooldown_after_win_min if pnl > 0 else self.cfg.entry.cooldown_after_loss_min
        cooldown_until[pos.symbol] = t + pd.Timedelta(minutes=wait)
