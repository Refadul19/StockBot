"""All strategy and simulation parameters live here.

Every number the strategy uses should be a field in one of these dataclasses,
so a backtest run can be reproduced exactly from its saved config.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _hhmm(v: Any) -> str:
    """YAML reads an unquoted 10:00 as the number 600 (minutes). Accept both."""
    if isinstance(v, int):
        return f"{v // 60:02d}:{v % 60:02d}"
    return str(v)


@dataclass
class ScannerConfig:
    min_price: float = 2.0                 # skip sub-$2 stocks
    max_price: float = 200.0
    min_gap_pct: float = 4.0               # open vs prior close, percent
    min_avg_dollar_volume: float = 5_000_000   # 20-day average daily $ volume
    min_avg_volume: float = 500_000            # 20-day average daily shares
    max_candidates: int = 15               # top N gappers loaded each day
    rescan_minutes: int = 5                # intraday re-rank interval
    active_watchlist: int = 6              # only the top N after re-rank may be entered


@dataclass
class EntryConfig:
    no_entry_before: str = "10:00"         # New York time; the 15m trend score needs two session bars, so 10:00 is the earliest it can fire
    no_entry_after: str = "15:00"
    min_trend_score: float = 60.0          # 15-minute trend score, 0..100
    vol_lookback_5m: int = 6               # regular-session 5m bars in the volume average
    min_vol_bars_5m: int = 3               # need at least this many earlier session bars to judge volume
    min_vol_ratio_5m: float = 1.2          # last 5m volume / average
    max_extension_atr: float = 2.0         # skip if price is this many 5m ATRs above 5m EMA9
    cooldown_after_win_min: int = 30       # no re-entry in a symbol right after a winning exit
    cooldown_after_loss_min: int = 15
    max_entries_per_symbol: int = 2
    entry_limit_offset_pct: float = 0.3    # marketable limit: skip the fill if the next open is this far above the signal price

    def __post_init__(self) -> None:
        self.no_entry_before = _hhmm(self.no_entry_before)
        self.no_entry_after = _hhmm(self.no_entry_after)


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5        # % of equity lost if the initial stop is hit
    max_position_pct: float = 10.0         # % of equity in one position
    max_open_positions: int = 5
    max_participation: float = 0.05        # order <= 5% of the stock's recent per-minute volume x participation window
    participation_window_min: int = 5
    daily_loss_limit_pct: float = 2.0      # stop opening trades after this daily drawdown


@dataclass
class ExitConfig:
    atr_period: int = 14                   # on regular-session 5-minute bars
    # Early in the day the 5m ATR has only a few bars behind it. When True, it is never
    # allowed below the 14-day daily ATR scaled to one 5-minute bar (daily ATR / sqrt(78)).
    atr_floor_from_daily: bool = True
    stop_atr_mult: float = 1.5
    target_r: float = 2.0                  # first target, in multiples of initial risk (R)
    target_fraction: float = 0.75          # sell this fraction at the target
    trail_atr_mult: float = 1.5            # trail the rest from the highest high
    # profit locks: when the open gain reaches `at_r`, raise the stop to entry + `lock_r` * R
    profit_locks: list[list[float]] = field(default_factory=lambda: [[1.0, 0.0], [1.5, 0.5]])
    time_stop_min: int = 20
    time_stop_min_progress_r: float = 0.3  # exit at the time stop unless the trade is at least this far up
    flatten_at: str = "15:55"

    def __post_init__(self) -> None:
        self.flatten_at = _hhmm(self.flatten_at)


@dataclass
class FillConfig:
    """How pessimistic the simulator is. Paper trading is usually far too kind."""
    slippage_bps: float = 5.0              # paid on every market fill
    half_spread_bps: float = 5.0           # crossing half the spread on market orders
    impact_bps_per_1pct_participation: float = 2.0  # extra cost per 1% of the bar's volume
    stop_extra_slippage_bps: float = 10.0  # stops trigger in fast markets and fill worse
    finra_taf_per_share: float = 0.000166  # sells only, capped below
    finra_taf_cap: float = 8.30
    sec_fee_rate: float = 0.0000278        # sells only, x notional; check the current SEC rate
    commission_per_share: float = 0.0      # Alpaca charges no commission on US stocks
    limit_fills_on_touch: bool = False     # True = paper-account style (a touch fills your limit)

    @classmethod
    def optimistic(cls) -> "FillConfig":
        """Roughly how a paper account fills you. Use only to compare against the realistic run."""
        return cls(slippage_bps=0, half_spread_bps=0, impact_bps_per_1pct_participation=0,
                   stop_extra_slippage_bps=0, limit_fills_on_touch=True)


@dataclass
class BacktestConfig:
    initial_equity: float = 100_000.0
    data_feed: str = "sip"                 # "sip" or "iex"
    cache_dir: str = "data_cache"
    out_dir: str = "runs"
    out_of_sample_fraction: float = 0.3    # last 30% of days are held out


@dataclass
class Config:
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    fill: FillConfig = field(default_factory=FillConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Config":
        d = d or {}
        sections = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for name in sections:
            section_cls = {
                "scanner": ScannerConfig, "entry": EntryConfig, "risk": RiskConfig,
                "exit": ExitConfig, "fill": FillConfig, "backtest": BacktestConfig,
            }[name]
            values = d.get(name, {}) or {}
            known = {f.name for f in fields(section_cls)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"Unknown keys in [{name}]: {sorted(unknown)}")
            kwargs[name] = section_cls(**values)
        extra = set(d) - set(sections)
        if extra:
            raise ValueError(f"Unknown config sections: {sorted(extra)}")
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """overrides like {"exit.target_r": 2.5}"""
        d = self.to_dict()
        for key, value in overrides.items():
            section, name = key.split(".", 1)
            if section not in d or name not in d[section]:
                raise ValueError(f"Unknown parameter {key}")
            d[section][name] = value
        return Config.from_dict(d)
