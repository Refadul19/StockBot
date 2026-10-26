"""Parameters for the Japan research module, in the same style as gapbot/config.py.

Every number the analysis uses is a field here, so a run can be reproduced from its
saved config.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _hhmm(v: Any) -> str:
    """YAML reads an unquoted 15:00 as the number 900 (minutes). Accept both."""
    if isinstance(v, int):
        return f"{v // 60:02d}:{v % 60:02d}"
    return str(v)


def _groups() -> dict[str, dict[str, str]]:
    return {
        "Semiconductors & chip equipment": {
            "6857.T": "Advantest", "8035.T": "Tokyo Electron", "285A.T": "Kioxia Holdings",
            "6723.T": "Renesas Electronics", "6146.T": "Disco", "6920.T": "Lasertec",
            "7735.T": "Screen Holdings"},
        "Electronic components": {
            "6981.T": "Murata Manufacturing", "6762.T": "TDK", "4062.T": "Ibiden"},
        "AI infrastructure": {
            "5803.T": "Fujikura", "5801.T": "Furukawa Electric", "5802.T": "Sumitomo Electric",
            "5016.T": "JX Advanced Metals", "5713.T": "Sumitomo Metal Mining", "3103.T": "Unitika"},
        "AI investment": {"9984.T": "SoftBank Group"},
    }


@dataclass
class UniverseConfig:
    # Group label -> {ticker: company name}. Results are compared by group, so keep the labels.
    groups: dict[str, dict[str, str]] = field(default_factory=_groups)
    signal: str = "^SOX"                   # the overnight US signal that defines the buckets
    context_signals: list[str] = field(default_factory=lambda: ["^GSPC", "^IXIC", "JPY=X"])
    benchmark: str = "1306.T"              # TOPIX ETF, used for buy and hold


@dataclass
class DataConfig:
    months: int = 6                        # history to download, counted back from `end` (3, 12, 60 ...)
    end: str | None = None                 # ISO date; None means today
    lookback_buffer_days: int = 10         # extra days fetched before the period so the first day has a previous close and two US closes
    cache_dir: str = "data_cache/jp"
    cache_max_age_days: int = 1            # re-download a ticker when its cache file is older than this
    request_pause_s: float = 0.5           # polite pause between yfinance downloads
    glitch_ratio: float = 3.0              # drop a row whose price is this many times above or below the 11-day median (bad data)
    glitch_window: int = 11                # trading days in that rolling median


@dataclass
class AnalysisConfig:
    # Bucket edges for the overnight SOX move, in percent. Fixed up front, not tuned.
    bucket_edges_pct: list[float] = field(default_factory=lambda: [-2.0, -1.0, 1.0, 2.0])
    min_bucket_days: int = 10              # buckets with fewer stock-days are marked "too few days, not reliable"
    out_of_sample_fraction: float = 0.3    # last 30% of days are held out, as in gapbot
    extreme_sox_pct: float = 4.0           # a SOX move bigger than this, up or down, marks the day "extreme"
    outlier_warn_pct: float = 25.0         # log a warning for any gap or intraday move bigger than this (likely a data error)
    min_days_for_ranking: int = 30         # stocks with fewer paired days are left out of the ranking


@dataclass
class CostConfig:
    cost_per_side_pct: float = 0.1         # commission + slippage, paid on the buy and again on the sell
    lot_size: int = 100                    # Japan's trading unit


@dataclass
class RulesConfig:
    """Two simple long-only rules, fixed before looking at results:
      follow: buy the open and sell the close when SOX rose at least threshold_pct overnight
      dip:    buy the open and sell the close when SOX fell at least threshold_pct overnight
    The main result is in percent per trade, with no budget or lot limit. The budget and capital
    below only drive the separate 'feasibility' view (what a real account could buy)."""
    threshold_pct: float = 1.0
    budget_per_trade_yen: float = 500_000.0   # yen per trade; one 100-share lot must fit inside it
    capital_yen: float = 3_000_000.0          # your account size; sets how many trades can be open at once


@dataclass
class CrossCheckConfig:
    proxies: list[str] = field(default_factory=lambda: ["SOXX", "SMH"])  # US chip ETFs, a second view of SOX
    min_sox_move_pct: float = 5.0          # only SOX days moving more than this are cross-checked
    tolerance_pp: float = 2.0              # an ETF within this many percentage points counts as a match


@dataclass
class EarningsConfig:
    """Days a stock moves on its own results, not on the US overnight move, are marked and
    (by default) left out of the main results.

    Most Japanese companies report after the 15:30 close, so the reaction shows at the NEXT
    day's open. That is why an after-close or unknown-time report marks two days."""
    exclude: bool = True                   # False (or --include-earnings) puts earnings days back in
    manual_file: str = "config/jp_earnings.csv"   # ticker,date,time,note ; overrides yfinance
    use_yfinance: bool = True
    yfinance_limit: int = 60               # how many past and upcoming report dates to ask yfinance for
    trust_yfinance_times: bool = False     # Yahoo's times for Japan are mostly placeholders; False = treat as unknown
    cache_max_age_days: int = 7
    unknown_time_as: str = "after_close"   # "after_close" or "during_hours": what to assume when no time is known
    after_close_from: str = "15:00"        # Japan time; a given time at or after this counts as after close (15:00 was the close until Nov 2024)
    mark_announcement_day: bool = True     # after close: mark the day of the report itself
    mark_next_day_after_close: bool = True # after close: also mark the next trading day, where the reaction shows
    override_window_days: int = 7          # a manual date replaces any yfinance date this close to it
    max_gap_days: int = 120                # a longer silence between two report dates is logged as uncertain data

    def __post_init__(self) -> None:
        if self.unknown_time_as not in ("after_close", "during_hours"):
            raise ValueError("earnings.unknown_time_as must be after_close or during_hours")
        self.after_close_from = _hhmm(self.after_close_from)


@dataclass
class PlanConfig:
    """Settings for the web dashboard's 'reference levels'. These are descriptions of how far this
    stock usually moves from the open, NOT predictions, and they are fixed here before any
    results are seen. Levels come from in-sample days only; the last part is the honest test."""
    target_percentile: float = 50.0        # target = this percentile of past open-to-high moves (higher = further away)
    stop_percentile: float = 50.0          # stop = this percentile of past open-to-low moves
    min_trades_for_verdict: int = 30       # fewer out-of-sample trades than this = "too few to judge"
    stale_days: int = 4                    # warn when the latest SOX session is older than this many calendar days
    earnings_warn_days: int = 3            # warn when a report date falls within this many days ahead
    min_history_days: int = 120            # refuse tickers with fewer usable paired days than this


@dataclass
class OutputConfig:
    out_dir: str = "runs/jp"


@dataclass
class JpConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    crosscheck: CrossCheckConfig = field(default_factory=CrossCheckConfig)
    earnings: EarningsConfig = field(default_factory=EarningsConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "JpConfig":
        d = d or {}
        section_cls = {f.name: f.default_factory for f in fields(cls)}
        extra = set(d) - set(section_cls)
        if extra:
            raise ValueError(f"Unknown config sections: {sorted(extra)}")
        kwargs = {}
        for name, factory in section_cls.items():
            values = d.get(name, {}) or {}
            known = {f.name for f in fields(factory)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"Unknown keys in [{name}]: {sorted(unknown)}")
            kwargs[name] = factory(**values)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None) -> "JpConfig":
        if path is None:
            return cls()
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))
