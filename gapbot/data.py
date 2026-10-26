"""Market data: Alpaca historical bars with a local cache, plus a synthetic source.

Conventions used everywhere else in the code:
  * daily frame: columns symbol, date (datetime.date), open, high, low, close, volume
  * minute frame (one symbol, one day): DatetimeIndex of bar START times in
    America/New_York, columns open, high, low, close, volume, from 04:00 to 15:59.
"""
from __future__ import annotations

import datetime as dt
import os
import zlib
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

NY = "America/New_York"
OHLCV = ["open", "high", "low", "close", "volume"]


class DataSource(Protocol):
    def daily(self, start: dt.date, end: dt.date) -> pd.DataFrame: ...
    def minute_many(self, symbols: list[str], day: dt.date) -> dict[str, pd.DataFrame]: ...
    def trading_days(self, start: dt.date, end: dt.date) -> list[dt.date]: ...


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------
class AlpacaSource:
    """Historical data from Alpaca, cached on disk so repeated backtests are fast.

    Needs ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment (paper keys are fine).

    Note on survivorship bias: Alpaca only lists assets that exist today, so stocks
    that were delisted during the test period are missing. That makes results a little
    better than reality. Keep it in mind when reading the report.
    """

    EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}

    def __init__(self, cache_dir: str = "data_cache", feed: str = "sip", symbols: list[str] | None = None):
        from alpaca.data.historical import StockHistoricalDataClient  # imported lazily

        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (see .env.example)")
        self._key, self._secret = key, secret
        self.client = StockHistoricalDataClient(key, secret)
        self.cache = Path(cache_dir)
        (self.cache / "minute").mkdir(parents=True, exist_ok=True)
        self.feed = feed
        self._symbols = symbols

    # -- universe ----------------------------------------------------------
    def universe(self) -> list[str]:
        if self._symbols:
            return self._symbols
        path = self.cache / "universe.csv"
        if path.exists():
            return pd.read_csv(path)["symbol"].tolist()
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        tc = TradingClient(self._key, self._secret, paper=True)
        assets = tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
        syms = sorted(
            a.symbol for a in assets
            if a.tradable and str(getattr(a.exchange, "value", a.exchange)) in self.EXCHANGES
            and a.symbol.isalpha()
        )
        pd.DataFrame({"symbol": syms}).to_csv(path, index=False)
        return syms

    # -- bars --------------------------------------------------------------
    def _bars(self, symbols: list[str], start: dt.datetime, end: dt.datetime, minute: bool) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute if minute else TimeFrame.Day,
            start=start, end=end,
            adjustment=Adjustment.SPLIT,
            feed=DataFeed(self.feed),
        )
        df = self.client.get_stock_bars(req).df
        if df.empty:
            return df
        return df.reset_index()  # columns: symbol, timestamp, open, high, low, close, volume, ...

    def daily(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        # 45 extra calendar days so 20-day averages exist on the first test day
        fetch_start = start - dt.timedelta(days=45)
        path = self.cache / f"daily_{fetch_start}_{end}.csv.gz"
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        syms = self.universe()
        frames = []
        for i in range(0, len(syms), 200):
            chunk = self._bars(
                syms[i:i + 200],
                dt.datetime.combine(fetch_start, dt.time()), dt.datetime.combine(end, dt.time(23, 59)),
                minute=False,
            )
            if not chunk.empty:
                frames.append(chunk)
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY).dt.date
        df = df[["symbol", "date", *OHLCV]].sort_values(["symbol", "date"]).reset_index(drop=True)
        df.to_csv(path, index=False)
        return df

    def minute_many(self, symbols: list[str], day: dt.date) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        missing = []
        for s in symbols:
            p = self.cache / "minute" / s / f"{day}.csv.gz"
            if p.exists():
                out[s] = _read_minute(p)
            else:
                missing.append(s)
        if missing:
            tz = pd.Timestamp(day).tz_localize(NY)
            start = (tz + pd.Timedelta(hours=4)).tz_convert("UTC").to_pydatetime()
            end = (tz + pd.Timedelta(hours=16)).tz_convert("UTC").to_pydatetime()
            raw = self._bars(missing, start, end, minute=True)
            for s in missing:
                part = raw[raw["symbol"] == s] if not raw.empty else raw
                frame = _normalize_minute(part)
                p = self.cache / "minute" / s / f"{day}.csv.gz"
                p.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(p)
                out[s] = frame
        return out

    def benchmark_daily(self, symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        """Daily closes for benchmark ETFs (SPY, IWM), with a few days before `start`
        so the first day's return can be computed. Columns: symbol, date, close."""
        fetch_start = start - dt.timedelta(days=10)
        path = self.cache / f"bench_{'_'.join(symbols)}_{fetch_start}_{end}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        raw = self._bars(symbols, dt.datetime.combine(fetch_start, dt.time()),
                         dt.datetime.combine(end, dt.time(23, 59)), minute=False)
        if raw.empty:
            return pd.DataFrame(columns=["symbol", "date", "close"])
        raw["date"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(NY).dt.date
        df = raw[["symbol", "date", "close"]].sort_values(["symbol", "date"]).reset_index(drop=True)
        df.to_csv(path, index=False)
        return df

    def trading_days(self, start: dt.date, end: dt.date) -> list[dt.date]:
        d = self.daily(start, end)
        days = sorted(set(d["date"]))
        return [x for x in days if start <= x <= end]


def _normalize_minute(part: pd.DataFrame) -> pd.DataFrame:
    if part is None or part.empty:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=NY, name="timestamp"))
    idx = pd.to_datetime(part["timestamp"], utc=True).dt.tz_convert(NY)
    frame = pd.DataFrame(part[OHLCV].to_numpy(), columns=OHLCV, index=pd.DatetimeIndex(idx, name="timestamp"))
    frame = frame.astype(float)
    return frame[(frame.index.time >= dt.time(4)) & (frame.index.time < dt.time(16))].sort_index()


def _read_minute(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(df.index, utc=True).tz_convert(NY)
    df.index = pd.DatetimeIndex(idx, name="timestamp")
    return df[OHLCV].astype(float)


# ---------------------------------------------------------------------------
# Synthetic data (tests and offline demo). Not a model of real markets.
# ---------------------------------------------------------------------------
class SyntheticSource:
    """Random-walk market with a few gappers each day. Deterministic for a given seed.

    Useful to check that the whole pipeline runs and that nothing looks ahead.
    Profits on this data mean nothing.
    """

    def __init__(self, n_symbols: int = 40, seed: int = 7):
        self.n = n_symbols
        self.seed = seed
        self.symbols = [f"SYN{i:03d}" for i in range(n_symbols)]
        self._daily: pd.DataFrame | None = None
        self._minute_cache: dict[tuple[str, dt.date], pd.DataFrame] = {}

    def trading_days(self, start: dt.date, end: dt.date) -> list[dt.date]:
        return [d.date() for d in pd.bdate_range(start, end)]

    def daily(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        if self._daily is not None:
            if not (self._range[0] <= start and end <= self._range[1]):
                raise ValueError(f"SyntheticSource was generated for {self._range}; asked for {start}..{end}")
            return self._daily
        self._range = (start, end)
        rng = np.random.default_rng(self.seed)
        days = [d.date() for d in pd.bdate_range(start - dt.timedelta(days=45), end)]
        rows = []
        for s in self.symbols:
            price = float(rng.uniform(4, 80))
            base_vol = float(rng.uniform(4e5, 6e6))
            for d in days:
                gap = rng.normal(0, 0.01)
                if rng.random() < 0.06:          # occasional big gap
                    gap = rng.uniform(0.04, 0.20) * (1 if rng.random() < 0.75 else -1)
                o = price * (1 + gap)
                drift = rng.normal(0, 0.025)
                c = o * (1 + drift)
                h = max(o, c) * (1 + abs(rng.normal(0, 0.01)))
                low = min(o, c) * (1 - abs(rng.normal(0, 0.01)))
                v = base_vol * (1 + 6 * abs(gap)) * rng.uniform(0.7, 1.3)
                rows.append((s, d, o, h, low, c, v))
                price = c
        self._daily = pd.DataFrame(rows, columns=["symbol", "date", *OHLCV])
        return self._daily

    def benchmark_daily(self, symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        """Random-walk stand-ins for SPY/IWM so the offline demo exercises the report code."""
        rng = np.random.default_rng(self.seed + 1000)
        days = [d.date() for d in pd.bdate_range(start - dt.timedelta(days=10), end)]
        rows = []
        for sym in symbols:
            px = 100.0
            for d in days:
                px *= 1 + rng.normal(0.0003, 0.01)
                rows.append((sym, d, px))
        return pd.DataFrame(rows, columns=["symbol", "date", "close"])

    def minute_many(self, symbols: list[str], day: dt.date) -> dict[str, pd.DataFrame]:
        daily = self._daily if self._daily is not None else self.daily(day, day)
        out = {}
        for s in symbols:
            key = (s, day)
            if key not in self._minute_cache:
                row = daily[(daily["symbol"] == s) & (daily["date"] == day)]
                if row.empty:
                    continue
                r = row.iloc[0]
                self._minute_cache[key] = synth_minutes(day, r["open"], r["close"], r["volume"],
                                                        seed=zlib.crc32(f"{s}|{day}|{self.seed}".encode()))
            out[s] = self._minute_cache[key]
        return out


def synth_minutes(day: dt.date, open_: float, close: float, volume: float, seed: int = 0) -> pd.DataFrame:
    """Brownian bridge from open to close over the regular session, with flat-ish premarket."""
    rng = np.random.default_rng(seed)
    session = pd.date_range(pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30),
                            periods=390, freq="1min")
    pre = pd.date_range(pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(hours=8),
                        periods=90, freq="1min")
    n = len(session)
    steps = rng.normal(0, 0.0015, n)
    path = np.cumsum(steps)
    bridge = path - np.linspace(0, 1, n) * path[-1]
    log_path = np.log(open_) + np.linspace(0, np.log(close / open_), n) + bridge
    closes = np.exp(log_path)
    opens = np.concatenate([[open_], closes[:-1]])
    wig = np.abs(rng.normal(0, 0.0008, n))
    highs = np.maximum(opens, closes) * (1 + wig)
    lows = np.minimum(opens, closes) * (1 - wig)
    shape = 1.5 + np.cos(np.linspace(0, 2 * np.pi, n))          # U-shaped volume
    vols = volume * shape / shape.sum() * rng.uniform(0.6, 1.4, n)
    reg = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=session)
    pre_px = np.full(len(pre), open_) * (1 + rng.normal(0, 0.001, len(pre)))
    pm = pd.DataFrame({"open": pre_px, "high": pre_px * 1.001, "low": pre_px * 0.999, "close": pre_px,
                       "volume": volume * 0.0002}, index=pre)
    df = pd.concat([pm, reg])
    df.index.name = "timestamp"
    return df


def bars_from_rows(day: dt.date, rows: Iterable[tuple], start: str = "09:30") -> pd.DataFrame:
    """Build a minute frame from explicit (open, high, low, close, volume) rows. Used in tests."""
    rows = list(rows)
    idx = pd.date_range(pd.Timestamp(f"{day} {start}").tz_localize(NY), periods=len(rows), freq="1min")
    df = pd.DataFrame(rows, columns=OHLCV, index=idx).astype(float)
    df.index.name = "timestamp"
    return df
