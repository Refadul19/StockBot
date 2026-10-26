"""Command line entry point.

  python -m gapbot backtest --start 2025-01-02 --end 2025-12-31
  python -m gapbot backtest --synthetic --start 2025-03-03 --end 2025-04-30
  python -m gapbot backtest ... --compare-optimistic
  python -m gapbot sweep --start ... --end ... --param exit.target_r=1.5,2,3 --param exit.stop_atr_mult=1,1.5
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
from pathlib import Path

import pandas as pd

from .config import Config, FillConfig
from .data import AlpacaSource, SyntheticSource
from .engine import Backtester
from .metrics import benchmark_daily_returns, slippage_cost, split_days, summarize_split

BENCHMARKS = ["SPY", "IWM"]


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _load_env(path: str = ".env") -> None:
    import os
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _source(args, cfg: Config):
    if args.synthetic:
        return SyntheticSource(n_symbols=args.synthetic_symbols, seed=args.seed)
    _load_env()
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    return AlpacaSource(cfg.backtest.cache_dir, cfg.backtest.data_feed, symbols)


def _parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def _print_summary(title: str, split: dict) -> None:
    print(f"\n=== {title} ===")
    keys = ["period", "days", "trades", "total_return_pct", "win_rate_pct", "profit_factor", "expectancy_r",
            "t_stat_mean_r", "max_drawdown_pct", "sharpe_annualized", "green_days_pct", "total_fees",
            "SPY_return_pct", "IWM_return_pct", "corr_SPY", "corr_IWM"]
    cols = [c for c in ("in_sample", "out_of_sample", "all") if c in split]
    print(f"{'':22}" + "".join(f"{c:>26}" for c in cols))
    for k in keys:
        print(f"{k:22}" + "".join(f"{str(split[c].get(k, '')):>26}" for c in cols))


def write_report(result, split: dict, out_dir: Path, extra: dict | None = None,
                 bench: pd.DataFrame | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out_dir / "trades.csv", index=False)
    result.fills.to_csv(out_dir / "fills.csv", index=False)
    result.daily.to_csv(out_dir / "daily.csv", index=False)
    result.equity_curve.rename("equity").to_csv(out_dir / "equity_curve.csv", index_label="time")
    result.config.save(out_dir / "config.yaml")
    summary = {"metrics": split, "skipped_orders": result.skipped,
               "slippage_cost": round(slippage_cost(result.fills), 2), **(extra or {})}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = ["# Backtest report", ""]
    for name, m in split.items():
        lines += [f"## {name.replace('_', ' ').title()} ({m.get('period', '')})", ""]
        lines += [f"- **{k}**: {v}" for k, v in m.items() if k != "period"]
        lines.append("")
    lines += ["## Costs", "", f"- Slippage and spread paid: ${summary['slippage_cost']:,.2f}",
              f"- Orders skipped: {result.skipped}", ""]
    if extra and "optimistic" in extra:
        o = extra["optimistic"]["all"]
        lines += ["## Same strategy with paper-account style fills", "",
                  f"- Return: {o.get('total_return_pct')}% vs {split['all'].get('total_return_pct')}% with realistic fills",
                  f"- Win rate: {o.get('win_rate_pct')}% vs {split['all'].get('win_rate_pct')}%",
                  "- The gap between these two is roughly what a paper account overstates.", ""]
    lines += ["## How to read this", "",
              "- Judge the strategy on the **out-of-sample** column. In-sample numbers were seen while tuning.",
              "- `t_stat_mean_r` under about 2 means the average trade is not clearly different from zero.",
              "- `expectancy_r` is average profit per trade in units of initial risk. Above 0.1R after costs is good.",
              "- `SPY_return_pct` / `IWM_return_pct` are buy-and-hold returns over the same days. A strategy that only",
              "  matches IWM is taking market risk, not finding an edge. `corr_*` near 1 says the same thing.",
              "- Survivorship bias: delisted stocks are missing from Alpaca's asset list, so real results are a bit worse.",
              ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        d = result.daily
        ax.plot(pd.to_datetime(d["date"]), d["end_equity"], color="#1f6feb", label="strategy")
        rets = benchmark_daily_returns(bench)
        for sym, color in zip(rets.columns, ["#d29922", "#8957e5"]):
            r = rets[sym].reindex(d["date"]).fillna(0) / 100
            ax.plot(pd.to_datetime(d["date"]), d["start_equity"].iloc[0] * (1 + r).cumprod().to_numpy(),
                    color=color, alpha=0.8, label=f"{sym} buy and hold")
        if "out_of_sample" in split:
            oos_start = split["out_of_sample"]["period"].split(" to ")[0]
            ax.axvline(pd.to_datetime(oos_start), color="#888", linestyle="--", label="out-of-sample starts")
        ax.legend()
        ax.set_title("End-of-day equity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "equity.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass


def cmd_backtest(args) -> None:
    cfg = Config.load(args.config)
    if args.equity:
        cfg = cfg.with_overrides({"backtest.initial_equity": args.equity})
    src = _source(args, cfg)
    print(f"Running {args.start} to {args.end} ...")
    result = Backtester(cfg, src).run(args.start, args.end, progress=args.progress)
    bench = None
    if hasattr(src, "benchmark_daily"):
        try:
            bench = src.benchmark_daily(BENCHMARKS, args.start, args.end)
        except Exception as e:  # a benchmark failure should not lose the backtest
            print(f"Benchmark download failed ({e}); continuing without it.")
    split = summarize_split(result, cfg.backtest.out_of_sample_fraction, bench)
    extra = {}
    if args.compare_optimistic:
        opt_cfg = Config.from_dict({**cfg.to_dict(), "fill": FillConfig.optimistic().__dict__})
        opt = Backtester(opt_cfg, src).run(args.start, args.end)
        extra["optimistic"] = summarize_split(opt, cfg.backtest.out_of_sample_fraction, bench)
    _print_summary("Realistic fills", split)
    if extra:
        _print_summary("Paper-style fills (for comparison only)", extra["optimistic"])
    out = Path(cfg.backtest.out_dir) / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_report(result, split, out, extra, bench)
    print(f"\nReport, trade journal and fills written to {out}/")


def cmd_sweep(args) -> None:
    """Grid search on the in-sample days only, then show how the best settings did out of sample."""
    base = Config.load(args.config)
    src = _source(args, base)
    grid = {}
    for p in args.param:
        name, values = p.split("=", 1)
        grid[name] = [_parse_value(v) for v in values.split(",")]
    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    days = src.trading_days(args.start, args.end)
    is_days, oos_days = split_days(days, base.backtest.out_of_sample_fraction)
    if not oos_days:
        raise SystemExit("Need an out-of-sample period; set backtest.out_of_sample_fraction > 0")
    print(f"{len(combos)} combinations on {len(is_days)} in-sample days, then {len(oos_days)} out-of-sample days")
    rows = []
    for i, combo in enumerate(combos, 1):
        cfg = base.with_overrides(combo)
        r = Backtester(cfg, src).run(args.start, args.end, only_days=is_days)
        s = summarize_split(r, 0.0)["all"]
        rows.append({**combo, "is_return_pct": s.get("total_return_pct"), "is_expectancy_r": s.get("expectancy_r"),
                     "is_trades": s.get("trades"), "is_max_dd_pct": s.get("max_drawdown_pct")})
        print(f"  [{i}/{len(combos)}] {combo} -> {s.get('total_return_pct')}% in-sample")
    table = pd.DataFrame(rows).sort_values("is_expectancy_r", ascending=False)
    top = table.head(args.top)
    oos = []
    for _, row in top.iterrows():
        combo = {k: row[k] for k in grid}
        combo = {k: (v.item() if hasattr(v, "item") else v) for k, v in combo.items()}
        r = Backtester(base.with_overrides(combo), src).run(args.start, args.end, only_days=oos_days)
        s = summarize_split(r, 0.0)["all"]
        oos.append({"oos_return_pct": s.get("total_return_pct"), "oos_expectancy_r": s.get("expectancy_r"),
                    "oos_trades": s.get("trades")})
    top = pd.concat([top.reset_index(drop=True), pd.DataFrame(oos)], axis=1)
    print("\nBest in-sample settings and how they held up out of sample:")
    print(top.to_string(index=False))
    out = Path(base.backtest.out_dir) / ("sweep_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "in_sample_grid.csv", index=False)
    top.to_csv(out / "top_with_out_of_sample.csv", index=False)
    print(f"\nWritten to {out}/")
    print("If out-of-sample results are much worse than in-sample, the settings are overfit.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gapbot", description="Gap-and-trend day trading backtester")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--start", type=_date, required=True)
        p.add_argument("--end", type=_date, required=True)
        p.add_argument("--config", default=None, help="YAML file, see config.example.yaml")
        p.add_argument("--symbols", default=None, help="Comma list to limit the universe (faster first runs)")
        p.add_argument("--synthetic", action="store_true", help="Use random synthetic data, no API keys needed")
        p.add_argument("--synthetic-symbols", type=int, default=40)
        p.add_argument("--seed", type=int, default=7)

    b = sub.add_parser("backtest")
    common(b)
    b.add_argument("--equity", type=float, default=None)
    b.add_argument("--compare-optimistic", action="store_true", help="Also run with paper-account style fills")
    b.add_argument("--progress", action="store_true")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("sweep")
    common(s)
    s.add_argument("--param", action="append", required=True, help="e.g. exit.target_r=1.5,2,3")
    s.add_argument("--top", type=int, default=5)
    s.set_defaults(func=cmd_sweep)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
