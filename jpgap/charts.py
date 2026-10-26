"""PNG charts. The first slots of the validated reference palette (blue, orange, aqua, yellow) are used in order, with a legend and direct value labels so colour is never the only cue."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
PALETTE = [BLUE, ORANGE, "#1baf7a", "#eda100"]   # reference palette slots 1 to 4
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bucket_chart(table: pd.DataFrame, title: str, path: Path) -> None:
    """Left: average gap vs average intraday move per SOX bucket. Right: % of days continued."""
    t = table.drop(index="all days")
    x = np.arange(len(t))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    w = 0.38
    for ax in (a1, a2):
        _style(ax)
    for off, col, colr, name in ((-w / 2, "avg_gap_pct", BLUE, "Average gap at the open"),
                                 (w / 2, "avg_intraday_pct", ORANGE, "Average open-to-close move")):
        vals = t[col].astype(float)
        bars = a1.bar(x + off, vals.fillna(0), w - 0.04, color=colr, label=name)
        for b, v in zip(bars, vals):
            if pd.notna(v):
                a1.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=8, color=INK,
                            xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
    a1.axhline(0, color=MUTED, linewidth=0.8)
    a1.set_ylabel("percent", color=MUTED)
    a1.set_title("Gap vs same-day move", loc="left", color=INK, fontsize=11)
    a1.legend(frameon=False, fontsize=8, loc="best")
    cont = t["pct_continued"].astype(float)
    bars = a2.bar(x, cont.fillna(0), 0.55, color=ORANGE, label="Days intraday followed the gap")
    for b, v, n, thin in zip(bars, cont, t["days"], t["note"] != ""):
        if pd.notna(v):
            a2.annotate(f"{v:.0f}%\nn={int(n)}{'*' if thin else ''}", (b.get_x() + b.get_width() / 2, max(v, 50)),
                        ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 3), textcoords="offset points")
    a2.axhline(50, color=MUTED, linewidth=1, linestyle="--", label="50% = coin flip")
    a2.set_ylim(0, 100)
    a2.set_ylabel("percent of days", color=MUTED)
    a2.set_title("How often the gap continued  (* = too few days, not reliable)", loc="left", color=INK, fontsize=11)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" to ", "\nto ") for s in t.index], fontsize=8)
        ax.set_xlabel("Overnight SOX move", color=MUTED)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def group_compare_chart(tables: dict[str, pd.DataFrame], path: Path) -> None:
    """Gap and open-to-close move per SOX bucket, one bar colour per group (up to four)."""
    names = list(tables)
    t0 = tables[names[0]].drop(index="all days")
    x = np.arange(len(t0))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, sharex=True)
    w = 0.8 / len(names)
    for ax, col, ttl in zip(axes, ("avg_gap_pct", "avg_intraday_pct"),
                            ("Average gap at the open", "Average open-to-close move")):
        _style(ax)
        for k, (nm, colr) in enumerate(zip(names, PALETTE)):
            tb = tables[nm].drop(index="all days")
            vals = tb[col].astype(float)
            ax.bar(x + (k - (len(names) - 1) / 2) * w, vals.fillna(0), w - 0.04, color=colr, label=nm)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(ttl, loc="left", color=INK, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" to ", "\nto ") for s in t0.index], fontsize=8)
        ax.set_xlabel("Overnight SOX move  (buckets with under 10 days are unreliable, see tables)", color=MUTED)
        ax.set_ylabel("percent", color=MUTED)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Do some groups react later than others?", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def scatter_chart(df: pd.DataFrame, title: str, path: Path) -> None:
    """Overnight SOX move (x) against Tokyo opening gap (y), one dot per stock-day, plus a
    straight-line fit. A slope near 1 means the gap copies the SOX move one for one."""
    d = df[["signal", "gap"]].dropna() * 100
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.scatter(d["signal"], d["gap"], s=14, color=BLUE, alpha=0.45, linewidths=0, label=f"stock-days (n={len(d)})")
    if len(d) > 2:
        m, c = np.polyfit(d["signal"], d["gap"], 1)
        xs = np.array([d["signal"].min(), d["signal"].max()])
        ax.plot(xs, m * xs + c, color=ORANGE, linewidth=2, label=f"fit: gap = {m:.2f} x SOX {c:+.2f}")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Overnight SOX move (%)", color=MUTED)
    ax.set_ylabel("Tokyo open gap (%)", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def equity_chart(curves: dict[str, pd.Series], oos_start, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    _style(ax)
    colors = [BLUE, ORANGE, MUTED, "#1baf7a"]
    for (nm, s), colr in zip(curves.items(), colors):
        ax.plot(pd.to_datetime(list(s.index)), s.values / s.iloc[0] * 100, color=colr, linewidth=2, label=nm)
    if oos_start is not None:
        ax.axvline(pd.Timestamp(oos_start), color=MUTED, linestyle="--", linewidth=1)
        ax.text(pd.Timestamp(oos_start), ax.get_ylim()[1], " out-of-sample starts", fontsize=8, color=MUTED, va="top")
    ax.set_ylabel("value, start = 100 (after costs)", color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
