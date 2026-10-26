"""Web dashboard: type a Tokyo stock code, see reference levels and how they would have done.

Run locally:   streamlit run app.py
Deployed on:   Streamlit Community Cloud (main file: app.py)

This is research, not advice. The levels are not predictions: the expected open comes from how
this stock's gap has followed the overnight US chip index (SOX), and the target and stop are how
far it has usually moved from the open. The page always shows how those levels did on days the
levels were not built from.
"""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from jpgap.analysis import bucket_table
from jpgap.charts import bucket_chart, scatter_chart
from jpgap.config import JpConfig
from jpgap.data import load_daily
from jpgap.earnings import load_manual, load_yfinance, merge_events
from jpgap.plan import build_plan, parse_ticker

ROOT = Path(__file__).resolve().parent
SOX = "^SOX"

st.set_page_config(page_title="Tokyo day-trade reference levels", page_icon="📈", layout="wide")


def make_cfg(months: int, cost_pct: float, exclude_earnings: bool) -> JpConfig:
    cfg = JpConfig()
    cfg.data.months = months
    cfg.data.cache_dir = str(ROOT / "data_cache" / "jp")
    cfg.costs.cost_per_side_pct = cost_pct
    cfg.earnings.exclude = exclude_earnings
    cfg.earnings.manual_file = str(ROOT / cfg.earnings.manual_file)
    return cfg


@st.cache_data(ttl=3600, show_spinner=False)
def fetch(ticker: str, months: int):
    """Daily bars for the stock and SOX, plus its earnings dates. Cached for an hour."""
    cfg = make_cfg(months, 0.1, True)
    end = dt.date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(months=months)).date() - dt.timedelta(days=cfg.data.lookback_buffer_days)
    jp = load_daily(ticker, start, end, cfg.data)
    sox = load_daily(SOX, start, end, cfg.data)["close"]
    try:
        yf_events = load_yfinance(ticker, cfg.data.cache_dir, cfg.earnings)
    except Exception:
        yf_events = []
    events = merge_events([e for e in load_manual(cfg.earnings.manual_file) if e.ticker == ticker], yf_events, cfg.earnings)
    return jp, sox, events


def png(draw) -> bytes:
    buf = io.BytesIO()
    draw(buf)
    return buf.getvalue()


st.title("Tokyo day-trade reference levels")
st.warning(
    "**Research and education only, not financial advice.** These are not predictions. In this project's own tests the "
    "overnight US move helps estimate where a stock *opens*, but what happens after the open was close to a coin flip, "
    "and simple rules did not beat buy and hold. Prices come from unofficial Yahoo Finance data and can be wrong. "
    "Never trade money you cannot afford to lose."
)

with st.sidebar.form("inputs"):
    code = st.text_input("Tokyo stock code", value="6857", help="For example 6857 (Advantest), 8035 (Tokyo Electron), 285A (Kioxia).")
    months = st.selectbox("History used", [12, 24, 36, 60], index=3, format_func=lambda m: f"{m} months")
    cost = st.number_input("Cost per side (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05,
                           help="Commission plus slippage, paid on the buy and again on the sell.")
    excl = st.checkbox("Leave out earnings days", value=True,
                       help="On results days a stock moves on its own news, not on the US overnight move.")
    submitted = st.form_submit_button("Analyse", type="primary")

if not (submitted or not st.session_state.get("ran")):
    st.info("Change the inputs in the sidebar and press Analyse.")
    st.stop()
st.session_state["ran"] = True

try:
    ticker = parse_ticker(code)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

cfg = make_cfg(months, cost, excl)
try:
    with st.spinner(f"Loading {ticker} and SOX..."):
        jp, sox, events = fetch(ticker, months)
    plan = build_plan(ticker, jp, sox, events, cfg)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Could not load data for {ticker}. Check the code, or try again in a minute. ({type(exc).__name__})")
    st.stop()

sig, lv = plan["signal"], plan["levels"]
st.subheader(f"{ticker}: levels for the next Tokyo session ({plan['next_tokyo']}, check for market holidays)")
for w in plan["warnings"]:
    st.warning(w)
st.caption(f"Latest US session in the data: {sig['date']}. SOX moved **{sig['move'] * 100:+.2f}%** (bucket: {plan['bucket']}). "
           f"Last Tokyo close: {plan['prev_close']:,.1f} yen.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected open (entry)", f"{lv['entry']:,.1f}", f"{lv['expected_gap_pct']:+.2f}% gap, typical error ±{lv['gap_error_pct']:.2f}%", delta_color="off")
c2.metric("Reference target", f"{lv['target']:,.1f}", f"{lv['target_pct']:+.2f}% from entry", delta_color="off")
c3.metric("Reference stop", f"{lv['stop']:,.1f}", f"-{lv['stop_pct']:.2f}% from entry", delta_color="off")
c4.metric("Typical close", f"{lv['expected_close']:,.1f}", "median day", delta_color="off")
st.caption(f"Target and stop are the typical open-to-high and open-to-low moves of this stock on {lv['levels_days']} past days "
           f"({lv['levels_from']}). Long only. Round to the stock's price step before placing any order. "
           "The real open can differ from the expected open by more than the typical error.")

st.subheader("Would these levels have worked?")
(st.error if "LOST" in plan["verdict"] else st.info)(plan["verdict"])
track = pd.DataFrame(plan["track"]).T
st.dataframe(track.round(2), width="stretch")
bh = plan["buy_hold_oos_pct"]
st.caption(f"Held-out part = the most recent {plan['oos_days']} trading days (levels are built only from the {plan['is_days']} earlier days). "
           f"For comparison, buying at the open on the first held-out day and holding to the last close returned {bh:+.1f}% after costs. "
           "Returns are per trade after costs. When a day touched both levels, the stop is assumed to have hit first. "
           f"{plan['earnings_excluded']} earnings stock-days were left out." if excl else
           f"Held-out part = most recent {plan['oos_days']} days. Earnings days are included.")

st.subheader(f"How {ticker} has reacted to SOX")
acfg = cfg.analysis
table = bucket_table(plan["pairs"].reset_index(names="date"), acfg, cost)
left, right = st.columns(2)
left.image(png(lambda b: scatter_chart(plan["pairs"], f"{ticker}: overnight SOX move vs Tokyo open gap", b)))
right.image(png(lambda b: bucket_chart(table, f"{ticker}: after an overnight SOX move", b)))
st.dataframe(table.round(2), width="stretch")
st.caption(f"A gap moves about {plan['slope']:.2f}% per 1% of SOX (correlation {plan['corr']:.2f}). Buckets under "
           f"{acfg.min_bucket_days} days are marked 'too few days, not reliable'. Small samples can mislead: a handful of days can "
           "produce a pattern by luck.")

with st.expander("How to read this / limits"):
    st.markdown(
        "- **Gap** = Tokyo open / previous close - 1. **Intraday** = close / open - 1.\n"
        "- The US session ends before Tokyo opens, so the overnight SOX move is known before 9:00 Japan time.\n"
        "- Daily bars cannot show what happens inside the session, only open, high, low and close.\n"
        "- Only stocks that exist today are tested (survivorship bias), and prices are not adjusted for dividends.\n"
        "- Source code and the full research report: https://github.com/Refadul19/StockBot (MIT license)."
    )
