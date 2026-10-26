# StockBot: US gap-and-trend backtester and Japan research tools

[![tests](https://github.com/Refadul19/StockBot/actions/workflows/tests.yml/badge.svg)](https://github.com/Refadul19/StockBot/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **For education and research only. This is not financial advice.** Past or simulated results
> do not predict future results, and the simple rules tested here did not beat buy and hold.
> The code is paper-trading only and has no live-money mode. Market data comes from Alpaca
> (US) and unofficial Yahoo Finance (Japan), and can be wrong.

This repository has two independent packages:

| Package | What it is | Needs |
|---|---|---|
| `jpgap` | Research on Japanese stocks: what they do after the Tokyo open following big overnight moves in the US chip index (SOX). Daily data, no trading. See [Japan research module](#japan-research-module-jpgap-research-only). | Nothing but Python. No account or keys. |
| `gapbot` | An intraday US gap-and-trend backtester with pessimistic fills, described below. | Free Alpaca paper keys for real data (a synthetic mode works without them). |

## Quick start

Python 3.10 or newer.

```bash
git clone https://github.com/Refadul19/StockBot.git
cd StockBot
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                              # all tests should pass

python -m jpgap research --months 60             # Japan research, writes tables and charts to runs/jp/
python -m gapbot backtest --synthetic --start 2025-03-03 --end 2025-04-30   # US backtester demo, no keys
```

## Web dashboard (type a ticker, see reference levels)

```bash
streamlit run app.py
```

Enter a Tokyo stock code (for example `6857`) and the page shows the expected open, a reference target
and stop, and **how those same levels would have done** on days they were not built from. They are
descriptions of how far the stock usually moves, not predictions. In the project's own tests they lost
money after costs on every stock tried, and the page says so whenever that is the case.

**Deploy it free on Streamlit Community Cloud:** sign in at https://share.streamlit.io with GitHub, press
"Create app", choose this repository, branch `main`, main file `app.py`, then Deploy. No keys or secrets are
needed. Note that Yahoo Finance sometimes limits requests from shared cloud servers; if so the page shows
"Could not load data" and works again later.

Contributions and bug reports are welcome. Run `python -m pytest -q` before opening a pull request.
Licensed under the [MIT License](LICENSE).

---

# gapbot: gap-and-trend day trading backtester

A backtester for an intraday US stock strategy: scan for gap-ups, buy on a 15-minute
trend confirmed by the 5-minute chart, manage the trade with ATR stops, profit locks,
a partial target and a trailing stop, and be flat before the close.

It is built to answer one question honestly before any money is involved:
**does this strategy have an edge after realistic costs, on days it was not tuned on?**

## The rules (all in `config.yaml`)

| Stage | Rule |
|---|---|
| Scan | Gap of at least 4% at the open, price $2 to $200, 20-day average volume and dollar volume above a floor. Top 15 loaded. |
| Re-rank | Every 5 minutes by 15m trend score, relative volume and how well the gap holds. Only the top 6 may be bought. |
| Entry | From 10:00. 15m trend score of 60+ (price above EMA9, EMA9 above EMA20, EMA9 rising, above VWAP, higher low), then 5m confirmation: volume above its recent average, MACD histogram not falling, above EMA9 and VWAP, not more than 2 ATRs above EMA9. Cooldown after a win or loss in the same stock. |
| Size | Smallest of: 0.5% of equity at risk, 10% of equity, 5% of the last 5 minutes' volume. |
| Stop | 1.5 x 5-minute ATR below entry, where the ATR is never below the stock's 14-day daily ATR / sqrt(78). Raised to breakeven at +1R and to +0.5R at +1.5R. |
| Target | Sell 75% at +2R, trail the last 25% at 1.5 ATR below the high. |
| Time stop | Exit if the trade is not at least +0.3R after 20 minutes. |
| Close | Everything sold from 15:55. New entries stop at 15:00 and after a 2% daily loss. |

All indicators (EMAs, ATR, MACD, VWAP, the volume average) use regular-session bars from 09:30
only. Premarket bars are thin, and mixing them in made early stops too tight, made the first
session bars look like volume surges, and compared the first 15m bar with a premarket low.
That is why the first entry is at 10:00: the trend score needs two completed 15m bars.

## How the simulator stays honest

- Decisions use only completed bars. An order decided at 10:15 fills at the 10:15 bar's open, never on a bar the decision already saw. `tests/test_no_lookahead.py` checks this by deleting future data and confirming nothing changes.
- Every market order pays slippage, half the spread and a market impact cost that grows with order size versus bar volume.
- Stops fill worse than the stop price, and from the open if price gaps through.
- A sell limit (the target) fills only if price trades **through** it. A touch is not a fill.
- If one bar hits both the stop and the target, the stop is assumed first.
- FINRA and SEC sell fees are charged.
- The last 30% of the date range is held out. Judge the strategy on that column.
- `--compare-optimistic` reruns with paper-account style fills, so you can see how much a paper account flatters results.
- Every report shows SPY and IWM buy-and-hold over the same days, and the strategy's daily correlation with each.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                    # all tests should pass
```

Try it with synthetic data first (no account needed; profits and the SPY/IWM lines here are random and mean nothing):

```bash
python -m gapbot backtest --synthetic --start 2025-03-03 --end 2025-04-30 --compare-optimistic
```

### Alpaca data

1. Sign up at alpaca.markets and open the **Paper Trading** dashboard.
2. Generate API keys there, copy `.env.example` to `.env`, and paste them in.
3. Start small, with a few symbols and one month, to check everything works:

```bash
python -m gapbot backtest --start 2025-09-01 --end 2025-09-30 --symbols TSLA,AMD,PLTR,SOFI,RIVN,MARA
```

4. Then the full universe over a longer period (the first run downloads and caches data, later runs are fast):

```bash
python -m gapbot backtest --start 2025-01-02 --end 2025-12-31 --progress --compare-optimistic
```

If Alpaca refuses SIP data on your plan, set `data_feed: "iex"` in `config.yaml`. Be aware that IEX
is only a small slice of total volume, so the volume filters and size caps become much stricter.

## Reading the results

Each run writes to `runs/<timestamp>/`:

- `report.md` and `summary.json`: metrics for in-sample, out-of-sample and all days
- `trades.csv`: one row per trade with entry, exits, P&L, R multiple and the signal values at entry
- `fills.csv`: every fill with the reference price, so you can see the slippage paid
- `daily.csv`, `equity_curve.csv`, `equity.png` (with SPY and IWM drawn on the same scale)
- `config.yaml`: the exact settings, so any run can be repeated

What to look at, in order:

1. **Out-of-sample expectancy_r** (average profit per trade in units of risk). Above 0 after costs is the minimum; above 0.1R is good.
2. **t_stat_mean_r**. Under about 2 means the average trade is not clearly different from luck. You need a few hundred trades to get there.
3. **max_drawdown_pct**. Could you sit through it with real money?
4. **Realistic vs paper-style fills.** If the edge disappears with costs, it was never there.
5. **SPY_return_pct, IWM_return_pct and corr_*.** If the strategy only matches IWM, or its daily
   correlation with IWM is high, a good period is mostly the market, not the rules.

## Tuning without fooling yourself

```bash
python -m gapbot sweep --start 2025-01-02 --end 2025-12-31 \
  --param exit.target_r=1.5,2,3 --param exit.stop_atr_mult=1,1.5,2
```

The sweep picks settings using in-sample days only, then shows how the best ones did out of sample.
If out-of-sample results are much worse, the settings are overfit. Keep the grid small: every extra
combination raises the chance that the "best" one is just lucky.

## Known limits

- **Survivorship bias.** Alpaca's asset list has only stocks that exist today, so delisted stocks are missing. Real results would be somewhat worse.
- **Scan at the open.** Candidates are chosen from the official 09:30 open, not the premarket. Entries start at 09:45, so no future data is used, but premarket scanning behaves a little differently.
- **Halts and short-sale rules** are not modelled. A halted stock simply has no bars until it reopens.
- **Long only.** No shorting.
- **One train/test split.** A walk-forward test (tune, test the next month, roll forward) would be stronger and is a good next addition.
- **No parabolic rule yet.** Your friend's bot has one; it can be added once the rule is defined.

## Next steps

1. Run a full year and read the out-of-sample column.
2. Only if it holds up: a live paper-trading runner using this same strategy code against Alpaca's paper account, with a daily comparison of paper fills vs what this simulator predicted.
3. Only after months of that: a small real-money test.

This is a research tool, not financial advice. Past results, simulated or paper, do not predict future returns.


## Japan research module (`jpgap`), research only

This part places no orders and never touches Alpaca or the US strategy. It asks one question
about Japanese stocks: after the US market has moved overnight, what does a Japanese stock do
once Tokyo is open?

Run it with one command:

```bash
python -m pip install -r requirements.txt
python -m jpgap research                       # last 6 months (the default)
python -m jpgap research --months 60           # five years
python -m jpgap research --months 12 --cost-pct 0.15
python -m jpgap research --offline             # re-use cached data, no network
python -m jpgap research --include-earnings    # keep earnings days in the main results
```

It downloads daily prices from Yahoo Finance (yfinance), caches them in `data_cache/jp/`, and
writes tables, charts and `report.txt` to `runs/jp/`. Settings are in `config.jp.example.yaml`
(copy it and pass `--config`). If one ticker fails to download, the error is logged and the
rest carry on. `report.txt` starts with a short "What this means" paragraph in plain English.

### The words used

- **Overnight SOX move**: the percent change of the US chip index (SOX) over the US session that
  ended just before Tokyo opened.
- **Gap**: Tokyo open divided by the previous Tokyo close, minus 1. Already public at 9:00.
- **Intraday**: Tokyo close divided by Tokyo open, minus 1. This is what you could still capture.
- **Continued**: the intraday move had the same sign as the gap (up then up, or down then down).
- **Bucket**: a group of days with a similar SOX move, for example "above +2%".
- **Extreme day**: SOX moved more than 4% up or down (`analysis.extreme_sox_pct`).
- **t-stat**: the average move divided by its normal wobble. Below about 2 in size, treat it as noise.
- **Out-of-sample**: the most recent 30% of days, kept apart from the rest. Results are shown for it first.

### How dates are matched

A US session ends at 05:00 or 06:00 Japan time the next morning, so it is always known before
Tokyo opens at 09:00. Tokyo day T is paired with the latest US session dated T-1 or earlier
(Monday pairs with Friday). A day is skipped, and logged, when the US was closed, or when Tokyo
was closed on a weekday in between so the gap would mix two US sessions. The pairing uses
calendar dates, so the US daylight saving change cannot shift it; tests check this.

### What the report contains

1. **Earnings days.** A day a company reports its own results is marked and, by default, left out
   of every main result (see the earnings section below). The report lists how many days were
   excluded per stock, how complete the date list looks, an earnings-days-only table, and the
   upcoming report dates.
2. **Data checks.** SOX is compared with the S&P 500 calendar (missing days would fake a big
   move), checked for impossible rows, and its biggest days are compared with two US chip ETFs
   (SOXX and SMH). They hold different baskets, so this is a sanity check, not an exact copy.
   A bad-data guard also drops a row whose price is 3x away from its recent median. It caught
   a real Yahoo glitch (the TOPIX ETF at one tenth of its price for two days).
3. **Simple rules versus the TOPIX ETF (main result).** Buy the open and sell the close after a
   SOX rise of 1% or more ("follow"), after a fall of 1% or more ("dip"), or every day. The
   result is percent per trade after 0.1% cost on each side, with no lot or budget limit, so no
   stock is skipped. Out-of-sample first.
4. **All stocks, each day counted once.** Averages the stocks within each day first, because
   all stocks react to the same SOX move. This gives an honest day count. Shown for all days,
   without extreme days, extreme days only, and for the out-of-sample and in-sample parts.
5. **Year by year**, so you can see if a pattern repeats or lives in one year.
6. **By group and by stock**, each with the number of days per bucket. Buckets under 10 days are
   marked "too few days, not reliable".
7. **Ranking** of stocks by how closely the open follows SOX and by how much of the move
   arrives after the open.
8. **Feasibility (a separate view).** Using `rules.budget_per_trade_yen` and `rules.capital_yen`,
   it lists which stocks one 100-share lot can buy, then re-runs the rules with only those,
   whole lots, and a cap on open trades. Trade size is fixed in yen with no compounding.

### Earnings days

On the day a company reports its own results, its price moves because of those results, not
because of what the US did overnight. Most Japanese companies report after the 15:30 close, so
the reaction shows at the NEXT day's open. A report after the close (or at an unknown time) marks
the report day and the next trading day. A report during market hours marks that day only.
Weekends and holidays are skipped, so a Friday report marks the Monday. These rules, and the
assumption for unknown times, are in the `earnings` section of the config.

Dates come from yfinance first and then from `config/jp_earnings.csv` (columns `ticker,date,time,note`;
`time` is optional, as `HH:MM` Japan time, `after_close` or `during_hours`). A manual date replaces
any yfinance date within a week of it. The file is pre-filled with the dates you gave me for
28 to 30 October 2026, all marked "to verify": check them on each company's investor page.
yfinance is unofficial. For Japan its dates are fairly complete for large companies, but it
returned too few dates for some stocks (see below) and its times are mostly placeholders, so they are
ignored by default. The report lists every stock with missing or uncertain dates.

### What this means: why earnings days are removed

This research asks whether the US overnight move helps predict what a Japanese stock does after
the open. On the day a company announces results, the price jumps because of the results, and that
jump can be much bigger than anything the US did. If those days stay in, a big earnings jump can
look like an "overnight US effect" and distort the averages. In this data the difference is large:
on earnings days the average gap was about 3.4% in size against 1.6% on normal days, and the gap
followed SOX much less closely (correlation 0.37 against 0.60). So the main results leave those days
out and show them in a separate table. Because the date lists are not perfect, a few earnings days
may still be mixed in.

### What the 5-year run showed (run on 26 Oct 2026, 1,111 Tokyo days, 17 stocks, earnings days excluded)

- The opening gap follows SOX: about 0.6% of gap per 1% of SOX across all stocks. The gap is
  public at 9:00, so it is not the opportunity by itself.
- After SOX rose more than 2% (207 days) the average stock then fell 0.27% from open to close
  (t-stat about -2.5), a small partial reversal. It was clearest for chip equipment makers
  (-0.34%). After SOX fell more than 2% the stocks recovered only 0.14%, which is within noise.
- Without the 113 extreme days the numbers were -0.22% and +0.23%, so a few crash days are not
  what drives the result. The reversal after big rises had the expected sign in all five
  calendar years with at least 10 such days (2021 had only 7), but its size is small and costs (0.2% round trip) are about as large as the effect.
- Laggards: there was no solid sign that smaller suppliers react later. The AI infrastructure
  group had the weakest link to SOX at the open (correlation 0.51 versus 0.69 for chip equipment)
  and Fujikura and Sumitomo Metal Mining showed the most catch-up after the open, but the effect
  is tiny (about 0.05% and 0.01% per 1% of SOX), so treat the ranking as rough.
- Simple rules, after costs: in the held-out last 333 days "dip" returned +27% and "follow" -38%,
  against +51% for holding the TOPIX ETF. In the earlier 778 days "dip" lost 28%, "follow" lost
  55% and TOPIX gained 28%. Trading every day loses most of the money to costs. So no rule beat
  buy and hold, and "dip" changing sign between the two periods is a warning that it is noise.
- All 61 SOX days over 5% matched SOXX and SMH within 2 points, including the -10.3% day on
  5 June 2026, so those large moves are real and not data errors.

None of this is proof of an edge. It is one market over about five years.

### Why a small sample can mislead you

Flip a coin 10 times and you can easily get 7 heads, which looks like a pattern but is luck. Flip it
1,000 times and it settles near 50%. A bucket with 5 or 10 days works the same way: one wild day
can move the average a lot, and a "70% continued" result may be luck. That is why every table
shows the number of days and marks buckets under 10 days as not reliable. The biggest SOX moves
are rare, so those buckets are always the thinnest. Counting every stock as its own data point
makes it worse, because stocks move together on the same day, which is why the report also
shows each day counted once. With many stocks and buckets, some will look interesting by chance
alone, so a real finding should repeat across years and in the out-of-sample part.

### Limits

- Earnings dates: yfinance returned too few dates for Kioxia, Fujikura, JX Advanced Metals, Unitika
  (none in the last five years, so no earnings days were marked for it) and SoftBank. Add their dates to
  `config/jp_earnings.csv` to mark them. The report names these stocks every run.
- Yahoo data is unofficial and can be wrong. Rows far from the recent price are dropped and
  logged. Other outlier moves over 25% are logged but kept (Unitika, 3103.T, had five days of
  25% to 30% intraday moves, which look like real limit moves in a low-priced stock).
- The stock list is companies that exist today (survivorship bias), and prices are not adjusted
  for dividends. Kioxia (285A.T) only listed in December 2024.
- Daily bars cannot show what happens inside the Tokyo session, only open to close.
- SOX is an index. You cannot trade it, and the "signal" is only a number to sort days by.
