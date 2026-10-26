# gapbot build plan

This plan is for Claude Code. Follow `CLAUDE.md` at all times. Do one phase at a time, finish
its acceptance checks, then **stop and report** to the owner before starting the next phase.

Current state: the backtester in `gapbot/` is complete and 53 tests pass on synthetic data.
The Alpaca download code (`AlpacaSource`) has never been run against the real API.

---

## Phase 0: Check the environment

**Goal:** the project runs on the owner's computer.

Tasks
1. Confirm Python 3.10+ (`python --version`). Create `.venv` and install `requirements.txt`.
2. Run `python -m pytest -q`. Fix only genuine environment problems (for example a Windows
   path or encoding issue). Do not weaken tests to make them pass.
3. Run the synthetic backtest with `--compare-optimistic` and check that `runs/<timestamp>/`
   contains `report.md`, `summary.json`, `trades.csv`, `fills.csv`, `daily.csv`,
   `equity_curve.csv`, `equity.png` and `config.yaml`.

Done when: all tests pass and the synthetic run writes every output file.

---

## Phase 1: Connect real Alpaca data

**Goal:** real historical data downloads, caches and passes sanity checks.

Before starting, ask the owner to create `.env` from `.env.example` with Alpaca **paper** keys.
Never ask them to paste keys into the chat.

Tasks
1. Run the small backtest from the README (one month, about six symbols). Fix any bugs in
   `AlpacaSource` (request shapes, pagination, timezone conversion, empty responses).
2. Check the current Alpaca docs for the owner's data plan. If historical SIP data is refused,
   explain the error and switch to `data_feed: "iex"` in `config.yaml`. Explain in the report
   that IEX volume is only a slice of total volume, so liquidity filters and size caps behave
   differently.
3. Add retries with exponential backoff for HTTP 429 and 5xx errors, and a clear message when
   the keys are missing or wrong.
4. Add a `python -m gapbot check-data --start ... --end ...` command that prints, for a sample
   of cached symbols and days: bar counts per day (about 390 regular-session bars for liquid
   stocks), first and last bar times in New York time, any duplicate timestamps, and whether
   daily open/close agree with the first/last minute bars within a small tolerance.
   Write tests for it with synthetic data.
5. Make the universe download robust: cache it, skip non-common-stock symbols where Alpaca
   provides that information, and log how many symbols were kept.

Done when: the small run completes, `check-data` shows no serious problems, and tests pass.

---

## Phase 2: Baseline backtest

**Goal:** an honest first answer to "is there an edge?".

Tasks
1. Run `2024-01-02` to `2025-12-31` (or as far back as the data plan allows) with
   `--progress --compare-optimistic`, default settings, no tuning.
2. Write `runs/<timestamp>/findings.md` in plain language covering:
   - out-of-sample expectancy (R), t-stat, trade count, profit factor, max drawdown
   - realistic versus paper-style fills: how much the paper-style model overstates
   - strategy versus SPY and IWM buy-and-hold, and the daily correlations
   - which exit reasons dominate (`final_exit` counts) and the average R per exit reason
   - how often the volume cap was the binding size limit (`sig_binding`)
   - the five best and five worst days, and whether a few trades explain most of the profit
3. Stop and report. Do not tune anything in this phase.

---

## Phase 3: Robustness checks

**Goal:** know whether a result is real or luck, before anyone trusts it.

Tasks
1. **Walk-forward command:** `python -m gapbot walkforward --start ... --end ...
   --train-months 6 --test-months 1 --param ...`. For each window, pick the best parameter
   combination on the training months only (by expectancy R, with a minimum trade count),
   then run it on the following test month. Report the stitched test-month results and how
   often the chosen parameters changed. Reuse `Backtester.run(..., only_days=...)`.
2. **Cost stress test:** rerun the baseline with every cost in `fill:` doubled. Report the
   change in expectancy.
3. **Parameter sensitivity:** for the main parameters (`exit.stop_atr_mult`, `exit.target_r`,
   `entry.min_trend_score`, `scanner.min_gap_pct`), run the default value and one step either
   side. A robust strategy degrades gently; a cliff means overfitting.
4. **Bootstrap:** resample trades with replacement 5,000 times and report a 95% interval for
   mean R and for total return.
5. Tests for the walk-forward window logic (no overlap between train and test days, correct
   roll-forward) and the bootstrap (deterministic with a seed).

Done when: all four checks appear in a `robustness.md` report. Stop and report.

---

## Phase 4: Parabolic rule (owner defines it first)

**Goal:** add the owner's parabolic rule, behind a config switch.

Before writing code, **ask the owner which rule they mean** and get exact numbers. Offer these
as starting points, not decisions:
- *Parabolic exit:* if price rises more than X ATRs above the 5m EMA9 within N bars, sell part
  or all of the position, or tighten the trailing stop to Y ATRs.
- *Parabolic entry filter:* skip entries when the last N 5m bars are all green and the move is
  more than X ATRs (buying the top of a vertical move).

Tasks
1. Implement exactly the agreed rule. All numbers go into config, default **off**.
2. Look-ahead test plus unit tests with hand-built bars that trigger and do not trigger it.
3. Compare baseline versus rule-on, using walk-forward from Phase 3, not one in-sample run.

---

## Phase 5: Premarket scanner (optional, ask first)

**Goal:** choose candidates before the open, like the owner's friend's bot.

Tasks
1. Add `scanner.mode: "open" | "premarket"`. In premarket mode, at 09:25 New York time the
   gap is the last premarket trade versus the prior close, and premarket volume is a filter.
   It must use only bars before 09:25.
2. Re-scan every 5 minutes from 08:00 to 09:25 to build the watchlist, logging how the ranking
   changes. Entries still follow the normal rules from 10:00.
3. Look-ahead tests: deleting all bars at or after 09:25 must not change the candidate list.
4. Compare the open and premarket modes with walk-forward.

---

## Go / no-go gate (owner decides)

Before Phase 6, present the evidence and let the owner decide. Suggested minimum bar:
- at least 300 out-of-sample or walk-forward test trades
- walk-forward expectancy above +0.10R after realistic costs, with the bootstrap interval for
  mean R above zero
- still positive with doubled costs
- not explained by the market (beats IWM over the same days, or has low correlation with it)
- a maximum drawdown the owner says they could accept

If it does not pass, say so clearly. Building the live runner anyway is the owner's choice,
but it should then be treated as an engineering exercise, not as a strategy expected to profit.

---

## Phase 6: Live paper-trading runner

**Goal:** trade the Alpaca **paper** account with the same strategy code, safely and
restart-safe.

### Architecture

```
gapbot/live/
  broker.py      Broker protocol; AlpacaPaperBroker; FakeBroker for tests
  feed.py        live 1-minute bars (Alpaca StockDataStream) + REST backfill on start
  state.py       SQLite store: positions, open orders, entries_today, cooldowns, decisions
  runner.py      the event loop: on each completed minute bar, run the same decision logic
  journal.py     trade journal from actual fill events, end-of-day report
```

- Move the per-minute decision logic out of `engine.py` into a shared module (for example
  `gapbot/decide.py`) that both the backtester and the live runner call. The backtester's
  results must not change; prove it by running the full test suite and comparing a synthetic
  backtest's `trades.csv` before and after the refactor (identical).
- `SymbolDay` is causal, so the live runner can rebuild it from "bars received so far" each
  minute and get the same signals the backtest saw.

### Requirements

1. **Modes:** `trading.mode: "dry_run" | "paper"`. Dry run makes every decision and logs it,
   but sends no orders. Start in dry run.
2. **Paper-only guard:** `AlpacaPaperBroker` always uses `TradingClient(..., paper=True)` and
   checks at start that the account is a paper account. If anything looks like a live
   account, exit with an error.
3. **Clock:** use Alpaca's market clock and calendar for open, close, early closes and holidays.
   Never hard-code Japan times. (US daylight saving moves the open between 22:30 and 23:30
   Japan time; the clock handles that.)
4. **Data feed:** subscribe to minute bars for the watchlist. On start or reconnect, backfill
   the day's bars by REST. Detect stale data (no bar for a subscribed symbol for 3 minutes
   during the session) and stop opening new trades until it recovers. Note that real-time data
   on Alpaca's free plan is IEX only; check the current plan and record which feed is used,
   because IEX volume is much lower than the SIP volume the backtest may have used.
5. **Orders:**
   - Entry: marketable limit order at signal price plus `entry.entry_limit_offset_pct`,
     time in force `day`, cancelled if not filled within 2 minutes.
   - Protective stop: as soon as an entry fills, place a broker-side stop order for the full
     quantity, so the position is protected even if the bot crashes.
   - Alpaca reserves shares for open sell orders, so a stop for 100% and a limit for 75% cannot
     rest at the same time. Handle the target in the bot: when a completed bar trades through
     the target, cancel the stop, sell 75%, then place a new stop for the remaining 25%.
     Verify this behaviour against the paper account and document what you found.
   - Moving stops (profit locks, trailing): use cancel/replace, and never leave the position
     without a stop for more than a few seconds. Log every replacement.
   - Time stop and end of day: cancel the stop, then market sell. From 15:55, cancel all open
     orders and flatten everything; at 15:58 check that the account is flat and log loudly if
     not.
6. **Risk switches:** daily loss limit (from config) stops new entries; a file named `STOP` in
   the project folder makes the runner cancel entries, flatten and exit; maximum orders per
   minute as a runaway guard.
7. **Restart safety:** all state is in SQLite. On start, reconcile with the broker (positions
   and open orders). Anything the broker has that the state store does not know about is
   logged, given a protective stop if it has none, and reported. The bot never assumes its
   memory is right when the broker disagrees.
8. **Logs and journal:** human-readable log plus JSON-lines log per day in `logs/`. The journal
   records every decision (with the signal values), every order and every fill from the trade
   updates stream, including the actual fill price.
9. **End-of-day report:** compare the account equity change with the journal's P&L
   (reconciliation), and compare actual fill prices with what `fills.py` would have predicted
   (real slippage versus simulated). This is the most valuable output of paper trading.
10. **Parity test:** replay a recorded or synthetic day through the live runner with
    `FakeBroker` and a fake feed, and assert that its decisions (symbols, times, entry and exit
    reasons) match the backtester on the same data.
11. **Tests:** FakeBroker-based tests for partial fills, rejected orders, a crash and restart in
    the middle of a trade, a stale feed, the STOP file, the end-of-day flatten, and
    reconciliation finding an unknown position.

Done when: all tests pass, one full session has run in dry-run mode and then one in paper mode
without errors, and the end-of-day report is produced. Stop and report.

---

## Phase 7: Running it every day

1. Command: `python -m gapbot live --config config.yaml`. It sleeps until shortly before the
   open, runs the session, writes the end-of-day report and exits.
2. Write `docs/OPERATIONS.md` for the owner in plain language: how to start it with Windows
   Task Scheduler (or cron on macOS/Linux) at about 21:30 Japan time, keeping the computer
   awake, where the logs and reports are, how to use the STOP file, and what to do if the
   computer restarts in the middle of a session.
3. Optional and only if the owner asks: a short daily summary sent to a webhook (Slack or
   Discord). The webhook URL goes in `.env`.

---

## Phase 8: Evaluate paper trading

Run in paper mode for at least one to three months. Each week, produce a short comparison:
paper results versus what the backtester says for the same days, and real versus simulated
slippage. If paper results are much better than the backtest, suspect the paper account's fills
first, not a hidden edge.

Real-money trading is out of scope for this plan and must not be built.
