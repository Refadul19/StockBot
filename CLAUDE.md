# gapbot: project rules for Claude Code

Read this file fully before doing anything. Then follow `docs/BUILD_PLAN.md` phase by phase.

## What this project is

An intraday US stock trading system (gap-and-trend, long only) for one owner.
It has two parts that must share the same strategy code:

1. A **backtester** (already built, in `gapbot/`) that replays historical Alpaca data with
   pessimistic fills, so results are not flattered the way a paper account flatters them.
2. A **live paper-trading runner** (to be built in Phase 6) that trades an Alpaca **paper**
   account using exactly the same rules.

The owner sets the trading rules. Your job is engineering, testing and honest reporting.

## Non-negotiable rules

1. **Paper trading only.** Never write code that can send orders to a real-money account.
   The Alpaca trading client is always created with `paper=True`, and the runner refuses to
   start if it detects anything else. Do not add a switch for live money, even if asked
   casually. Real money is outside the scope of this project.
2. **Secrets.** API keys live only in `.env` (git-ignored). Never print, log, commit or paste
   them into code, tests, docs or chat.
3. **No look-ahead.** A decision at time `now` may only use bars that finished at or before
   `now`. Orders decided at `now` fill on the next bar. `tests/test_no_lookahead.py` and
   `tests/test_session_fixes.py` must always pass. Any new indicator or rule needs its own
   look-ahead test (compute it on full data, then on data cut at `now`, and assert equality).
4. **Costs stay pessimistic by default.** Never make the default fill model more optimistic.
   The optimistic model exists only for the `--compare-optimistic` comparison.
5. **Every strategy number is a config field.** No magic numbers in strategy, engine or runner
   code. New parameters go into `gapbot/config.py` and `config.example.yaml` with a comment.
6. **Do not change the trading rules on your own.** If you think a rule is wrong or a
   parameter should change, explain why with evidence and ask. Bug fixes that make the code
   match the documented rule are fine; say what you fixed.
7. **Never tune on out-of-sample data.** Parameter choices use in-sample or walk-forward
   training windows only. Never pick settings because they looked good on the held-out period.
8. **Tests before "done".** Run `python -m pytest -q` after every change. A task is not done
   while any test fails. Add tests with every feature and every bug fix.
9. **Run things, do not assume.** If you claim a command works, you ran it. If you could not
   run something (no network, no keys, market closed), say so plainly.
10. **Report honestly.** When you show results, lead with the out-of-sample numbers, the
    benchmark comparison and the trade count. Never describe a short or in-sample result as
    proof that the strategy works.

## Layout

```
gapbot/
  config.py      all parameters (dataclasses), YAML load/save
  data.py        AlpacaSource (cached historical bars), SyntheticSource (tests/demo)
  indicators.py  causal indicators, resampling, last_completed()
  scanner.py     daily gap scan from prior daily bars + today's open
  strategy.py    SymbolDay (signals, ATR with daily floor), position_size()
  fills.py       pessimistic fill prices and FINRA/SEC fees
  engine.py      minute-by-minute portfolio backtest
  metrics.py     stats, in/out-of-sample split, SPY/IWM benchmark
  cli.py         `python -m gapbot backtest | sweep`
tests/           pytest suite
config.example.yaml, README.md, .env.example
```

## Commands

```bash
python -m pytest -q
python -m gapbot backtest --synthetic --start 2025-03-03 --end 2025-04-30 --compare-optimistic
python -m gapbot backtest --start 2025-09-01 --end 2025-09-30 --symbols TSLA,AMD,PLTR
python -m gapbot backtest --start 2025-01-02 --end 2025-12-31 --progress --compare-optimistic
python -m gapbot sweep --start ... --end ... --param exit.target_r=1.5,2,3
```

The owner may be on Windows. Use `python` (not `python3`) in docs, and paths that work on both.

## Timing model (keep it identical in backtest and live)

- Times are America/New_York. Regular session 09:30 to 16:00.
- Indicators use regular-session bars only. Premarket bars are ignored for signals.
- Each 1-minute bar is labelled by its start time; it is complete one minute later.
- Entry checks run every 5 minutes from 10:00 to 15:00. Exits are checked every minute.
- Stop is checked before target when one bar touches both.
- New entries stop at 15:00; everything is flattened from 15:55.

## How to work

- Work one phase of `docs/BUILD_PLAN.md` at a time. At the end of each phase, stop and give
  the owner a short plain-language summary: what was built, what was run, results, open
  questions. Wait for approval before the next phase.
- Prefer small, reviewable changes. Keep the existing style (type hints, dataclasses, docstrings
  that explain *why*).
- Explain trading or statistics terms briefly when you first use them in a summary.
- In docs and summaries, write plainly and do not use em dashes.
