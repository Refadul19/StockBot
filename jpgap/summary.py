"""The plain-English 'What this means' paragraph at the top of report.txt.

Every number is filled in from the results, and the wording changes with them: a move is only
called a reversal or a continuation when the t-stat says it is bigger than normal noise.
"""
from __future__ import annotations

import math


def _word(avg_gap: float, avg_intraday: float, t: float) -> str:
    if not math.isfinite(t) or abs(t) < 2:
        return "too small compared with the normal day-to-day noise to call a pattern"
    return "a continuation of the gap" if avg_gap * avg_intraday > 0 else "a partial reversal of the gap"


def what_this_means(n: dict) -> str:
    up, dn = n["up"], n["down"]
    s = []
    s.append(f"This report covers {n['days']} Tokyo trading days ({n['start']} to {n['end']}) and {n['stocks']} Japanese stocks, "
             f"and asks what the stocks do after the open when the US chip index (SOX) moved a lot overnight. {n.get('earnings_note', '')}".rstrip())
    s.append(f"The opening price already copies the US move: for every 1% that SOX moved, the average stock opened about "
             f"{n['gap_slope']:.2f}% in the same direction, so the gap itself is not the opportunity.")
    s.append(f"After SOX rose more than 2% ({up['days']:.0f} days), the average stock moved {up['avg_intraday_pct']:+.2f}% from open to close, "
             f"which is {_word(up['avg_gap_pct'], up['avg_intraday_pct'], up['t_intraday'])}.")
    s.append(f"After SOX fell more than 2% ({dn['days']:.0f} days), the average move from open to close was {dn['avg_intraday_pct']:+.2f}%, "
             f"which is {_word(dn['avg_gap_pct'], dn['avg_intraday_pct'], dn['t_intraday'])}.")
    s.append(f"{n['extreme_days']} days had a SOX move bigger than {n['extreme_pct']:g}% and are marked extreme; without them the same two "
             f"buckets give {n['up_ex']:+.2f}% and {n['down_ex']:+.2f}%, so check whether a few crash days drive the result.")
    if n["years_total"]:
        s.append(f"Year by year, the open-to-close move went against the gap after big SOX moves in {n['years_fade']} of the "
                 f"{n['years_total']} year-and-bucket cases that had enough days, so the pattern is "
                 f"{'fairly steady' if n['years_fade'] / n['years_total'] >= 0.75 else 'not steady' if n['years_fade'] / n['years_total'] <= 0.5 else 'only partly steady'}.")
    s.append(f"In the most recent {n['oos_days']} days that were held out, buying the open and selling the close after a SOX fall of "
             f"{n['thr']:g}% or more returned {n['oos_dip']:+.1f}% after costs, and after a rise of {n['thr']:g}% or more {n['oos_follow']:+.1f}%, "
             f"against {n['oos_topix']:+.1f}% for simply holding the TOPIX ETF; in the earlier training part the same rules returned "
             f"{n['is_dip']:+.1f}% and {n['is_follow']:+.1f}% against {n['is_topix']:+.1f}% for TOPIX.")
    s.append(f"Treat all of this as a rough look, not proof: a few years in one market can be luck, buckets with fewer than "
             f"{n['min_days']} days are marked unreliable, and costs of {n['cost']:g}% on each side take a big bite out of a one-day trade.")
    return "\n".join(["WHAT THIS MEANS (plain English)", " ".join(s), ""])
