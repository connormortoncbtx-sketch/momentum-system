"""
automation/shadow_earnings_hold.py
==================================
Shadow #5 (no orders): does the live rule that skips stocks reporting earnings
during the holding week cost money?  (Research, Oct 10 2026: strong-momentum
names beat the universe by ~1%/wk in their report week in dev AND holdout, and
dropping the exclusion improved the weekly top-20 basket in both periods.)

Two baskets are formed from the SAME data/scores_final.csv each entry week and
held with identical timing, so their difference isolates the rule:
  A  live replica      top 10 by composite rank (earnings names excluded)
  B  earnings allowed  top 10 by alpha rank among names that are either ranked
                       or excluded ONLY for "earnings_in_holding_period"
                       (composite rank is the alpha-rank order of eligible names)
Timing: entry-day official close -> that week's last close on or before Friday
(the live system exits Thu/Fri afternoon). Equal weight, 15 bps per side,
full turnover weekly (same as live). Idempotent per entry date.

Ledger: data/shadow_earnings_ledger.jsonl -- one row per completed week.
"""
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation.shadow_lc_momentum import entry_date_this_week, daily_closes  # noqa: E402
from automation.system_logger import log_event, LogStatus  # noqa: E402

log = logging.getLogger("shadow_earn")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

DATA = ROOT / "data"
STATE = DATA / "shadow_earnings_state.json"
LEDGER = DATA / "shadow_earnings_ledger.jsonl"
SCORES = DATA / "scores_final.csv"
N, COST_BPS = 10, 15.0


def baskets(scores: pd.DataFrame) -> tuple[list[str], list[str]]:
    s = scores.copy()
    a = s.dropna(subset=["composite_rank"]).nsmallest(N, "composite_rank").symbol.tolist()
    excl = s.get("excluded_from_ranking", False)
    excl = excl.astype(str).str.lower().isin(["true", "1"]) if not isinstance(excl, bool) else pd.Series(False, index=s.index)
    eligible = (~excl) | (s.get("exclusion_reason", "").astype(str) == "earnings_in_holding_period")
    b = s[eligible & s.alpha_rank.notna()].nsmallest(N, "alpha_rank").symbol.tolist()
    return a, b


def basket_return(px: pd.DataFrame, names, d0, d1):
    rets = []
    for s in names:
        if s in px.columns and d0 in px.index and d1 in px.index:
            p0, p1 = px.at[d0, s], px.at[d1, s]
            rets.append(p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 else 0.0)
        else:
            rets.append(0.0)                      # no price: treated as flat (rare)
    return float(np.mean(rets)) - 2 * COST_BPS / 1e4 if rets else 0.0


def main():
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    entry = entry_date_this_week(today)
    if entry is None or today < entry:
        log.info("No entry session yet this week")
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "nav_a": 100.0, "nav_b": 100.0, "open": None, "last_entry": None}
    if state["last_entry"] == entry.isoformat():
        log.info(f"Already formed baskets for {entry} -- idempotent skip")
        return

    # 1. close out last week's baskets at that week's last close on/before Friday
    op = state["open"]
    if op:
        d0 = dt.date.fromisoformat(op["entry"])
        fri = d0 + dt.timedelta(days=4 - d0.weekday())
        px = daily_closes(set(op["a"]) | set(op["b"]) | {"SPY"}, d0, fri)
        days = [d for d in px.index if d0 <= d <= fri]
        if not days or days[-1] == d0:
            log.warning("Exit-day closes unavailable -- will retry next run")
            return
        d1 = days[-1]
        ra, rb = basket_return(px, op["a"], d0, d1), basket_return(px, op["b"], d0, d1)
        spy = float(px.at[d1, "SPY"] / px.at[d0, "SPY"] - 1) if "SPY" in px.columns else None
        state["nav_a"] *= 1 + ra
        state["nav_b"] *= 1 + rb
        row = {"entry": op["entry"], "exit": d1.isoformat(),
               "live_replica_ret_pct": round(ra * 100, 3), "earnings_allowed_ret_pct": round(rb * 100, 3),
               "diff_pct": round((rb - ra) * 100, 3), "spy_ret_pct": round(spy * 100, 3) if spy is not None else None,
               "nav_live_replica": round(state["nav_a"], 3), "nav_earnings_allowed": round(state["nav_b"], 3),
               "earnings_names": op["earn_names"], "dropped_for_them": op["dropped"]}
        with LEDGER.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        msg = (f"Shadow earnings-hold week of {op['entry']}: live replica {row['live_replica_ret_pct']:+.2f}% vs "
               f"earnings allowed {row['earnings_allowed_ret_pct']:+.2f}% (diff {row['diff_pct']:+.2f}%)")
        log.info(msg)
        log_event("shadow_earnings", LogStatus.SUCCESS, msg,
                  metrics={k: row[k] for k in ("live_replica_ret_pct", "earnings_allowed_ret_pct", "diff_pct")})

    # 2. form this week's baskets from the scores the live trader used
    sc = pd.read_csv(SCORES, low_memory=False)
    a, b = baskets(sc)
    state["open"] = {"entry": entry.isoformat(), "a": a, "b": b,
                     "earn_names": sorted(set(b) - set(a)), "dropped": sorted(set(a) - set(b))}
    state["last_entry"] = entry.isoformat()
    STATE.write_text(json.dumps(state, indent=1))
    log.info(f"Formed {entry}: earnings names in B: {state['open']['earn_names'] or 'none'}")


if __name__ == "__main__":
    main()
