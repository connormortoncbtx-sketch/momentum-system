"""
automation/shadow_slow_momentum.py
==================================
Shadow #4 (no orders): slow 12-1 momentum with a rank-based exit -- the best
condition-based design from "Rank tranches and hold length" (Oct 10, 2026) in
docs/2026-10_strategy_research.md (research/hold_conditions.py):

  universe  live pipeline symbols (data/scores_final.csv), price >= $5,
            4-week avg $ volume >= $2M/day, >= 53 weeks of history, no funds,
            no reverse split in 52 weeks  (research/backtest.py universe())
  score     12-1 momentum: close 4 weeks ago / close 52 weeks ago - 1
  rule      each entry day, using the prior Friday's ranks: KEEP every holding
            still ranked in the top 100; SELL the rest; FILL back to 10 names
            from the top of the list; equal weight (as backtested)
  backtest  dev 2017-23: 26.4%/yr, Sharpe 0.75, max DD -46%
            holdout 2024-26: 38.1%/yr, Sharpe 0.85, max DD -42% (SPY 20.6% / 1.31)
            median hold ~21 weeks. Best of 106 designs -> level is optimistic.
  prices    entry-day official closes fetched after the fact; idempotent per date
  costs     15 bps per side on traded weight

Ledger: data/shadow_slow_ledger.jsonl
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
sys.path.insert(0, str(ROOT / "research"))
import backtest as B  # noqa: E402
from automation.shadow_lc_momentum import entry_date_this_week, live_equity_on  # noqa: E402
from automation.shadow_regime_switch import fetch_daily, build_panel, HISTORY_DAYS  # noqa: E402
from automation.system_logger import log_event, LogStatus  # noqa: E402
from automation.notifier import notify_alert  # noqa: E402

log = logging.getLogger("shadow_slow")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

DATA = ROOT / "data"
STATE = DATA / "shadow_slow_state.json"
LEDGER = DATA / "shadow_slow_ledger.jsonl"
SCORES = DATA / "scores_final.csv"
N, KEEP_WITHIN, COST_BPS = 10, 100, 15.0


def ranks_at(p, entry):
    """12-1 momentum ranks (1 = best) within the research universe at the formation Friday."""
    f = B.features(p)
    U = B.universe(p, f, 5.0, 2e6)
    entry_fri = pd.Timestamp(entry) + pd.Timedelta(days=4 - entry.weekday())
    form = p.weeks[p.weeks.get_loc(entry_fri) - 1]
    s = f["m12_1"].loc[form].where(U.loc[form]).dropna()
    return s.rank(ascending=False), form, s


def main():
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    entry = entry_date_this_week(today)
    if entry is None or today < entry:
        log.info("No entry session yet this week")
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "holdings": {}, "entered": {}, "cash": 100.0, "last_date": None, "nav0_spy": None, "nav0_live": None}
    if state["last_date"] == entry.isoformat():
        log.info(f"Already recorded {entry} -- idempotent skip")
        return

    sc = pd.read_csv(SCORES, low_memory=False)
    names = dict(zip(sc.symbol.astype(str), sc.get("name", sc.symbol).astype(str)))
    names.setdefault("SPY", "SPDR S&P 500 ETF Trust")
    syms = list(names) + list(state["holdings"])
    start = entry - dt.timedelta(days=HISTORY_DAYS)
    adj = fetch_daily(syms, start, entry, "all")
    raw = fetch_daily(syms, start, entry, "raw")
    if adj.empty:
        log.warning("No bars fetched -- will retry next run")
        return
    for d in (adj, raw):
        d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.date
    if entry not in set(adj.date):
        log.warning(f"No closes for {entry} yet -- will retry next run")
        return
    p = build_panel(adj, raw, {k: names.get(k, k) for k in set(syms)}, entry)
    ranks, form, mom = ranks_at(p, entry)

    # 1. mark to market from the last recorded close to this entry-day close
    closes = adj.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    last = dt.date.fromisoformat(state["last_date"]) if state["last_date"] else entry
    held, exited = state["holdings"], []
    for sym in list(held):
        p0 = closes.at[last, sym] if (last in closes.index and sym in closes.columns) else np.nan
        p1 = closes.at[entry, sym] if sym in closes.columns else np.nan
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            held[sym] *= p1 / p0
        elif not np.isfinite(p1):                  # stopped trading: book at last value, exit
            state["cash"] += held.pop(sym)
            exited.append(sym)
    nav = state["cash"] + sum(held.values())

    # 2. keep names still in the top KEEP_WITHIN, fill to N from the top, equal weight
    keep = [h for h in held if h in ranks.index and ranks[h] <= KEEP_WITHIN]
    sold = sorted(set(held) - set(keep))
    fill = [x for x in ranks.sort_values().index if x not in keep][: N - len(keep)]
    names_new = keep + fill
    tgt = {s: nav / len(names_new) for s in names_new}
    traded = sum(abs(tgt.get(s, 0) - held.get(s, 0)) for s in set(tgt) | set(held))
    cost = traded * COST_BPS / 1e4
    nav -= cost
    held = {s: v * nav / sum(tgt.values()) for s, v in tgt.items()}
    entered = state.get("entered", {})
    entered = {s: entered.get(s, entry.isoformat()) for s in held}
    state.update(holdings=held, entered=entered, cash=0.0, last_date=entry.isoformat())

    spy = float(closes.at[entry, "SPY"])
    live = live_equity_on(entry)
    state["nav0_spy"] = state["nav0_spy"] or spy
    state["nav0_live"] = state["nav0_live"] or live
    row = {"date": entry.isoformat(), "formation_week": str(form.date()),
           "shadow_nav": round(nav, 4), "live_equity": live,
           "live_index": round(100 * live / state["nav0_live"], 4) if live and state["nav0_live"] else None,
           "spy_index": round(100 * spy / state["nav0_spy"], 4),
           "turnover_pct": round(100 * traded / (nav + cost), 1),
           "bought": fill, "sold": sold, "exited_no_price": exited,
           "holdings": {s: {"rank": int(ranks[s]) if s in ranks.index else None,
                            "mom_12_1_pct": round(float(mom[s]) * 100, 1) if s in mom.index else None,
                            "since": entered[s]} for s in sorted(held)},
           "universe_size": int(len(ranks))}
    STATE.write_text(json.dumps(state, indent=1))
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    msg = (f"Shadow slow momentum {entry}: nav {row['shadow_nav']:.1f} | live {row['live_index']} | "
           f"SPY {row['spy_index']:.1f} | bought {len(fill)}, sold {len(sold)}")
    log.info(msg)
    log_event("shadow_slow", LogStatus.SUCCESS, msg,
              metrics={k: row[k] for k in ("shadow_nav", "live_index", "spy_index", "turnover_pct")})
    if fill or sold:
        notify_alert("shadow_slow", msg)


if __name__ == "__main__":
    main()
