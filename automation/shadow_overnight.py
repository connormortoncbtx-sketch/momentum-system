"""
automation/shadow_overnight.py
==============================
Shadow #8 (no orders): OVERNIGHT-ONLY momentum (research/overnight_momentum.py, Oct 17, 2026).

  picks     the LIVE-proxy composite (0.35 RS + 0.25 trend + 0.20 52-week-high ranks, research
            universe: price >= $5, >= $2M/day), top 10, formed each Friday close exactly as in the
            research (research/backtest.py features via the shadow panel builder)
  holding   buy every name at the CLOSE (closing auction) and sell at the next OPEN (opening
            auction), every night incl. the weekend night, from the Monday close after formation
            until the next formation's Monday close
  prices    official daily open / close prints (Alpaca SIP, adjusted), fetched after the fact
  books     raw at 1 and 2 bp per side; IWM-hedged (beta 0.67, fitted on 2017-23) at 1 bp;
            the same basket's DAYTIME return is logged for the tug-of-war check
  backtest  dev pick at 1 bp: raw 28%/yr (Sharpe 1.68), hedged Sharpe 1.27 (2017-23) -> 0.77 (2024-26)

It does not trade: official prints stand in for auction fills. Whether real MOC/MOO fills match
the prints (the cost question that decides this strategy) can only be measured with real orders.
State: data/shadow_overnight_state.json   Ledger: data/shadow_overnight_ledger.jsonl (one row per night)
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
from automation.shadow_regime_switch import fetch_daily, build_panel, HISTORY_DAYS  # noqa: E402
from automation.system_logger import log_event, LogStatus  # noqa: E402

log = logging.getLogger("shadow_overnight")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
DATA = ROOT / "data"
STATE, LEDGER, SCORES = DATA / "shadow_overnight_state.json", DATA / "shadow_overnight_ledger.jsonl", DATA / "scores_final.csv"
N, BETA = 10, 0.67
START = dt.date(2026, 10, 19)          # first entry close of the shadow (Monday)


def form_friday(d):
    """Most recent Friday strictly before date d (the formation for entries on/after the Monday after it)."""
    back = (d.weekday() - 4) % 7 or 7
    return d - dt.timedelta(days=back)


def picks_for(fri, state):
    key = fri.isoformat()
    if key in state["picks"]:
        return state["picks"][key]
    sc = pd.read_csv(SCORES, low_memory=False)
    names = dict(zip(sc.symbol.astype(str), sc.get("name", sc.symbol).astype(str)))
    syms = list(names) + ["SPY"]
    adj = fetch_daily(syms, fri - dt.timedelta(days=HISTORY_DAYS), fri, "all")
    raw = fetch_daily(syms, fri - dt.timedelta(days=HISTORY_DAYS), fri, "raw")
    for d in (adj, raw):
        d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.date
    if fri not in set(adj.date):
        raise RuntimeError(f"no closes for formation {fri} yet")
    # build_panel labels weeks by Friday; pass the following Monday so the formation week is complete
    p = build_panel(adj, raw, {k: names.get(k, k) for k in set(syms)}, fri + dt.timedelta(days=3))
    f = B.features(p); U = B.universe(p, f, 5.0, 2e6)
    rk = lambda x: B.xrank(x, U)
    live = (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80
    w = pd.Timestamp(fri)
    s = live.loc[w].where(U.loc[w]).dropna().sort_values(ascending=False)
    top = list(s.index[:N])
    state["picks"][key] = top
    log.info(f"formation {fri}: {top}")
    return top


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "picks": {}, "last_open_date": None,
        "nav": {"raw_1bp": 100.0, "raw_2bp": 100.0, "hedged_1bp": 100.0, "day_only": 100.0, "iwm_night": 100.0}}
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo("America/New_York")).date()
    if today <= START:
        log.info(f"Shadow starts with the {START} close; first night marks the next morning")
        return
    # nights to mark: trading days t (open dates) after START and after the last marked one
    look_from = START if not state["last_open_date"] else dt.date.fromisoformat(state["last_open_date"])
    # gather all formation baskets that may be needed
    bars = fetch_daily(["IWM", "SPY"], look_from - dt.timedelta(days=10), today, "all")
    bars["date"] = pd.to_datetime(bars.timestamp).dt.tz_convert("America/New_York").dt.date
    tdays = sorted(d for d in set(bars.date) if d > look_from)
    alldays = sorted(set(bars.date))
    if not tdays:
        log.info("No new trading day to mark"); return
    rows = []
    for t in tdays:
        prev = max(d for d in alldays if d < t)                  # entry close of this night
        fri = form_friday(prev)
        first_entry = min((d for d in alldays if d > fri), default=None)   # Monday close after formation
        if first_entry is None or prev < first_entry:
            fri = form_friday(fri)                               # still on the previous formation's basket
        try:
            names = picks_for(fri, state)
        except Exception as e:
            log.warning(f"picks for {fri}: {e} -- stop and retry next run"); break
        px = fetch_daily(names + ["IWM"], prev - dt.timedelta(days=5), t, "all")
        px["date"] = pd.to_datetime(px.timestamp).dt.tz_convert("America/New_York").dt.date
        O = px.pivot_table(index="date", columns="symbol", values="open")
        C = px.pivot_table(index="date", columns="symbol", values="close")
        if t not in O.index or prev not in C.index:
            log.warning(f"bars for {prev}->{t} not complete -- retry next run"); break
        nights = (O.loc[t] / C.loc[prev] - 1).reindex(names).dropna()
        # daytime return of the same basket on the entry day (complete session; today's close isn't final yet)
        days = (C.loc[prev] / O.loc[prev] - 1).reindex(names).dropna() if prev in O.index else pd.Series(dtype=float)
        iwm = float(O.at[t, "IWM"] / C.at[prev, "IWM"] - 1)
        r = float(nights.mean()) if len(nights) else 0.0
        nav = state["nav"]
        nav["raw_1bp"] *= 1 + r - 2e-4
        nav["raw_2bp"] *= 1 + r - 4e-4
        nav["hedged_1bp"] *= 1 + r - 2e-4 - BETA * iwm
        nav["day_only"] *= 1 + (float(days.mean()) if len(days) else 0.0)
        nav["iwm_night"] *= 1 + iwm
        rows.append({"open_date": t.isoformat(), "entry_close": prev.isoformat(), "formation": fri.isoformat(),
                     "weekend": (t - prev).days > 1, "basket_night_bp": round(r * 1e4, 2),
                     "median_name_bp": round(float(nights.median()) * 1e4, 2) if len(nights) else None,
                     "iwm_night_bp": round(iwm * 1e4, 2), "entry_day_daytime_bp": round(float(days.mean()) * 1e4, 2) if len(days) else None,
                     "names": {k: round(float(v) * 1e4, 1) for k, v in nights.items()},
                     "nav": {k: round(v, 3) for k, v in nav.items()}})
        state["last_open_date"] = t.isoformat()
    if rows:
        with LEDGER.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    STATE.write_text(json.dumps(state, indent=1))
    msg = f"Shadow overnight: {len(rows)} night(s) marked; NAV " + ", ".join(f"{k} {v:.2f}" for k, v in state["nav"].items())
    log.info(msg)
    log_event("shadow_overnight", LogStatus.SUCCESS, msg, metrics=state["nav"])


if __name__ == "__main__":
    main()
