"""
automation/shadow_macro_sized.py
================================
Shadow #3 (no orders): momentum exposure sized by the pass-3c macro rule
(docs/2026-10_strategy_research.md, "Pass 3c").

  monthly   weight = min(1, vol scale x macro on/off), from research/macro_signal.py
            decided at the last month end, with Ken French data used at the 1-month
            lag it is actually published with (backtest-verified: holdout Sharpe
            0.66-0.74 at lag 1-2 vs 0.27 always-on)
  weekly    `weight` of NAV in the top-20 large-cap live-momentum basket (same
            basket as shadow #2's ON state, buffer 40), rest in SPY
  prices    entry-day official closes, fetched after the fact; idempotent per date
  costs     15 bps per side on traded weight

Ledger: data/shadow_macro_ledger.jsonl
"""
import datetime as dt
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import macro_signal  # noqa: E402
import fetch_macro  # noqa: E402
from automation.shadow_lc_momentum import entry_date_this_week, live_equity_on  # noqa: E402
from automation.shadow_regime_switch import fetch_daily, build_panel, decide, SCORES, HISTORY_DAYS  # noqa: E402
from automation.system_logger import log_event, LogStatus  # noqa: E402
from automation.notifier import notify_alert  # noqa: E402

log = logging.getLogger("shadow_macro")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

DATA = ROOT / "data"
STATE = DATA / "shadow_macro_state.json"
LEDGER = DATA / "shadow_macro_ledger.jsonl"
N, BUFFER, COST_BPS, FRENCH_LAG = 20, 40, 15.0, 1


def month_weight(entry: dt.date) -> dict:
    """Macro weight decided at the month end before `entry`."""
    decision = (pd.Timestamp(entry) - pd.offsets.MonthEnd(1)).normalize()
    tmp = Path(tempfile.mkdtemp())
    fetch_macro.fred(tmp)
    fetch_macro.french(tmp)
    w, det = macro_signal.build(tmp, french_lag=FRENCH_LAG, end=decision)
    row = det.loc[:decision].dropna(subset=["weight"])
    if row.empty:
        raise RuntimeError("macro weight unavailable")
    last = row.iloc[-1]
    french_end = pd.read_parquet(tmp / "french_momentum.parquet").date.max()
    return {"decision_month": str(decision.date()), "weight_used_from": str(row.index[-1].date()),
            "raw_weight": float(last.weight), "weight": float(min(1.0, last.weight)),
            "vol_scale": float(last.vol_scale), "macro_on": bool(last.ridge_on),
            "ridge_pred_pct": round(float(last.ridge_pred) * 100, 3),
            "momentum_vol6_pct": round(float(last.umd_vol6) * 100, 1),
            "french_data_through": str(french_end.date())}


def main():
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    entry = entry_date_this_week(today)
    if entry is None or today < entry:
        log.info("No entry session yet this week")
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "holdings": {}, "cash": 100.0, "last_date": None, "nav0_spy": None, "nav0_live": None, "months": {}}
    if state["last_date"] == entry.isoformat():
        log.info(f"Already recorded {entry} -- idempotent skip")
        return

    mkey = entry.strftime("%Y-%m")
    if mkey not in state["months"]:
        state["months"][mkey] = month_weight(entry)
        log.info(f"Macro weight for {mkey}: {state['months'][mkey]}")
    mw = state["months"][mkey]
    w = mw["weight"]

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
    _, _, ranks, form, _ = decide(p, entry)

    closes = adj.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    last = dt.date.fromisoformat(state["last_date"]) if state["last_date"] else entry
    held = state["holdings"]
    for sym in list(held):
        p0 = closes.at[last, sym] if (last in closes.index and sym in closes.columns) else np.nan
        p1 = closes.at[entry, sym] if sym in closes.columns else np.nan
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            held[sym] *= p1 / p0
        elif not np.isfinite(p1):
            state["cash"] += held.pop(sym)
    nav = state["cash"] + sum(held.values())

    keep = [h for h in held if h != "SPY" and h in ranks.index and ranks[h] <= BUFFER]
    fill = [x for x in ranks.sort_values().index if x not in keep][: N - len(keep)]
    tgt = {s: nav * w / N for s in keep + fill} if w > 0 else {}
    if w < 1:
        tgt["SPY"] = nav * (1 - w)
    traded = sum(abs(tgt.get(s, 0) - held.get(s, 0)) for s in set(tgt) | set(held))
    cost = traded * COST_BPS / 1e4
    nav -= cost
    held = {s: v * (nav / sum(tgt.values())) for s, v in tgt.items()}
    state.update(holdings=held, cash=0.0, last_date=entry.isoformat())

    spy = float(closes.at[entry, "SPY"])
    live = live_equity_on(entry)
    state["nav0_spy"] = state["nav0_spy"] or spy
    state["nav0_live"] = state["nav0_live"] or live
    row = {"date": entry.isoformat(), "formation_week": str(form.date()),
           "momentum_weight": round(w, 3), "macro_on": mw["macro_on"], "vol_scale": round(mw["vol_scale"], 3),
           "shadow_nav": round(nav, 4), "live_equity": live,
           "live_index": round(100 * live / state["nav0_live"], 4) if live and state["nav0_live"] else None,
           "spy_index": round(100 * spy / state["nav0_spy"], 4),
           "turnover_pct": round(100 * traded / (nav + cost), 1),
           "french_data_through": mw["french_data_through"], "holdings": sorted(held)}
    STATE.write_text(json.dumps(state, indent=1))
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    msg = (f"Shadow macro-sized {entry}: momentum {w:.0%} / SPY {1 - w:.0%} "
           f"(macro {'ON' if mw['macro_on'] else 'OFF'}, vol scale {mw['vol_scale']:.2f}) "
           f"nav {row['shadow_nav']:.1f} | live {row['live_index']} | SPY {row['spy_index']:.1f}")
    log.info(msg)
    log_event("shadow_macro", LogStatus.SUCCESS, msg,
              metrics={k: row[k] for k in ("shadow_nav", "live_index", "spy_index", "momentum_weight")})
    prev = state.get("prev_on")
    if prev is not None and prev != mw["macro_on"]:
        notify_alert("shadow_macro", "Macro switch flipped: " + msg)
    state["prev_on"] = mw["macro_on"]
    STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
