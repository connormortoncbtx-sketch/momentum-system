"""
automation/shadow_fip_veto.py
=============================
Shadow #9 (no orders), approved Nov 14, 2026: steady-gains 12-1 momentum with the chart-CNN veto -- the best stock
book from the Nov 13-14 research (docs/2026-10_strategy_research.md: "Trend quality among momentum stocks",
"CNN as a veto", reruns on the delisted-inclusive panel).

  universe  live pipeline symbols (data/scores_final.csv): price >= $5, 4-week $ volume >= $2M/day, >= 53 weeks,
            no funds, no reverse split in 52 weeks (research/backtest.py universe())
  momentum  12-1: top 20% of the universe at the formation Friday
  steady    "frog in the pan" (Da, Gurun & Warachka 2014): within that top 20%, the half with the most continuous
            gains (lowest sign(12-1) x (% down days - % up days) over the 12-1 window)
  veto      chart-image CNN (models/cnn_v1, trained 2017-21 incl. delisted stocks, frozen): no stock in the CNN's
            bottom 10% of the day's scored universe
  book      top 30 eligible by 12-1, equal weight, rebalanced every 4th entry week; a holding is kept while it stays
            eligible and ranks within the top 60 (as backtested); marked to market weekly; 15 bps per side
  backtest  delisted-inclusive panel: 2022-23 +20.4%/yr, Sharpe 0.72, max DD -27%; 2024-26 +52.6%/yr, Sharpe 1.21,
            max DD -36% (research/cnn_veto.py). Without the veto 2017-23: +17.5%/yr, Sharpe 0.61 (trend_quality.py).
            Many variants were tried on the way here -> the level is optimistic; judged at the Jan 5 review with the
            deflated-Sharpe / overfitting checks.
Ledger: data/shadow_fipveto_ledger.jsonl
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
import chart_images as CI  # noqa: E402
from automation.shadow_lc_momentum import entry_date_this_week, live_equity_on  # noqa: E402
from automation.shadow_regime_switch import fetch_daily, build_panel, HISTORY_DAYS  # noqa: E402
from automation.system_logger import log_event, LogStatus  # noqa: E402
from automation.notifier import notify_alert  # noqa: E402

log = logging.getLogger("shadow_fipveto")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

DATA = ROOT / "data"
STATE = DATA / "shadow_fipveto_state.json"
LEDGER = DATA / "shadow_fipveto_ledger.jsonl"
SCORES = DATA / "scores_final.csv"
MODELS = [ROOT / "models" / "cnn_v1" / f"chart_cnn_seed{i}.pt" for i in (0, 1)]
N, BUFFER, REBAL_WEEKS, COST_BPS, VETO = 30, 60, 4, 15.0, 0.10


def fip_at(adj, day):
    """FIP per symbol at `day` from adjusted daily closes, same definition as research/trend_quality.py."""
    C = adj.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    C = C[C.index <= day]
    r = C.pct_change(fill_method=None).where(lambda x: x.abs() < 0.7)
    s = np.sign(r)
    up = (s > 0).astype("float32").where(r.notna()).shift(21).rolling(231, min_periods=180).mean()
    dn = (s < 0).astype("float32").where(r.notna()).shift(21).rolling(231, min_periods=180).mean()
    mom = C.shift(21) / C.shift(252) - 1
    return (np.sign(mom) * (dn - up)).iloc[-1]


def eligible_at(p, adj, form):
    f = B.features(p)
    U = B.universe(p, f, 5.0, 2e6)
    m12 = f["m12_1"].loc[form].where(U.loc[form]).dropna()
    top = m12[m12.rank(pct=True) >= 0.8]
    fday = adj[adj.date <= form.date()].date.max()
    fip = fip_at(adj, fday).reindex(top.index)
    steady = top[(-fip).rank(pct=True) >= 0.5]
    daily = adj.assign(date=pd.to_datetime(adj.date))
    X, syms = CI.images_at(daily[daily.date >= pd.Timestamp(fday) - pd.Timedelta(days=90)], pd.Timestamp(fday))
    cnn = pd.Series(CI.score(CI.load_models(MODELS), X), index=syms).rank(pct=True)
    vetoed = sorted(s for s in steady.index if s in cnn.index and cnn[s] <= VETO)
    elig = steady.drop(vetoed)
    return elig.rank(ascending=False), m12, fip, cnn, vetoed, len(U.loc[form][U.loc[form]]), len(top)


def main():
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    entry = entry_date_this_week(today)
    if entry is None or today < entry:
        log.info("No entry session yet this week"); return
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "holdings": {}, "entered": {}, "cash": 100.0, "last_date": None, "weeks_since_rebal": None,
        "nav0_spy": None, "nav0_live": None}
    if state["last_date"] == entry.isoformat():
        log.info(f"Already recorded {entry} -- idempotent skip"); return
    sc = pd.read_csv(SCORES, low_memory=False)
    names = dict(zip(sc.symbol.astype(str), sc.get("name", sc.symbol).astype(str)))
    names.setdefault("SPY", "SPDR S&P 500 ETF Trust")
    syms = list(names) + list(state["holdings"])
    start = entry - dt.timedelta(days=max(HISTORY_DAYS, 420))
    adj = fetch_daily(syms, start, entry, "all")
    raw = fetch_daily(syms, start, entry, "raw")
    if adj.empty:
        log.warning("No bars fetched -- will retry next run"); return
    for d in (adj, raw):
        d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.date
    if entry not in set(adj.date):
        log.warning(f"No closes for {entry} yet -- will retry next run"); return
    p = build_panel(adj, raw, {k: names.get(k, k) for k in set(syms)}, entry)
    entry_fri = pd.Timestamp(entry) + pd.Timedelta(days=4 - entry.weekday())
    form = p.weeks[p.weeks.get_loc(entry_fri) - 1]

    closes = adj.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    last = dt.date.fromisoformat(state["last_date"]) if state["last_date"] else entry
    held, exited = state["holdings"], []
    for sym in list(held):
        p0 = closes.at[last, sym] if (last in closes.index and sym in closes.columns) else np.nan
        p1 = closes.at[entry, sym] if sym in closes.columns else np.nan
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            held[sym] *= p1 / p0
        elif not np.isfinite(p1):
            state["cash"] += held.pop(sym); exited.append(sym)
    nav = state["cash"] + sum(held.values())

    wsr = state.get("weeks_since_rebal")
    rebal = wsr is None or wsr + 1 >= REBAL_WEEKS or not held
    bought, sold, vetoed, info = [], [], [], {}
    if rebal:
        ranks, m12, fip, cnn, vetoed, n_univ, n_top = eligible_at(p, adj, form)
        keep = [h for h in held if h in ranks.index and ranks[h] <= BUFFER]
        bought = [x for x in ranks.sort_values().index if x not in keep][: N - len(keep)]
        names_new = keep + bought
        sold = sorted(set(held) - set(names_new))
        tgt = {s: nav / len(names_new) for s in names_new} if names_new else {}
        traded = sum(abs(tgt.get(s, 0) - held.get(s, 0)) for s in set(tgt) | set(held))
        cost = traded * COST_BPS / 1e4
        nav -= cost
        held = {s: v * nav / sum(tgt.values()) for s, v in tgt.items()} if tgt else {}
        state["cash"] = 0.0 if held else nav
        state["weeks_since_rebal"] = 0
        info = dict(universe=n_univ, top20=n_top, eligible=int(len(ranks)), vetoed=vetoed,
                    turnover_pct=round(100 * traded / (nav + cost), 1),
                    holdings={s: {"mom_12_1_pct": round(float(m12[s]) * 100, 1) if s in m12.index else None,
                                  "fip": round(float(fip[s]), 3) if s in fip.index and np.isfinite(fip[s]) else None,
                                  "cnn_pct": round(float(cnn[s]), 3) if s in cnn.index else None} for s in sorted(held)})
    else:
        state["cash"] = nav - sum(held.values())
        state["weeks_since_rebal"] = wsr + 1
    entered = state.get("entered", {})
    entered = {s: entered.get(s, entry.isoformat()) for s in held}
    state.update(holdings=held, entered=entered, last_date=entry.isoformat())
    spy = float(closes.at[entry, "SPY"])
    live = live_equity_on(entry)
    state["nav0_spy"] = state["nav0_spy"] or spy
    state["nav0_live"] = state["nav0_live"] or live
    row = {"date": entry.isoformat(), "formation_week": str(form.date()), "rebalanced": rebal,
           "shadow_nav": round(nav, 4), "live_equity": live,
           "live_index": round(100 * live / state["nav0_live"], 4) if live and state["nav0_live"] else None,
           "spy_index": round(100 * spy / state["nav0_spy"], 4),
           "bought": bought, "sold": sold, "exited_no_price": exited, **info}
    STATE.write_text(json.dumps(state, indent=1))
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    msg = (f"Shadow FIP+veto {entry}: nav {row['shadow_nav']:.1f} | SPY {row['spy_index']:.1f} | "
           + (f"rebalanced: bought {len(bought)}, sold {len(sold)}, vetoed {len(vetoed)}" if rebal else "held"))
    log.info(msg)
    log_event("shadow_fipveto", LogStatus.SUCCESS, msg, metrics={k: row[k] for k in ("shadow_nav", "spy_index")})
    if rebal:
        notify_alert("shadow_fipveto", msg)


if __name__ == "__main__":
    main()
