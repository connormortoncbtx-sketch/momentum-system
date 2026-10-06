"""
automation/shadow_regime_switch.py
==================================
Shadow #2 (no orders): the contrarian regime switch from Round 5 of
docs/2026-10_strategy_research.md, using the SAME research code
(research/backtest.py, research/fetch_history.py) on a live price panel.

Each entry day (Mon, or Tue on holiday weeks), at the official close:
  factor   = weekly excess return of the top decile by the live-momentum proxy
             over the equal-weight universe (price >= $5, $2M/day, no funds)
  switch   = trailing 8-week factor sum (through today's close) <= 0  -> ON
  ON       = top 20 of the 500 most-traded names by live-momentum proxy,
             equal weight, holdings kept while ranked <= 40
  OFF      = 100% SPY
  costs    = 15 bps per side on traded weight

Universe source: the live pipeline's data/scores_final.csv symbols (~1,800),
so the factor is computed on a slightly smaller universe than the backtest.
Ledger: data/shadow_regime_ledger.jsonl (idempotent per entry date).
"""
import datetime as dt
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import backtest as B  # noqa: E402
from fetch_history import weekly, weekly_raw  # noqa: E402
from automation.shadow_lc_momentum import entry_date_this_week, live_equity_on  # noqa: E402
from automation.system_logger import log_event, LogStatus  # noqa: E402
from automation.notifier import notify_alert  # noqa: E402

log = logging.getLogger("shadow_regime")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

DATA = ROOT / "data"
STATE = DATA / "shadow_regime_state.json"
LEDGER = DATA / "shadow_regime_ledger.jsonl"
SCORES = DATA / "scores_final.csv"
N, BUFFER, LOOKBACK_W, COST_BPS, HISTORY_DAYS = 20, 40, 8, 15.0, 480


def fetch_daily(symbols, start, end, adjustment):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames = []
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 200):
        try:
            req = StockBarsRequest(symbol_or_symbols=syms[i:i + 200], timeframe=TimeFrame.Day,
                                   start=dt.datetime.combine(start, dt.time()),
                                   end=dt.datetime.combine(end + dt.timedelta(days=1), dt.time()),
                                   adjustment=Adjustment(adjustment), feed=DataFeed.SIP)
            df = client.get_stock_bars(req).df
            if len(df):
                frames.append(df.reset_index()[["symbol", "timestamp", "open", "high", "low", "close", "volume"]])
        except Exception as e:
            log.warning(f"  bars batch {i // 200} failed: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_panel(daily_adj, daily_raw, names, entry):
    """Weekly panel in exactly the research format, loaded through research code."""
    w = weekly(daily_adj)
    wr = weekly_raw(daily_raw)
    w = w.merge(wr, on=["symbol", "week"], how="left")
    tmp = Path(tempfile.mkdtemp())
    for yr, chunk in w.groupby(w.week.dt.year):
        chunk.to_parquet(tmp / f"weekly_{yr}.parquet", index=False)
    pd.DataFrame({"symbol": list(names), "name": list(names.values()), "status": "active",
                  "has_data": True}).to_parquet(tmp / "assets.parquet", index=False)
    fri = pd.Timestamp(entry) + pd.Timedelta(days=4 - entry.weekday())
    return B.load_panel(str(tmp), str(fri.date()))


def decide(p, entry):
    f = B.features(p)
    U = B.universe(p, f, 5.0, 2e6)
    U500 = U & (f["adv"].where(U).rank(axis=1, ascending=False) <= 500)
    rk = lambda x, m: B.xrank(x, m)
    proxy = lambda m: (0.35 * rk(f["rs_live"], m) + 0.25 * rk(f["trend"], m) + 0.20 * rk(f["hi52"], m)) / 0.8
    d1c = p.df("d1_close")
    wk_ret = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)
    s_base = proxy(U)
    top = s_base.where(U & wk_ret.notna()).rank(axis=1, pct=True) > 0.9
    fx = wk_ret.where(top).mean(axis=1) - wk_ret.where(U).mean(axis=1)
    entry_fri = pd.Timestamp(entry) + pd.Timedelta(days=4 - entry.weekday())
    t = p.weeks.get_loc(entry_fri) - 1                 # formation week = previous Friday
    form = p.weeks[t]
    trail = fx.iloc[t - LOOKBACK_W:t].sum(min_count=LOOKBACK_W)   # = rolling(8).sum().shift(1) at t
    on = bool(np.isfinite(trail) and trail <= 0)
    ranks = proxy(U500).loc[form].where(U500.loc[form]).dropna().rank(ascending=False)
    return on, float(trail) if np.isfinite(trail) else None, ranks, form, d1c


def main():
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    entry = entry_date_this_week(today)
    if entry is None or today < entry:
        log.info("No entry session yet this week")
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "holdings": {}, "cash": 100.0, "last_date": None, "nav0_spy": None, "nav0_live": None}
    if state["last_date"] == entry.isoformat():
        log.info(f"Already recorded {entry} -- idempotent skip")
        return

    sc = pd.read_csv(SCORES, low_memory=False)
    names = dict(zip(sc.symbol.astype(str), sc.get("name", sc.symbol).astype(str)))
    names.setdefault("SPY", "SPDR S&P 500 ETF Trust")
    syms = list(names) + list(state["holdings"])
    start = entry - dt.timedelta(days=HISTORY_DAYS)
    adj = fetch_daily(syms, start, entry, "all")
    raw = fetch_daily(syms, start, entry, "raw")   # price filter + reverse-split screen need the full window
    if adj.empty:
        log.warning("No bars fetched -- will retry next run")
        return
    for d in (adj, raw):
        d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.date
    if entry not in set(adj.date):
        log.warning(f"No closes for {entry} yet -- will retry next run")
        return
    p = build_panel(adj, raw, {k: names.get(k, k) for k in set(syms)}, entry)
    on, trail, ranks, form, _ = decide(p, entry)

    closes = adj.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    last = dt.date.fromisoformat(state["last_date"]) if state["last_date"] else entry
    held = state["holdings"]
    for sym in list(held):                                       # mark to market
        p0 = closes.at[last, sym] if (last in closes.index and sym in closes.columns) else np.nan
        p1 = closes.at[entry, sym] if sym in closes.columns else np.nan
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            held[sym] *= p1 / p0
        elif not np.isfinite(p1):
            state["cash"] += held.pop(sym)
    nav = state["cash"] + sum(held.values())

    if on:
        keep = [h for h in held if h != "SPY" and h in ranks.index and ranks[h] <= BUFFER]
        fill = [x for x in ranks.sort_values().index if x not in keep][: N - len(keep)]
        tgt = {s: nav / N for s in keep + fill}
    else:
        tgt = {"SPY": nav}
    traded = sum(abs(tgt.get(s, 0) - held.get(s, 0)) for s in set(tgt) | set(held))
    cost = traded * COST_BPS / 1e4
    nav -= cost
    held = {s: v * (nav / sum(tgt.values())) for s, v in tgt.items()}
    state.update(holdings=held, cash=0.0, last_date=entry.isoformat())

    spy = float(closes.at[entry, "SPY"])
    live = live_equity_on(entry)
    state["nav0_spy"] = state["nav0_spy"] or spy
    state["nav0_live"] = state["nav0_live"] or live
    row = {"date": entry.isoformat(), "formation_week": str(form.date()), "state": "ON" if on else "OFF (SPY)",
           "trailing_8w_factor_pct": round(trail * 100, 3) if trail is not None else None,
           "shadow_nav": round(nav, 4), "live_equity": live,
           "live_index": round(100 * live / state["nav0_live"], 4) if live and state["nav0_live"] else None,
           "spy_index": round(100 * spy / state["nav0_spy"], 4),
           "turnover_pct": round(100 * traded / (nav + cost), 1), "holdings": sorted(held)}
    STATE.write_text(json.dumps(state, indent=1))
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    msg = (f"Shadow regime switch {entry}: {row['state']} (trail8 {row['trailing_8w_factor_pct']}%) "
           f"nav {row['shadow_nav']:.1f} | live {row['live_index']} | SPY {row['spy_index']:.1f}")
    log.info(msg)
    log_event("shadow_regime", LogStatus.SUCCESS, msg,
              metrics={k: row[k] for k in ("shadow_nav", "live_index", "spy_index", "trailing_8w_factor_pct")})
    prev_on = state.get("prev_on")
    if prev_on is not None and prev_on != on:
        notify_alert("shadow_regime", "Regime switch flipped: " + msg)
    state["prev_on"] = on
    STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
