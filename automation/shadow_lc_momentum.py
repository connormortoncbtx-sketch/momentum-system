"""
automation/shadow_lc_momentum.py
================================
Shadow portfolio (no orders): large-cap monthly momentum, run side by side
with the live weekly system. Design = the best large-cap design from the
Oct 2026 research (docs/2026-10_strategy_research.md):

  universe   500 most-traded names in data/scores_final.csv (last_price * avg_vol_20d)
  score      0.35*RS + 0.25*trend + 0.20*breakout (percentile ranks within the 500)
  basket     30 names, equal weight; rebalance every 4th entry week;
             holdings still ranked in the top 60 are kept (buffer)
  prices     official CLOSE of the week's entry day (Mon, or Tue on holiday weeks),
             split/dividend-adjusted, fetched after the fact -> immune to cron drift
  costs      15 bps per side on traded weight

Each run (idempotent per entry date) appends one row to
data/shadow_lc_ledger.jsonl with shadow NAV, live Alpaca paper equity and SPY,
all on the same entry-day close, so the three lines are directly comparable.
"""
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation.system_logger import log_event, LogStatus  # noqa: E402
from automation.notifier import notify_alert  # noqa: E402

log = logging.getLogger("shadow_lc")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

DATA = Path("data")
STATE = DATA / "shadow_lc_state.json"
LEDGER = DATA / "shadow_lc_ledger.jsonl"
SCORES = DATA / "scores_final.csv"
N, BUFFER, EVERY, TOP_UNIVERSE, COST_BPS = 30, 60, 4, 500, 15.0


def entry_date_this_week(today: dt.date) -> dt.date | None:
    """The week's entry session (Mon, or Tue if Mon is a holiday)."""
    from automation.tz_utils import get_entry_day
    day = get_entry_day(today)
    if day == "skip":
        return None
    monday = today - dt.timedelta(days=today.weekday())
    return monday if day == "monday" else monday + dt.timedelta(days=1)


def daily_closes(symbols, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Adjusted daily closes (date x symbol)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames = []
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 200):
        req = StockBarsRequest(symbol_or_symbols=syms[i:i + 200], timeframe=TimeFrame.Day,
                               start=dt.datetime.combine(start, dt.time()),
                               end=dt.datetime.combine(end + dt.timedelta(days=1), dt.time()),
                               adjustment=Adjustment.ALL, feed=DataFeed.SIP)
        df = client.get_stock_bars(req).df
        if len(df):
            frames.append(df.reset_index()[["symbol", "timestamp", "close"]])
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames)
    d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.date
    return d.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")


def live_equity_on(date: dt.date):
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"],
                           paper=os.environ.get("ALPACA_PAPER", "true").lower() != "false")
        h = tc.get_portfolio_history(GetPortfolioHistoryRequest(period="1M", timeframe="1D"))
        for ts, eq in zip(h.timestamp, h.equity):
            from zoneinfo import ZoneInfo
            if dt.datetime.fromtimestamp(ts, ZoneInfo("America/New_York")).date() == date and eq:
                return float(eq)
    except Exception as e:
        log.warning(f"live equity unavailable: {e}")
    return None


import re
FUND_HARD = re.compile(r"\b(ETF|ETN|FUND|PROSHARES|ISHARES|DIREXION|SPDR|VANGUARD|WISDOMTREE|"
                       r"WARRANTS?|RIGHTS?|UNITS?|PREFERRED|NOTES?|BITCOIN|ETHER)\b", re.I)
FUND_SOFT = re.compile(r"\b(TRUST|INVESCO|INDEX|PORTFOLIO)\b", re.I)
CORP = re.compile(r"\b(CORP(ORATION)?|INC|LTD|LIMITED|COMPANY|CO|PLC|HOLDINGS?|GROUP|BANCORP|BANK|REALTY|PROPERTIES)\b", re.I)


def is_fund(name, sector) -> bool:
    """Same fund rule as research/backtest.py; ETFs also carry no sector."""
    name = name if isinstance(name, str) else ""
    if not isinstance(sector, str) or FUND_HARD.search(name):
        return True
    return bool(FUND_SOFT.search(name)) and not CORP.search(name)


def target_basket(held: list[str]) -> list[str]:
    s = pd.read_csv(SCORES, low_memory=False)
    s = s[(s.last_price >= 5)].copy()
    s = s[~s.apply(lambda r: is_fund(r.get("name"), r.get("sector")), axis=1)]
    s["dollar_vol"] = s.last_price * s.avg_vol_20d
    s = s.nlargest(TOP_UNIVERSE, "dollar_vol")
    r = lambda c: s[c].rank(pct=True)
    s["score"] = (0.35 * r("sig_momentum_rs") + 0.25 * r("sig_momentum_trend")
                  + 0.20 * r("sig_momentum_breakout")) / 0.80
    s = s.dropna(subset=["score"])
    rank = s.set_index("symbol").score.rank(ascending=False)
    keep = [h for h in held if h in rank.index and rank[h] <= BUFFER]
    fill = [x for x in rank.sort_values().index if x not in keep][: N - len(keep)]
    return keep + fill


def main():
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    entry = entry_date_this_week(today)
    if entry is None or today < entry:
        log.info("No entry session yet this week -- nothing to do")
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "holdings": {}, "cash": 100.0, "weeks": 0, "last_date": None, "nav0_live": None, "nav0_spy": None}
    if state["last_date"] == entry.isoformat():
        log.info(f"Already recorded {entry} -- idempotent skip")
        return

    held = state["holdings"]                       # symbol -> dollar value at last_date close
    last = dt.date.fromisoformat(state["last_date"]) if state["last_date"] else entry
    px = daily_closes(list(held) + ["SPY"], last, entry)
    if entry not in px.index:
        log.warning(f"No closes for {entry} yet -- will retry next run")
        return

    # 1. mark to market from last recorded close to this entry-day close
    exited = []
    for sym in list(held):
        p0 = px.at[last, sym] if (last in px.index and sym in px.columns) else np.nan
        p1 = px.at[entry, sym] if sym in px.columns else np.nan
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            held[sym] *= p1 / p0
        elif not np.isfinite(p1):                  # stopped trading (acquired/delisted): cash at last value
            state["cash"] += held.pop(sym)
            exited.append(sym)
    nav = state["cash"] + sum(held.values())

    # 2. rebalance every EVERY entry weeks (first run buys the initial basket)
    traded = 0.0
    rebalanced = state["weeks"] % EVERY == 0
    if rebalanced:
        names = target_basket(list(held))
        tgt = {s: nav / N for s in names}
        traded = sum(abs(tgt.get(s, 0) - held.get(s, 0)) for s in set(tgt) | set(held))
        cost = traded * COST_BPS / 1e4
        held = {s: v - cost / len(tgt) for s, v in tgt.items()}
        state["cash"] = 0.0
        nav -= cost
    state["holdings"], state["weeks"] = held, state["weeks"] + 1

    spy = float(px.at[entry, "SPY"])
    live = live_equity_on(entry)
    state["nav0_spy"] = state["nav0_spy"] or spy
    state["nav0_live"] = state["nav0_live"] or live
    row = {"date": entry.isoformat(), "shadow_nav": round(nav, 4),
           "live_equity": live, "live_index": round(100 * live / state["nav0_live"], 4) if live and state["nav0_live"] else None,
           "spy_close": spy, "spy_index": round(100 * spy / state["nav0_spy"], 4),
           "rebalanced": rebalanced, "turnover_pct": round(100 * traded / nav, 1) if nav else 0,
           "n_holdings": len(held), "exited_no_price": exited, "holdings": sorted(held)}
    state["last_date"] = entry.isoformat()
    STATE.write_text(json.dumps(state, indent=1))
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    msg = (f"Shadow LC momentum {entry}: {row['shadow_nav']:.1f} | live {row['live_index']} | "
           f"SPY {row['spy_index']:.1f}" + (" | rebalanced" if rebalanced else ""))
    log.info(msg)
    log_event("shadow_lc", LogStatus.SUCCESS, msg, metrics={k: row[k] for k in ("shadow_nav", "live_index", "spy_index")})
    if rebalanced:
        notify_alert("shadow_lc", msg)


if __name__ == "__main__":
    main()
