"""
automation/shadow_stack.py
==========================
Shadow #7 (no orders): the diversified, market-hedged STRATEGY STACK from
docs/2026-10_strategy_research.md ("Strategy stack" / "More books, and hedging out the market"),
restricted to the books that exist live. Weights and hedge ratios were fixed on 2017-2023
data only (research/hedged_stack.py) and are frozen here:

  SLOW_h      14.5%  shadow #4 slow 12-1 momentum (data/shadow_slow_ledger.jsonl), minus 0.798 x IWM
  CONTRACT_h  35.3%  shadow #6 diversified contract-win book (state["div"]["history"]), minus 0.743 x IWM
  ETFTREND    29.4%  41 cross-asset ETFs, 63-day trend, long-only, 10% vol target per asset,
                     monthly at month-end close, 5 bps per unit traded (research/trend_test.py)
  CRYPTO      20.8%  16 coins, long when above the 50-day average (decided at yesterday's
                     close), 40% vol target per coin, equal weight, 25 bps (research/trend_test.py)
Backtest of exactly this mix: Sharpe 1.35 (2017-23) / 1.72 (2024-26); realistic live
expectation lower (see the research log).

Marked once per week over the same window as shadow #4 (its Monday-close to Monday-close
dates): ETF and crypto books are recomputed from daily bars each run (deterministic, no
state), the contract book is read from shadow #6's daily NAV, IWM from daily bars.
Tracks the stack NAV at 1x, 2x and 3x (financing 6%/yr on the borrowed part) and each sleeve.
Ledger: data/shadow_stack_ledger.jsonl   State: data/shadow_stack_state.json
"""
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation.system_logger import log_event, LogStatus  # noqa: E402

log = logging.getLogger("shadow_stack")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
DATA = ROOT / "data"
STATE, LEDGER = DATA / "shadow_stack_state.json", DATA / "shadow_stack_ledger.jsonl"
SLOW_LEDGER, FILINGS_STATE = DATA / "shadow_slow_ledger.jsonl", DATA / "shadow_filings_state.json"
W = {"SLOW_h": 0.145, "CONTRACT_h": 0.353, "ETFTREND": 0.294, "CRYPTO": 0.208}
BETA = {"SLOW_h": 0.798, "CONTRACT_h": 0.743}
LEVERAGE, FIN_YR = (1.0, 2.0, 3.0), 0.06
ETFS = ['BITO', 'COPX', 'DBA', 'DBC', 'DIA', 'EEM', 'EFA', 'EWJ', 'FXB', 'FXE', 'FXI', 'FXY', 'GDX', 'GLD', 'HYG',
        'IBIT', 'IEF', 'IWM', 'LQD', 'QQQ', 'SHY', 'SLV', 'SMH', 'SPY', 'TIP', 'TLT', 'UNG', 'URA', 'USO', 'UUP',
        'VNQ', 'XBI', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
COINS = ['AAVE/USD', 'AVAX/USD', 'BCH/USD', 'BTC/USD', 'DOGE/USD', 'DOT/USD', 'ETH/USD', 'GRT/USD', 'LINK/USD',
         'LTC/USD', 'MKR/USD', 'SHIB/USD', 'SOL/USD', 'UNI/USD', 'XRP/USD', 'XTZ/USD']


def etf_daily(start):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    d = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=ETFS, timeframe=TimeFrame.Day,
                                          start=dt.datetime.combine(start, dt.time()), adjustment=Adjustment.ALL,
                                          feed=DataFeed.SIP)).df.reset_index()
    d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return d.pivot_table(index="date", columns="symbol", values="close").sort_index()


def crypto_daily(start):
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    c = CryptoHistoricalDataClient()
    d = c.get_crypto_bars(CryptoBarsRequest(symbol_or_symbols=COINS, timeframe=TimeFrame.Day,
                                            start=dt.datetime.combine(start, dt.time()))).df.reset_index()
    d["date"] = pd.to_datetime(d.timestamp).dt.tz_localize(None).dt.normalize()
    return d.pivot_table(index="date", columns="symbol", values="close").sort_index()


def etf_book(px):
    """Daily returns of the ETF trend book (same rules as research/trend_test.py, 63-day, long-only)."""
    ret = px.pct_change(); vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    me = [px.index[px.index <= m][-1] for m in px.resample("ME").last().index if (px.index <= m).any()]
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns); prev = pd.Series(0.0, index=px.columns)
    cost = pd.Series(0.0, index=px.index)
    for i, d in enumerate(me):
        if px.index.get_loc(d) < 63:
            continue
        sig = np.sign(px.loc[d] / px.shift(63).loc[d] - 1).clip(lower=0)
        tgt = (sig * (0.10 / vol.loc[d]).clip(upper=2.0)).fillna(0)
        tgt = tgt / max((tgt != 0).sum(), 1) * 4
        nxt = me[i + 1] if i + 1 < len(me) else px.index[-1] + pd.Timedelta(days=1)
        w.loc[(w.index > d) & (w.index <= nxt)] = tgt.values
        after = px.index[px.index > d]
        if len(after):
            cost.loc[after[0]] = (tgt - prev).abs().sum() * 5 / 1e4
        prev = tgt
    return (w * ret.fillna(0)).sum(axis=1) - cost


def crypto_book(cp):
    cr = cp.pct_change(); cv = cr.rolling(30, min_periods=20).std() * np.sqrt(365)
    sig = (cp > cp.rolling(50, min_periods=50).mean()).astype(float).shift(1)
    w = (sig * (0.40 / cv.shift(1)).clip(upper=1.5)).where(cp.notna())
    w = w.div(w.notna().sum(axis=1).clip(lower=1), axis=0).fillna(0)
    return (w * cr.fillna(0)).sum(axis=1) - w.diff().abs().sum(axis=1) * 25 / 1e4


def window_ret(daily, a, b):
    """Compound daily returns on dates in (a, b]."""
    x = daily[(daily.index > pd.Timestamp(a)) & (daily.index <= pd.Timestamp(b))]
    return float((1 + x).prod() - 1) if len(x) else 0.0


def main():
    rows = [json.loads(l) for l in SLOW_LEDGER.read_text().splitlines() if l.strip()] if SLOW_LEDGER.exists() else []
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "start": None, "last_date": None, "nav": {f"{k:g}x": 100.0 for k in LEVERAGE},
        "sleeves": {k: 100.0 for k in W}}
    if not rows:
        log.info("Shadow #4 has no marks yet"); return
    if state["start"] is None:
        state["start"] = state["last_date"] = rows[-1]["date"]
        STATE.write_text(json.dumps(state, indent=1))
        log.info(f"Started at shadow #4's {state['start']} mark"); return
    new = [r for r in rows if r["date"] > state["last_date"]]
    if not new:
        log.info(f"No new shadow #4 mark since {state['last_date']}"); return
    prev_nav = {r["date"]: r["shadow_nav"] for r in rows}
    start = dt.date.fromisoformat(state["last_date"]) - dt.timedelta(days=420)
    px = etf_daily(start); etf = etf_book(px)
    iwm = px["IWM"].pct_change().fillna(0)
    cry = crypto_book(crypto_daily(start))
    hist = {pd.Timestamp(k): v for k, v in json.loads(FILINGS_STATE.read_text()).get("div", {}).get("history", {}).items()} \
        if FILINGS_STATE.exists() else {}
    cn = pd.Series(hist).sort_index()
    with LEDGER.open("a") as lf:
        for r in new:
            a, b = state["last_date"], r["date"]
            r_iwm = window_ret(iwm, a, b)
            slow = r["shadow_nav"] / prev_nav[a] - 1 if a in prev_nav else 0.0
            if len(cn):
                na = cn[cn.index <= pd.Timestamp(a)]; nb = cn[cn.index <= pd.Timestamp(b)]
                n0 = na.iloc[-1] if len(na) else 100.0
                n1 = nb.iloc[-1] if len(nb) else n0
                contract = n1 / n0 - 1
            else:
                contract = 0.0
            sl = {"SLOW_h": slow - BETA["SLOW_h"] * r_iwm, "CONTRACT_h": contract - BETA["CONTRACT_h"] * r_iwm,
                  "ETFTREND": window_ret(etf, a, b), "CRYPTO": window_ret(cry, a, b)}
            stack = sum(W[k] * v for k, v in sl.items())
            days = (pd.Timestamp(b) - pd.Timestamp(a)).days
            for k in LEVERAGE:
                state["nav"][f"{k:g}x"] *= 1 + k * stack - max(k - 1, 0) * FIN_YR * days / 365
            for k, v in sl.items():
                state["sleeves"][k] *= 1 + v
            row = {"date": b, "from": a, "iwm_ret": round(r_iwm, 5), "sleeve_ret": {k: round(v, 5) for k, v in sl.items()},
                   "stack_ret": round(stack, 5), "nav": {k: round(v, 3) for k, v in state["nav"].items()},
                   "sleeve_nav": {k: round(v, 3) for k, v in state["sleeves"].items()}}
            lf.write(json.dumps(row) + "\n")
            state["last_date"] = b
            log.info(f"{a} -> {b}: stack {stack * 100:+.2f}% | " + " ".join(f"{k} {v * 100:+.2f}%" for k, v in sl.items()))
    STATE.write_text(json.dumps(state, indent=1))
    msg = "Shadow stack: NAV " + ", ".join(f"{k} {v:.2f}" for k, v in state["nav"].items())
    log.info(msg)
    log_event("shadow_stack", LogStatus.SUCCESS, msg, metrics=state["nav"])


if __name__ == "__main__":
    main()
