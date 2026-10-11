"""
research/reversal_lc.py -- large-cap short-term reversal (Oct 2026)

Buy last week's biggest losers among the most-traded names, hold one week
(Mon close -> Fri close, designed timing). Variants: universe size, basket size,
excluding losers whose drop came with an earnings report (information, not
liquidity), cost per side. Dev 2017-23 grid, holdout 2024-26 once.
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
fwd = B.next_week_returns(p, "designed")
start, split = pd.Timestamp("2017-01-06"), pd.Timestamp("2024-01-01")

e = pd.read_parquet(f"{P}/earnings_events.parquet")
e["d0"] = pd.to_datetime(e.d0)
e["wk"] = e.d0 + pd.to_timedelta((4 - e.d0.dt.weekday) % 7, unit="D")   # week containing the reaction
arr = np.zeros((len(p.weeks), len(p.symbols)), dtype=bool)
ii = p.weeks.get_indexer(e.wk); jj = p.symbols.get_indexer(e.symbol); ok = (ii >= 0) & (jj >= 0)
arr[ii[ok], jj[ok]] = True
ev_form = pd.DataFrame(arr, index=p.weeks, columns=p.symbols)   # reported during formation week

rows = []
for top_univ, n, excl_earn, cost in itertools.product((100, 250, 500, 1000), (10, 20, 50), (False, True), (5.0, 15.0)):
    m = U & fwd.notna() & (f["adv"].where(U).rank(axis=1, ascending=False) <= top_univ)
    if excl_earn:
        m &= ~ev_form
    s = f["r1w"].where(m)
    out = {}
    for wk, row in s.iterrows():
        row = row.dropna()
        if len(row) < n * 2:
            continue
        out[wk] = fwd.loc[wk, row.nsmallest(n).index].mean() - 2 * cost / 1e4
    r = pd.Series(out); r = r[r.index >= start]
    d, h = B.stats(r[r.index < split]), B.stats(r[r.index >= split])
    rows.append(dict(univ=top_univ, n=n, excl_earn=excl_earn, cost=cost,
                     dev_CAGR=d["CAGR%"], dev_Sharpe=d["Sharpe"], dev_DD=d["maxDD%"],
                     hold_CAGR=h["CAGR%"], hold_Sharpe=h["Sharpe"], hold_DD=h["maxDD%"]))
res = pd.DataFrame(rows).round(2)
spy = fwd["SPY"]; spy = spy[spy.index >= start]
print("SPY same timing: dev", {k: round(v, 2) for k, v in B.stats(spy[spy.index < split]).items() if k in ("CAGR%", "Sharpe", "maxDD%")},
      "holdout", {k: round(v, 2) for k, v in B.stats(spy[spy.index >= split]).items() if k in ("CAGR%", "Sharpe", "maxDD%")})
pd.set_option("display.width", 200)
print(res.sort_values("dev_Sharpe", ascending=False).to_string(index=False))
res.to_csv(os.environ.get("OUT", "research/results") + "/reversal_lc.csv", index=False)
