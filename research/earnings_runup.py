"""
research/earnings_runup.py -- do strong-momentum stocks run up INTO and THROUGH a scheduled
earnings report? (Report dates are announced weeks ahead; using actual dates is a mild
look-ahead on timing only.)
Event time k = week relative to the report week (0 = week containing the first reaction day).
Excess vs equal-weight universe ($5, $2M/day), Mon close -> Mon close weeks.
Groups by 12-1 momentum percentile at the formation week before k = -2.
Strategy: each week buy every top-20% 12-1 name whose report falls in the NEXT week
(k = +1 from formation), hold 2 weeks (through the report), 15 bps/side.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B
P = os.environ.get("PANEL", "research_data")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
d1c = p.df("d1_close")
wk = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)   # formation week t -> return over week t+1
ex = wk.sub(wk.where(U).mean(axis=1), axis=0)
e = pd.read_parquet(f"{P}/earnings_events.parquet", columns=["symbol", "d0"])
e["d0"] = pd.to_datetime(e.d0)
e["wk"] = e.d0 + pd.to_timedelta((4 - e.d0.dt.weekday) % 7, unit="D")
e = e[e.symbol.isin(set(p.symbols)) & e.wk.isin(set(p.weeks))].drop_duplicates(["symbol", "wk"])
mom = f["m12_1"].where(U).rank(axis=1, pct=True)
W = {w: i for i, w in enumerate(p.weeks)}
rows = []
for r in e.itertuples(index=False):
    i0 = W[r.wk]; j = p.symbols.get_loc(r.symbol)
    if i0 < 4 or i0 + 3 >= len(p.weeks):
        continue
    q = mom.iat[i0 - 3, j]
    if not np.isfinite(q):
        continue
    for k in range(-3, 4):                 # return over event week k = formation week i0+k-1
        x = ex.iat[i0 + k - 1, j]
        if np.isfinite(x):
            rows.append((r.wk, q, k, x))
d = pd.DataFrame(rows, columns=["wk", "momq", "k", "x"])
d["grp"] = pd.cut(d.momq, [0, 0.2, 0.8, 0.9, 1.0], labels=["bottom 20%", "middle", "80-90%", "top 10%"])
d["per"] = np.where(d.wk < "2024-01-01", "dev", "holdout")
t = d.pivot_table(index=["per", "grp"], columns="k", values="x", aggfunc="mean", observed=True) * 100
pd.set_option("display.width", 200)
print("Excess return (%) by week relative to the report week (0), by 12-1 momentum group")
print(t.round(2).to_string())
# strategy
rep_next = pd.DataFrame(False, index=p.weeks, columns=p.symbols)
a = np.zeros(rep_next.shape, bool)
ii = np.array([W[w] - 1 for w in e.wk if W[w] >= 1]); jj = np.array([p.symbols.get_loc(s) for w, s in zip(e.wk, e.symbol) if W[w] >= 1])
a[ii, jj] = True
rep_next = pd.DataFrame(a, index=p.weeks, columns=p.symbols)      # report falls in week t+1
wk2 = (d1c.shift(-3) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 2)   # hold weeks t+1 and t+2
ew2 = wk2.where(U).mean(axis=1)
out = {}
for t_, row in (U & rep_next & (mom > 0.8)).iterrows():
    names = row[row].index
    if len(names) >= 5:
        out[t_] = wk2.loc[t_, names].mean() - ew2.loc[t_] - 2 * 15 / 1e4
s = pd.Series(out)
s = s[s.index >= "2017-01-06"]
for per, sl in (("dev", s.index < "2024-01-01"), ("holdout", s.index >= "2024-01-01")):
    x = s[sl]
    print(f"strategy {per}: {len(x)} weeks, mean 2-week excess {x.mean() * 100:+.2f}% "
          f"(t {x.mean() / (x.std() / np.sqrt(len(x) / 2)):.2f}), hit {(x > 0).mean() * 100:.0f}%, "
          f"names/week {(U & rep_next & (mom > 0.8)).sum(axis=1)[x.index].median():.0f}")
