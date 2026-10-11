"""
research/pdufa_test.py -- do biotech stocks run up into a known FDA decision (PDUFA) date?

Calendar: research/fetch_pdufa.py (dates mined from SEC full-text search; each date is
used only from the first filing that mentioned it). Weekly data; excess vs the
equal-weight universe (price >= $2, >= $500k/day, 26+ weeks listed).
Event week 0 = the week containing the PDUFA date (decisions can also come early).
A. Average excess by event week k = -8 .. +4.
B. Strategy "ride the run-up, skip the coin flip": buy at the Monday close 4 weeks
   before the PDUFA week (only if the date was public by then), sell at the Friday close
   of the week before it; 15 bps/side. Dev = decisions before 2023, holdout = 2023 on.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B
P = os.environ.get("PANEL", "research_data")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
C, d1c = p.df("close"), p.df("d1_close")
U = (f["price"] >= 2) & (f["adv"] >= 5e5) & (f["hist"] >= 26) & ~pd.Series(p.is_fund, index=p.symbols)
r = (C / C.shift(1) - 1).where(lambda x: x.abs() < 3)
ex = r.sub(r.where(U).mean(axis=1), axis=0)
pm = pd.read_parquet(f"{P}/biotech/pdufa_mentions.parquet")
fi = pd.concat(pd.read_parquet(x, columns=["cik", "tickers"]) for x in glob.glob(f"{P}/sec/filings_*.parquet")).drop_duplicates("cik")
tick = dict(zip(fi.cik, fi.tickers.str.split(",").str[0]))
pm["symbol"] = pm.cik.map(tick)
ev = pm.dropna(subset=["symbol"]).groupby(["symbol", "pdufa_date"]).filed.min().reset_index(name="first_public")
ev = ev[ev.symbol.isin(set(p.symbols))]
fri = lambda d: d + pd.to_timedelta((4 - d.dt.weekday) % 7, unit="D")
ev["wk0"] = fri(ev.pdufa_date)
ev["pub_wk"] = fri(ev.first_public)
ev = ev[ev.wk0.isin(set(p.weeks))].drop_duplicates(["symbol", "wk0"])
print(f"PDUFA events with a ticker in the price panel: {len(ev):,} ({ev.symbol.nunique()} companies), "
      f"{ev.wk0.min().date()} .. {ev.wk0.max().date()}; median lead time "
      f"{(ev.pdufa_date - ev.first_public).dt.days.median():.0f} days")
W = {w: i for i, w in enumerate(p.weeks)}
rows = []
for e in ev.itertuples(index=False):
    i0, j = W[e.wk0], p.symbols.get_loc(e.symbol)
    for k in range(-8, 5):
        i = i0 + k
        if 0 < i < len(p.weeks) and U.iat[i - 1, j] and p.weeks[i] > e.pub_wk:
            rows.append((e.wk0, k, ex.iat[i, j]))
d = pd.DataFrame(rows, columns=["wk0", "k", "x"]).dropna()
d["per"] = np.where(d.wk0 < "2023-01-01", "dev", "holdout")
t = d.groupby(["per", "k"]).x.agg(["mean", "median", "count"])
t[["mean", "median"]] *= 100
pd.set_option("display.width", 200)
print("\nA. excess return (%) by week relative to the PDUFA week (only weeks after the date was public)")
print(t.unstack(0).round(2).to_string())
# B. strategy
res = []
for e in ev.itertuples(index=False):
    i0, j = W[e.wk0], p.symbols.get_loc(e.symbol)
    ib = i0 - 4
    if ib < 1 or p.weeks[ib - 1] < e.pub_wk or not U.iat[ib - 1, j]:
        continue
    entry = d1c.iat[ib, j]; exit_ = C.iat[i0 - 1, j]
    uni = (C.iloc[i0 - 1] / d1c.iloc[ib] - 1).where(U.iloc[ib - 1]).clip(-0.9, 3).mean()
    if np.isfinite(entry) and np.isfinite(exit_) and entry > 0:
        res.append((e.wk0, exit_ / entry - 1 - uni - 0.003, exit_ / entry - 1))
s = pd.DataFrame(res, columns=["wk0", "ex", "raw"])
for per, x in (("dev", s[s.wk0 < "2023-01-01"]), ("holdout", s[s.wk0 >= "2023-01-01"])):
    print(f"B. {per}: {len(x)} trades, mean excess {x.ex.mean() * 100:+.2f}% (t {x.ex.mean() / (x.ex.std() / np.sqrt(len(x))):.2f}), "
          f"median {x.ex.median() * 100:+.2f}%, win {(x.ex > 0).mean() * 100:.0f}%, mean raw {x.raw.mean() * 100:+.2f}%")
