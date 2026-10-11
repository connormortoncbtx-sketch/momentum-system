"""
research/item_events.py -- 8-K item-type event study on weekly data (Oct 2026)

For every 8-K (2017+) by issuer, the filing week (Fri 16:00 ET cut-off; stored
times may be up to 5h late -> conservative) and the next 4 weeks. Excess return
vs the equal-weight universe (price >= $2, $500k/day, 26+ weeks listed).
  week 0   : the week containing the filing (partly BEFORE the filing -- not tradeable)
  weeks 1-4: after the filing week closes -- tradeable drift
Dev 2017-2023, holdout 2024-2026. Small = bottom half of the universe by $ volume.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
SEC = os.environ.get("SEC", f"{P}/sec")
OUT = os.environ.get("OUT", "research/results")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
C = p.df("close")
U = (f["price"] >= 2) & (f["adv"] >= 5e5) & (f["hist"] >= 26) & ~pd.Series(p.is_fund, index=p.symbols)
r = (C / C.shift(1) - 1).where(lambda x: x.abs() < 3)
ex = r.sub(r.where(U).mean(axis=1), axis=0)
small = f["adv"].where(U).rank(axis=1, pct=True) <= 0.5

fi = pd.concat(pd.read_parquet(x, columns=["tickers", "form", "items", "accepted_et"])
               for x in sorted(glob.glob(f"{SEC}/filings_*.parquet")))
k = fi[(fi.form == "8-K") & (fi.accepted_et >= "2017-01-01")].reset_index(drop=True)
k["symbol"] = k.tickers.str.split(",").str[0]
sh = (k.accepted_et + pd.Timedelta(hours=8)).dt.normalize()
k["week"] = sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")
k["item"] = k["items"].fillna("").str.split(",")
k = k.explode("item"); k["item"] = k.item.str.strip()
k = k[k.item.ne("") & k.item.ne("9.01")].drop_duplicates(["symbol", "week", "item"])
i = p.weeks.get_indexer(k.week); j = p.symbols.get_indexer(k.symbol)
ok = (i >= 1) & (j >= 0)
k, i, j = k[ok], i[ok], j[ok]
E, Um, Sm = ex.to_numpy(), U.to_numpy(), small.to_numpy()
inu = Um[i - 1, j]
k = k[inu]; i, j = i[inu], j[inu]
for h in range(0, 5):
    v = np.full(len(k), np.nan); okh = i + h < len(p.weeks)
    v[okh] = E[i[okh] + h, j[okh]]
    k[f"w{h}"] = v
k["post1_4"] = k[["w1", "w2", "w3", "w4"]].sum(axis=1, min_count=4)
k["small"] = Sm[i - 1, j]
k["period"] = np.where(k.week < "2024-01-01", "dev", "holdout")

NAMES = {"1.01": "material agreement", "1.02": "agreement terminated", "2.01": "acquisition/disposal done",
         "2.02": "earnings", "2.03": "new debt obligation", "2.04": "debt acceleration", "2.05": "restructuring/exit costs",
         "2.06": "impairment", "3.01": "listing deficiency", "3.02": "unregistered share sale", "3.03": "holder rights modified",
         "4.01": "auditor change", "4.02": "non-reliance (restatement)", "5.01": "change of control",
         "5.02": "officer/director change", "5.03": "bylaws/charter change", "5.07": "shareholder vote",
         "5.08": "nomination date", "7.01": "Reg FD release", "8.01": "other events"}


def tstat(x):
    x = x.dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 30 else np.nan


rows = []
for (item, per), g in k.groupby(["item", "period"]):
    if item not in NAMES or len(g) < 100:
        continue
    for size, gg in (("all", g), ("small", g[g.small == True])):
        rows.append({"item": f"{item} {NAMES[item]}", "period": per, "size": size, "n": len(gg),
                     "wk0_%": gg.w0.mean() * 100, "wk1_%": gg.w1.mean() * 100,
                     "wk1-4_%": gg.post1_4.mean() * 100, "t1-4": tstat(gg.post1_4)})
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/item_events.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
piv = res.pivot_table(index=["item", "size"], columns="period", values=["n", "wk0_%", "wk1-4_%", "t1-4"])
piv = piv[[("n", "dev"), ("n", "holdout"), ("wk0_%", "dev"), ("wk0_%", "holdout"),
           ("wk1-4_%", "dev"), ("t1-4", "dev"), ("wk1-4_%", "holdout"), ("t1-4", "holdout")]]
print(piv.round(2).to_string())
