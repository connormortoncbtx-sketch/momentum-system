"""
research/avoid_filter.py -- does the filing-based avoid list improve the slow-momentum
design (12-1 momentum, top N, hold while ranked <= 100)? Entry filter only.
Avoid = in the last 8 weeks: 8-K item 3.02 (unregistered share sale), 3.01/3.03/5.01
(listing deficiency / holder rights / change of control), or a 424B1/3/4/5 prospectus
(an offering). Spike filter = top-10 gainer over 1 or 4 weeks (as shadow #4).
"""
import glob, os
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hold_conditions.py")).read().split("rows = []")[0]
exec(SRC)

fi = pd.concat(pd.read_parquet(x, columns=["tickers", "form", "items", "accepted_et"])
               for x in sorted(glob.glob(f"{P}/sec/filings_*.parquet")))
fi["symbol"] = fi.tickers.str.split(",").str[0]
sh = (fi.accepted_et + pd.Timedelta(hours=8)).dt.normalize()
fi["week"] = sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")
bad_item = (fi.form == "8-K") & fi["items"].fillna("").str.contains(r"3\.02|3\.01|3\.03|5\.01")
offer = fi.form.isin(["424B1", "424B3", "424B4", "424B5"])
ev = fi[bad_item | offer]
ev = ev[ev.symbol.isin(set(p.symbols)) & ev.week.isin(set(weeks))]
g = ev.groupby(["week", "symbol"]).size().unstack().reindex(index=weeks, columns=p.symbols).fillna(0)
recent_bad = g.rolling(8, min_periods=1).sum() > 0
r1r = f["r1w"].where(U).rank(axis=1, ascending=False); r4r = f["r4w"].where(U).rank(axis=1, ascending=False)
spiker = (r1r <= 10) | (r4r <= 10)
ok_avoid = (~recent_bad).astype(float)
ok_spike = (~spiker).astype(float)
ok_both = ((~recent_bad) & (~spiker)).astype(float)
print("share of top-50 12-1 names flagged by the avoid list:",
      round(float(recent_bad.where(f["m12_1"].where(U).rank(axis=1, ascending=False) <= 50).stack().mean()) * 100, 1), "%")
rows = []
for n in (10, 20):
    for name, ent in (("no filter", None), ("spike filter", ok_spike), ("avoid list", ok_avoid), ("both", ok_both)):
        r = sim(f["m12_1"], n=n, buffer=100, entry=ent)
        d, h = st(r[r.index < split]), st(r[r.index >= split])
        rows.append({"n": n, "filter": name, **{f"dev_{k}": v for k, v in d.items()}, **{f"hold_{k}": v for k, v in h.items()}})
res = pd.DataFrame(rows).round(2)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
res.to_csv("research/results/avoid_filter.csv", index=False)
