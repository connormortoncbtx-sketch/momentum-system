"""research/winner_sample.py -- quarterly top-1% winners and bottom-1% losers, 2024-2026,
with every 8-K they filed from 2 weeks before to the end of the 13-week run.
Writes research/samples/winner_sample.parquet for research/winner_postmortems.py."""
import glob, os
import numpy as np, pandas as pd
P = os.environ.get("PANEL", "research_data")
CACHE = os.environ.get("CACHE", "research/results/signatures_table.parquet")
df = pd.read_parquet(CACHE, columns=["week", "symbol", "next13", "vol12", "log_price", "log_adv"])
weeks = sorted(df.week.unique()); qs = set(weeks[::13])
q = df[df.week.isin(qs) & df.next13.notna() & (df.week >= "2024-01-01")].copy()
q["r_pct"] = q.groupby("week").next13.rank(pct=True)
s = pd.concat([q[q.r_pct > 0.99].assign(group="winner"), q[q.r_pct <= 0.01].assign(group="loser")])
fi = pd.concat(pd.read_parquet(x, columns=["cik", "tickers", "form", "items", "accepted_et", "accession"])
               for x in sorted(glob.glob(f"{P}/sec/filings_202[3-6].parquet")))
fi = fi[fi.form.isin(["8-K", "6-K"])]
fi = fi.assign(symbol=fi.tickers.str.split(",")).explode("symbol")
fi = fi[fi.symbol.isin(set(s.symbol))]
rows = []
for r in s.itertuples(index=False):
    a, b = r.week - pd.Timedelta(days=14), r.week + pd.Timedelta(weeks=13)
    f = fi[(fi.symbol == r.symbol) & (fi.accepted_et >= a) & (fi.accepted_et < b)].sort_values("accepted_et")
    rows.append({"week": r.week, "symbol": r.symbol, "group": r.group, "next13": r.next13,
                 "cik": int(f.cik.iloc[0]) if len(f) else None, "n_filings": len(f),
                 "accessions": "|".join(f.accession), "accepted": "|".join(f.accepted_et.astype(str)),
                 "items": "|".join(f["items"].fillna(""))})
out = pd.DataFrame(rows)
os.makedirs("research/samples", exist_ok=True)
out.to_parquet("research/samples/winner_sample.parquet", index=False)
print(out.groupby("group").agg(n=("symbol", "size"), with_filings=("n_filings", lambda x: (x > 0).mean()),
                               med_filings=("n_filings", "median"), med_ret=("next13", "median")))
