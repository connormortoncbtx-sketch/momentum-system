"""
research/forced_flows.py -- trading against sellers who have no choice (Oct 2026)

  A. Tax-loss selling / January rebound: December's YTD losers (small, liquid enough),
     bought in late December, held into January.
  B. IPO lock-up expiry: IPOs (first trading week + a 424B4 prospectus within 3 weeks),
     returns around day 180 when insiders may sell.
  C. Follow-on offerings: 424B4/424B5 prospectuses from companies listed > 6 months;
     returns in the weeks after the offering.
Excess returns vs the equal-weight universe ($5, $2M/day, no funds), weekly data.
Dev = events before 2023, holdout = 2023 on (few-observation designs need the
longer holdout). Weekly granularity only: day-level timing is not modelled.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
SEC = os.environ.get("SEC", f"{P}/sec")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
U1 = B.universe(p, f, 2.0, 5e5, exclude_rsplit=False)    # looser universe for small names
# IPOs have < 53 weeks of history at lock-up, so they need a mask without the history rule
U_ipo = (f["price"] >= 2.0) & (f["adv"] >= 5e5) & ~pd.Series(p.is_fund, index=p.symbols)
C, d1c = p.df("close"), p.df("d1_close")
wk = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)   # Mon close -> next Mon close, aligned to formation week
ew = wk.where(U).mean(axis=1)
ex = wk.sub(ew, axis=0)
SPLIT = pd.Timestamp("2023-01-01")


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


# ── A. tax-loss / January ────────────────────────────────────────────────────
print("A. December YTD losers, bought at the Monday close of the week containing Dec 20,")
print("   held N weeks; excess vs equal-weight universe, per year")
rows = []
for yr in range(2016, 2026):
    yend = p.weeks[p.weeks <= pd.Timestamp(yr - 1, 12, 31)]
    form = p.weeks[(p.weeks >= pd.Timestamp(yr, 12, 12)) & (p.weeks <= pd.Timestamp(yr, 12, 18))]
    if len(yend) == 0 or len(form) == 0:
        continue
    t0, tf = yend[-1], form[0]
    ytd = C.loc[tf] / C.loc[t0] - 1
    m = U1.loc[tf] & ytd.notna()
    losers = ytd[m].rank(pct=True) <= 0.10
    small = f["adv"].loc[tf][m].rank(pct=True) <= 0.5
    i = p.weeks.get_loc(tf)
    for hold in (2, 4, 6):
        if i + hold >= len(p.weeks):
            continue
        win = ex.iloc[i:i + hold]                     # weekly excess, formation weeks i..i+hold-1
        cum = (1 + win).prod() - 1
        for name, sel in (("all losers", losers), ("small losers", losers & small)):
            names = sel[sel].index
            rows.append({"year": yr, "hold_wk": hold, "group": name, "n": len(names),
                         "excess_%": cum[names].mean() * 100})
a = pd.DataFrame(rows)
print(a.pivot_table(index="year", columns=["group", "hold_wk"], values="excess_%").round(1).to_string())
for (g, h), x in a.groupby(["group", "hold_wk"]):
    d, o = x[x.year < 2023]["excess_%"], x[x.year >= 2023]["excess_%"]
    print(f"   {g:13s} {h}w: dev mean {d.mean():+.1f}% (t {tstat(d):+.1f}, {len(d)} yrs) | holdout {o.mean():+.1f}% ({len(o)} yrs)")

# ── B. IPO lock-up expiry ────────────────────────────────────────────────────
fi = pd.concat(pd.read_parquet(x, columns=["cik", "tickers", "form", "filing_date"]) for x in sorted(glob.glob(f"{SEC}/filings_*.parquet")))
fi["symbol"] = fi.tickers.str.split(",").str[0]
first = C.notna().idxmax()
first = first[first > pd.Timestamp("2016-03-01")]
p4 = fi[fi.form == "424B4"].groupby("symbol").filing_date.min()
ipo = pd.DataFrame({"first_wk": first}).join(p4.rename("p424b4"), how="inner")
ipo = ipo[(ipo.p424b4 - ipo.first_wk).abs() <= pd.Timedelta(days=21)]
print(f"\nB. IPOs identified: {len(ipo):,}")
rel = range(-3, 7)
res = {k: [] for k in rel}
dates = []
for sym, r in ipo.iterrows():
    lock = r.first_wk + pd.Timedelta(days=180)
    lw = p.weeks[p.weeks >= lock]
    if len(lw) == 0 or sym not in ex.columns:
        continue
    i = p.weeks.get_loc(lw[0])
    for k in rel:
        j = i + k - 1                                 # formation week before event week i+k
        if 0 <= j < len(p.weeks) and U_ipo.iat[j, ex.columns.get_loc(sym)] and np.isfinite(ex.iat[j, ex.columns.get_loc(sym)]):
            res[k].append((lw[0], ex.iat[j, ex.columns.get_loc(sym)]))
tbl = []
for k in rel:
    d = pd.DataFrame(res[k], columns=["wk", "x"])
    dv, ho = d[d.wk < SPLIT].x, d[d.wk >= SPLIT].x
    tbl.append({"week_vs_lockup": k, "n_dev": len(dv), "dev_%": dv.mean() * 100, "dev_t": tstat(dv),
                "n_hold": len(ho), "hold_%": ho.mean() * 100, "hold_t": tstat(ho)})
print(pd.DataFrame(tbl).round(2).to_string(index=False))

# ── C. follow-on offerings ───────────────────────────────────────────────────
off = fi[fi.form.isin(["424B4", "424B5"])].copy()
off = off.merge(ipo[["first_wk"]], left_on="symbol", right_index=True, how="left")
firstall = C.notna().idxmax()
off["listed"] = off.symbol.map(firstall)
off = off[off.listed.notna() & (off.filing_date - off.listed > pd.Timedelta(days=180))]
off = off.sort_values("filing_date").drop_duplicates(["symbol", "filing_date"])
off["gap"] = off.groupby("symbol").filing_date.diff().dt.days
off = off[(off.gap.isna()) | (off.gap > 30)]          # first prospectus of a cluster (ATM programs file many)
print(f"\nC. follow-on offering events: {len(off):,}")
rel = range(-1, 6)
res = {k: [] for k in rel}
for r in off.itertuples():
    wks = p.weeks[p.weeks >= r.filing_date]
    if len(wks) == 0 or r.symbol not in ex.columns:
        continue
    i = p.weeks.get_loc(wks[0]); c = ex.columns.get_loc(r.symbol)
    for k in rel:
        j = i + k - 1
        if 0 <= j < len(p.weeks) and U1.iat[j, c]:
            res[k].append((wks[0], r.form, ex.iat[j, c]))
tbl = []
for k in rel:
    d = pd.DataFrame(res[k], columns=["wk", "form", "x"])
    for form in ("424B4", "424B5"):
        x = d[d.form == form]
        dv, ho = x[x.wk < SPLIT].x, x[x.wk >= SPLIT].x
        tbl.append({"form": form, "week_vs_offering": k, "n_dev": len(dv), "dev_%": dv.mean() * 100,
                    "dev_t": tstat(dv), "n_hold": len(ho), "hold_%": ho.mean() * 100, "hold_t": tstat(ho)})
print(pd.DataFrame(tbl).round(2).to_string(index=False))
