"""
research/pead_v2.py -- earnings drift with a REAL earnings surprise (Oct 17, 2026 audit redo).

The first PEAD test (pead.py) had no earnings surprise: its "sue" was the 2-day announcement
return scaled by volatility (EAR). Here:
  XSUE  standardized unexpected earnings from XBRL: quarterly diluted EPS (basic if missing) minus the
        same quarter a year earlier, divided by the std of the previous 8 such changes (>= 4).
        Q4 = fiscal-year EPS - Q1..Q3. Known at the 10-Q/10-K acceptance time (conservative: the
        8-K release is often earlier, so part of the drift is missed).
  EAR   the 2-day abnormal announcement return (the old score), latest within 30 days before.
  BOTH  rank(XSUE) + rank(EAR), only events where both are positive (the literature's strongest).
Events become tradable at the Friday close of the acceptance week; Monday-close entry (weekly
panel), 15 bps; grid on dev 2017-2023; the single dev pick is scored once on 2024-2026; IWM-hedged
version and its effect on the honest stack (research/stack_v2.py).
"""
import contextlib, glob, io, itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
DEV, HOLD = ("2017-01-06", "2023-12-29"), ("2024-01-05", "2026-09-18")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)

# --- XBRL EPS -> quarterly series per CIK ----------------------------------------------------
fu = pd.concat(pd.read_parquet(x, columns=["adsh", "tag", "qtrs", "value", "cik", "form", "period", "fp", "accepted_et"])
               for x in sorted(glob.glob(f"{P}/sec/fundamentals_20*.parquet")))
eps = fu[fu.tag.isin(["EarningsPerShareDiluted", "EarningsPerShareBasic"]) & fu.qtrs.isin([1, 4])].dropna(subset=["accepted_eT".lower()] if False else ["accepted_et"])
eps = eps.sort_values("tag", ascending=False)           # Diluted first
eps = eps.drop_duplicates(["cik", "period", "qtrs", "adsh"])
eps["period"] = pd.to_datetime(eps.period.astype(str), errors="coerce")
eps = eps.dropna(subset=["period"])
# keep the FIRST report of each (cik, period, qtrs): what was known when first filed
eps = eps.sort_values("accepted_et").drop_duplicates(["cik", "period", "qtrs"])
q = eps[eps.qtrs == 1][["cik", "period", "value", "accepted_et"]].rename(columns={"value": "eps"})
fy = eps[eps.qtrs == 4][["cik", "period", "value", "accepted_et"]].rename(columns={"value": "fy"})
# Q4 = FY - (Q1+Q2+Q3) using the three quarters ending within the 9 months before FY end
rows = []
qi = q.set_index("cik")
for r in fy.itertuples():
    if r.cik not in qi.index:
        continue
    g = qi.loc[[r.cik]]
    prior = g[(g.period < r.period) & (g.period > r.period - pd.Timedelta(days=290))]
    if len(prior) == 3:
        rows.append((r.cik, r.period, r.fy - prior.eps.sum(), r.accepted_et))
q4 = pd.DataFrame(rows, columns=["cik", "period", "eps", "accepted_et"])
Q = pd.concat([q, q4]).sort_values(["cik", "period"]).drop_duplicates(["cik", "period"])
Q["qend"] = Q.period.dt.to_period("Q")
Q = Q.drop_duplicates(["cik", "qend"])
Q = Q.set_index(["cik", "qend"])
lag = Q.groupby(level=0).eps.shift(4)                    # needs contiguous quarters: check below
qend_lag = Q.reset_index().groupby("cik").qend.shift(4).values
Q["d"] = Q.eps - lag
Q.loc[(Q.reset_index().qend - pd.Series(qend_lag)).apply(lambda x: getattr(x, "n", np.nan)).values != 4, "d"] = np.nan
Q["sd"] = Q.groupby(level=0).d.transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
Q["xsue"] = (Q.d / Q.sd).clip(-10, 10)
Q = Q.reset_index().dropna(subset=["xsue"])
print(f"XBRL surprises: {len(Q):,} company-quarters, {Q.cik.nunique():,} companies, {Q.accepted_et.min():%Y}..{Q.accepted_et.max():%Y}")

# cik -> symbol (current tickers in the filings set) and the EAR of the latest 8-K release
fi = pd.concat(pd.read_parquet(x, columns=["cik", "tickers"]) for x in sorted(glob.glob(f"{P}/sec/filings_20*.parquet")))
cmap = fi.dropna().drop_duplicates("cik").assign(symbol=lambda d: d.tickers.str.split(",").str[0]).set_index("cik").symbol
Q["symbol"] = Q.cik.map(cmap)
Q = Q.dropna(subset=["symbol"])
ea = pd.read_parquet(f"{P}/earnings_events.parquet")[["symbol", "accepted_et", "r2_abn", "sue"]].rename(
    columns={"accepted_et": "ea_t", "sue": "ear"})
Q = Q.sort_values("accepted_et"); ea = ea.sort_values("ea_t").dropna(subset=["ea_t"])
Q = pd.merge_asof(Q, ea, left_on="accepted_et", right_on="ea_t", by="symbol", direction="backward",
                  tolerance=pd.Timedelta(days=30))
sh = (Q.accepted_et + pd.Timedelta(hours=8)).dt.normalize()
Q["week"] = sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")
Q = Q[Q.symbol.isin(p.symbols) & Q.week.isin(p.weeks)]
print(f"usable events: {len(Q):,} ({Q.ear.notna().mean() * 100:.0f}% with a matched announcement return)")

widx = {w: i for i, w in enumerate(p.weeks)}
def score(col, age, cond=None):
    m = pd.DataFrame(np.nan, index=p.weeks, columns=p.symbols)
    src = Q if cond is None else Q[cond(Q)]
    for e in src.itertuples():
        i = widx[e.week]; v = getattr(e, col); j = m.columns.get_loc(e.symbol)
        for a in range(age):
            if i + a < len(p.weeks):
                m.iat[i + a, j] = v
    return m

# --- event study (dev): next 13-week excess vs SPY by XSUE quintile ----------------------------
d1c, cl = p.df("d1_close"), p.df("close")
rows = []
for e in Q.itertuples():
    i = widx[e.week]
    if i + 14 >= len(p.weeks) or not U.iat[i, U.columns.get_loc(e.symbol)]:
        continue
    j = d1c.columns.get_loc(e.symbol)
    a0, s0 = d1c.iat[i + 1, j], d1c["SPY"].iat[i + 1]
    x = cl.iat[i + 13, j]
    if np.isfinite(a0) and a0 > 0 and np.isfinite(x):
        rows.append((e.week, e.xsue, e.ear, (x / a0 - 1) - (cl["SPY"].iat[i + 13] / s0 - 1)))
es = pd.DataFrame(rows, columns=["week", "xsue", "ear", "abn13"])
es["abn13"] = es.abn13.clip(-1, 3)
for nm, (a, b) in (("dev", DEV), ("holdout", HOLD)):
    x = es[(es.week >= a) & (es.week <= b)].copy()
    x["q"] = pd.qcut(x.xsue, 5, labels=False, duplicates="drop") + 1
    t = x.groupby("q").abn13.agg(["mean", "count"]); spread = t["mean"].iloc[-1] - t["mean"].iloc[0]
    q5, q1 = x[x.q == 5].abn13, x[x.q == 1].abn13
    tt = spread / np.sqrt(q5.var() / len(q5) + q1.var() / len(q1))
    print(f"EVENT {nm}: 13-wk excess by XSUE quintile (%): " + " ".join(f"{v * 100:+.2f}" for v in t["mean"]) +
          f" | Q5-Q1 {spread * 100:+.2f}% (t {tt:.1f})")
    both = x[(x.xsue > 0) & (x.ear > 0)].abn13; nb = x[~((x.xsue > 0) & (x.ear > 0))].abn13
    print(f"      both XSUE>0 & EAR>0: {both.mean() * 100:+.2f}% (n {len(both)}) vs rest {nb.mean() * 100:+.2f}%")

# --- portfolio grid on dev -------------------------------------------------------------------
rk = lambda x: B.xrank(x, U)
SC = {}
def sc(name, age):
    if (name, age) in SC:
        return SC[(name, age)]
    if name == "xsue":
        s = score("xsue", age)
    elif name == "ear":
        s = score("ear", age)
    else:
        s1 = score("xsue", age, lambda d: (d.xsue > 0) & (d.ear > 0)); s2 = score("ear", age, lambda d: (d.xsue > 0) & (d.ear > 0))
        s = (rk(s1) + rk(s2)).where(s1.notna())
    SC[(name, age)] = s
    return s
grid = list(itertools.product(["xsue", "ear", "both"], (8, 13), (1, 4), (20, 30), ("equal", "invvol")))
res, ser = [], {}
for g in grid:
    s = sc(g[0], g[1])
    mask = U & s.notna() & (s > 0)
    with contextlib.redirect_stdout(io.StringIO()):
        r = B.simulate(p, s.loc[DEV[0]:HOLD[1]], mask.loc[DEV[0]:HOLD[1]], n=g[3], buffer=2 * g[3], k=g[2],
                       cost_bps=15, weighting=g[4], vol=f["vol12"])
    ser[g] = r
    d, h = B.stats(r.loc[DEV[0]:DEV[1]]), B.stats(r.loc[HOLD[0]:HOLD[1]])
    res.append(dict(score=g[0], age=g[1], k=g[2], n=g[3], wt=g[4], dev_Sh=d.get("Sharpe"), dev_CAGR=d.get("CAGR%"),
                    hold_Sh=h.get("Sharpe"), hold_CAGR=h.get("CAGR%")))
res = pd.DataFrame(res).sort_values("dev_Sh", ascending=False)
pd.set_option("display.width", 200)
print("\nGRID (top 12 by dev Sharpe; holdout shown for reference, pick = first row)")
print(res.head(12).round(2).to_string(index=False))
print("median by score:", res.groupby("score")[["dev_Sh", "hold_Sh"]].median().round(2).to_dict())
best = tuple(res.iloc[0][["score", "age", "k", "n", "wt"]])
best = (best[0], int(best[1]), int(best[2]), int(best[3]), best[4])
rb = ser[best]; rb.index = rb.index + pd.Timedelta(days=7)          # formation week -> realised week
rb.to_frame("PEAD2").to_csv(f"{P}/results/pead_v2_weekly.csv")
print(f"\nDEV PICK {best}: dev {B.stats(rb[rb.index < '2024-01-01'])}  holdout {B.stats(rb[rb.index >= '2024-01-01'])}")
