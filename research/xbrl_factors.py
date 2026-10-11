"""
research/xbrl_factors.py -- point-in-time fundamental factors from SEC XBRL (Oct 2026)

Quarterly values (qtrs=1; Q4 derived as annual minus Q1-Q3) -> trailing-12-month
flows; balance-sheet items as reported. Each value is usable only after the
filing's acceptance time (FSDS 'accepted', Eastern). CIK -> ticker uses today's
SEC ticker list (via the filings index), so companies delisted before 2026 drop
out: a survivorship caveat for this test.

Factors (as of each Friday):
  gross_prof   TTM gross profit / assets            (Novy-Marx: high = good)
  roa          TTM net income / assets
  accruals     (TTM net income - TTM operating cash flow) / assets   (low = good)
  asset_growth assets / assets 4 quarters earlier - 1               (low = good)
  issuance     shares / shares 4 quarters earlier - 1               (low = good)
  rev_growth   TTM revenue / TTM revenue 4 quarters earlier - 1
  rev_accel    latest quarter's YoY growth - previous quarter's YoY growth
  earn_yield   TTM net income / market cap
  book_mkt     equity / market cap
Test: within the universe ($5, $2M/day, no funds), top-minus-bottom quintile
excess return over the next 4 and 13 weeks (Mon close entry). Dev 2017-2023,
holdout 2024-2026. Newey-West t-stats.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
SEC = os.environ.get("SEC", f"{P}/sec")
OUT = os.environ.get("OUT", "research/results")
TAGS = {"rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
        "gp": ["GrossProfit"], "ni": ["NetIncomeLoss"], "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
        "assets": ["Assets"], "equity": ["StockholdersEquity"],
        "shares": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"]}
FLOWS = {"rev", "gp", "ni", "cfo"}

fu = pd.concat(pd.read_parquet(x, columns=["cik", "tag", "qtrs", "value", "period", "accepted_et", "form"])
               for x in sorted(glob.glob(f"{SEC}/fundamentals_*.parquet")))
fu = fu.dropna(subset=["accepted_et", "period", "value"])
fu["tag"] = fu.tag.astype(str)
fu = fu[~fu.form.astype(str).str.endswith("/A")]               # originals only (amendments come later)


def quarterly_flow(tags):
    d = fu[fu.tag.isin(tags)].copy()
    d["prio"] = d.tag.map({t: i for i, t in enumerate(tags)})
    d = d.sort_values(["cik", "period", "qtrs", "prio", "accepted_et"]).drop_duplicates(["cik", "period", "qtrs"])
    q1 = d[d.qtrs == 1][["cik", "period", "value", "accepted_et"]]
    yr = d[d.qtrs == 4][["cik", "period", "value", "accepted_et"]]
    # derive Q4 = annual - the three quarters ending in the prior ~9.5 months
    q1s = q1.sort_values("period")
    rows = []
    for cik, g in yr.groupby("cik"):
        qq = q1s[q1s.cik == cik] if False else None
    m = yr.merge(q1, on="cik", suffixes=("", "_q"))
    m = m[(m.period_q < m.period) & (m.period_q > m.period - pd.Timedelta(days=290))]
    s = m.groupby(["cik", "period"]).agg(qsum=("value_q", "sum"), nq=("value_q", "size")).reset_index()
    q4 = yr.merge(s, on=["cik", "period"])
    q4 = q4[q4.nq == 3]
    q4["value"] = q4.value - q4.qsum
    q = pd.concat([q1, q4[["cik", "period", "value", "accepted_et"]]]).sort_values(["cik", "period"])
    q = q.drop_duplicates(["cik", "period"])
    # TTM over 4 consecutive quarters (span <= 380 days)
    g = q.groupby("cik")
    q["ttm"] = g.value.rolling(4).sum().reset_index(level=0, drop=True)
    span = q.period - g.period.shift(3)
    q.loc[span > pd.Timedelta(days=380), "ttm"] = np.nan
    q["yoy_q"] = q.value / g.value.shift(4) - 1
    q["ttm_lag4"] = g.ttm.shift(4)
    return q


def stock(tags):
    d = fu[fu.tag.isin(tags) & (fu.qtrs.fillna(0) == 0)].copy()
    d["prio"] = d.tag.map({t: i for i, t in enumerate(tags)})
    d = d.sort_values(["cik", "period", "prio", "accepted_et"]).drop_duplicates(["cik", "period"])
    d = d[["cik", "period", "value", "accepted_et"]].sort_values(["cik", "period"])
    d["lag4"] = d.groupby("cik").value.shift(4)
    return d


p = B.load_panel(P, "2026-10-02"); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
C, d1c = p.df("close"), p.df("d1_close")
fwd = {h: (C.shift(-h - 1) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 5) for h in (4, 13)}

fi = pd.concat(pd.read_parquet(x, columns=["cik", "tickers"]) for x in sorted(glob.glob(f"{SEC}/filings_*.parquet")))
tick = fi.drop_duplicates("cik").assign(symbol=lambda d: d.tickers.str.split(",")).explode("symbol")[["cik", "symbol"]]
tick = tick[tick.symbol.isin(set(p.symbols))]
print(f"CIK->ticker matches in the price panel: {tick.symbol.nunique():,} symbols")


def to_panel(df, col):
    """value usable from the first Friday close after acceptance; forward-filled up to 400 days."""
    d = df.dropna(subset=[col]).merge(tick, on="cik")
    sh = (d.accepted_et + pd.Timedelta(hours=8)).dt.normalize()
    d["week"] = sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")
    d = d.sort_values("accepted_et").drop_duplicates(["symbol", "week"], keep="last")
    w = d.pivot(index="week", columns="symbol", values=col).reindex(columns=p.symbols)
    w = w.reindex(w.index.union(p.weeks)).sort_index().ffill(limit=58).reindex(p.weeks)
    return w


fl = {k: quarterly_flow(TAGS[k]) for k in FLOWS}
st = {k: stock(TAGS[k]) for k in ("assets", "equity", "shares")}
A = to_panel(st["assets"], "value")
A_lag = to_panel(st["assets"], "lag4")
SH = to_panel(st["shares"], "value")
SH_lag = to_panel(st["shares"], "lag4")
EQ = to_panel(st["equity"], "value")
GP = to_panel(fl["gp"], "ttm")
NI = to_panel(fl["ni"], "ttm")
CFO = to_panel(fl["cfo"], "ttm")
REV = to_panel(fl["rev"], "ttm")
REV_lag = to_panel(fl["rev"], "ttm_lag4")
rq = fl["rev"].copy()
rq["accel"] = rq.yoy_q - rq.groupby("cik").yoy_q.shift(1)
ACC = to_panel(rq, "accel")
mcap = SH * f["price"]
A_ = A.where(A > 0)
FACT = {
    "gross_prof (high good)": GP / A_,
    "roa (high good)": NI / A_,
    "accruals (low good)": -((NI - CFO) / A_),
    "asset_growth (low good)": -(A / A_lag - 1),
    "issuance (low good)": -(SH / SH_lag - 1),
    "rev_growth": REV / REV_lag.where(REV_lag > 0) - 1,
    "rev_accel": ACC,
    "earn_yield": NI / mcap.where(mcap > 0),
    "book_mkt": EQ / mcap.where(mcap > 0),
}
start, split = pd.Timestamp("2017-01-06"), pd.Timestamp("2024-01-01")


def nw_t(x, lag):
    x = x.dropna().to_numpy(); n = len(x); e = x - x.mean(); v = e @ e / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(v / n)


rows = []
for name, F in FACT.items():
    F = F.replace([np.inf, -np.inf], np.nan)
    for h in (4, 13):
        m = U & F.notna() & fwd[h].notna()
        q = F.where(m).rank(axis=1, pct=True)
        exr = fwd[h].where(m).sub(fwd[h].where(m).mean(axis=1), axis=0)
        top = exr.where(q > 0.8).mean(axis=1); bot = exr.where(q <= 0.2).mean(axis=1)
        sp = (top - bot)[lambda s: s.index >= start]
        cov = m.sum(axis=1).div(U.sum(axis=1))[lambda s: s.index >= start].mean()
        d, o = sp[sp.index < split], sp[sp.index >= split]
        rows.append({"factor": name, "h_wk": h, "coverage": cov, "dev_top-bot_%": d.mean() * 100,
                     "dev_t": nw_t(d, h - 1), "hold_top-bot_%": o.mean() * 100, "hold_t": nw_t(o, h - 1),
                     "dev_top_%": top[(top.index >= start) & (top.index < split)].mean() * 100,
                     "hold_top_%": top[top.index >= split].mean() * 100})
res = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
res.to_csv(f"{OUT}/xbrl_factors.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(2).to_string(index=False))
