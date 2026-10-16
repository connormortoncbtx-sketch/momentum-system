"""
research/theme_clusters.py -- do cross-industry "themes" persist, and can we trade them? (Oct 16, 2026)

Themes are found from price action, not industry codes: each formation Friday t, stocks in
the universe (price >= $3, >= $1M/day, >= 53 weeks listed, no funds, no recent reverse
split) are clustered (k-means on z-scored MARKET-RESIDUAL weekly returns over the last 26
weeks, about 40 stocks per cluster). Stocks riding the same story co-move whatever their
industry.

Cluster "heat" at t (known at the Friday close):
  c_r4 / c_r13  mean member return over 4 / 13 weeks
  breadth       share of members up over 4 weeks;  d_breadth = breadth - breadth 4 weeks ago
  accel         c_r4 - mean member return over weeks t-8..t-4
Forward (tradeable): Monday close after t -> Monday close 4 weeks later, minus the
equal-weight universe.

  1. Cluster IC: Spearman across clusters (>= 8 members) of heat vs next-4-week cluster
     return; t-stat on non-overlapping weeks. Dev 2017-2023, holdout 2024-2026.
  2. Stock level (Fama-MacBeth, weekly cross-sectional rank regression): does the stock's
     cluster heat predict its next 4 weeks beyond its own momentum?
  3. Books (4 staggered tranches, 4-week hold, 25 bps per side): top 10 stocks by own
     13-week return inside the hottest clusters, vs the same without clusters.
  4. Snapshot: the hottest clusters at the latest week, with members (for labelling).
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
W, S = p.weeks, p.symbols
C, d1c = p.df("close"), p.df("d1_close")
U = B.universe(p, f, 3.0, 1e6)
r1 = (C / C.shift(1) - 1).where(lambda x: x.abs() < 3)
resid = r1.sub(r1.where(U).mean(axis=1), axis=0)
r13 = C / C.shift(13) - 1
F4 = (d1c.shift(-5) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 5)
F4x = F4.sub(F4.where(U).mean(axis=1), axis=0)
START, SPLIT, LOOK = pd.Timestamp("2017-01-06"), pd.Timestamp("2024-01-01"), 26
COST = 25.0
names = pd.read_parquet(f"{P}/assets.parquet").set_index("symbol").reindex(S)["name"].fillna("")


def cluster_week(i):
    u = U.iloc[i].fillna(False).to_numpy()
    X = resid.iloc[i - LOOK + 1:i + 1].to_numpy().T[u]
    ok = np.isfinite(X).sum(axis=1) >= LOOK - 3
    cols = np.nonzero(u)[0][ok]
    if len(cols) < 200:
        return None, None
    X = np.nan_to_num(X[ok])
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)
    k = max(10, len(cols) // 40)
    lab = KMeans(n_clusters=k, n_init=2, random_state=0).fit_predict(X)
    return cols, lab


rows, stock_rows, labels = [], [], {}
prev_breadth = {}
idx = [i for i, w in enumerate(W) if w >= START and i >= LOOK + 13]
for n, i in enumerate(idx):
    cols, lab = cluster_week(i)
    if cols is None:
        continue
    labels[W[i]] = pd.Series(lab, index=S[cols])
    r4 = f["r4w"].iloc[i].to_numpy()[cols]; r13v = r13.iloc[i].to_numpy()[cols]
    r4p = (C.iloc[i - 4] / C.iloc[i - 8] - 1).to_numpy()[cols]
    fx = F4x.iloc[i].to_numpy()[cols]
    df = pd.DataFrame({"c": lab, "r4": r4, "r13": r13v, "r4p": r4p, "fx": fx, "sym": S[cols]})
    g = df.groupby("c").agg(n=("sym", "size"), c_r4=("r4", "mean"), c_r13=("r13", "mean"),
                            breadth=("r4", lambda x: (x > 0).mean()), c_r4p=("r4p", "mean"), c_fx=("fx", "mean"))
    g["accel"] = g.c_r4 - g.c_r4p
    # breadth 4 weeks ago for the same members (cluster ids are not stable over time)
    r4_old = (C.iloc[i - 4] / C.iloc[i - 8] - 1).to_numpy()[cols]
    g["d_breadth"] = g.breadth - df.assign(o=r4_old > 0).groupby("c").o.mean()
    g = g[g.n >= 8]
    g["week"] = W[i]
    rows.append(g.reset_index())
    m = df.merge(g[["c_r4", "c_r13", "breadth", "accel", "d_breadth"]], left_on="c", right_index=True)
    m["week"] = W[i]
    stock_rows.append(m)
    if n % 50 == 0:
        print(f"  {W[i].date()} clusters {len(g)} stocks {len(cols)}", flush=True)
CL = pd.concat(rows, ignore_index=True)
ST = pd.concat(stock_rows, ignore_index=True)
CL["period"] = np.where(CL.week < SPLIT, "dev", "holdout")
ST["period"] = np.where(ST.week < SPLIT, "dev", "holdout")
pd.set_option("display.width", 220)

# 1. cluster-level IC
print("\n1. CLUSTER IC (Spearman of heat vs next-4-week cluster excess return); t on non-overlapping weeks")
wk_all = sorted(CL.week.unique()); keep_wk = set(wk_all[::4])
for feat in ("c_r4", "c_r13", "breadth", "accel", "d_breadth"):
    ic = CL.dropna(subset=["c_fx"]).groupby("week").apply(lambda g: g[feat].rank().corr(g.c_fx.rank()))
    out = []
    for per in ("dev", "holdout"):
        x = ic[[(w < SPLIT) == (per == "dev") for w in ic.index]]
        xn = x[[w in keep_wk for w in x.index]]
        out.append(f"{per} IC {x.mean():+.3f} (t {xn.mean() / xn.std() * np.sqrt(len(xn)):+.2f})")
    # spread: top-quintile minus bottom-quintile cluster next-4w excess
    q = CL.dropna(subset=["c_fx"]).copy()
    q["qq"] = q.groupby("week")[feat].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    sp = q.groupby(["period", "qq"]).c_fx.mean().unstack() * 100
    print(f"  {feat:10s} " + " | ".join(out) + f" | top-bottom quintile 4w spread dev {sp.loc['dev', 4] - sp.loc['dev', 0]:+.2f}% "
          f"holdout {sp.loc['holdout', 4] - sp.loc['holdout', 0]:+.2f}%")

# 2. stock-level Fama-MacBeth on ranks
print("\n2. STOCK LEVEL Fama-MacBeth (weekly rank regression of next-4w excess; coef x100, t on non-overlapping weeks)")
X = ["r4", "r13", "c_r4", "c_r13", "breadth", "accel"]
coefs = []
for w, g in ST.dropna(subset=["fx"] + X).groupby("week"):
    Z = g[X].rank(pct=True) - 0.5
    y = g.fx.rank(pct=True) - 0.5
    b = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z.to_numpy()], y.to_numpy(), rcond=None)[0][1:]
    coefs.append(pd.Series(b, index=X, name=w))
FM = pd.DataFrame(coefs)
for per in ("dev", "holdout"):
    x = FM[[(w < SPLIT) == (per == "dev") for w in FM.index]]
    xn = x[[w in keep_wk for w in x.index]]
    print(f"  {per:8s} " + "  ".join(f"{c} {x[c].mean() * 100:+.2f} (t {xn[c].mean() / xn[c].std() * np.sqrt(len(xn)):+.1f})" for c in X))

# 3. books
WR = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 3)


def book(select, H=4, n=10):
    """select(week) -> list of symbols; H staggered tranches, weekly marks, costs on entry+exit."""
    tranches = {}
    out = {}
    for w in sorted(ST.week.unique()):
        tranches[w] = select(w)
        live = [t for t in tranches if 0 <= (W.get_loc(w) - W.get_loc(t)) < H]
        rets = []
        for t in live:
            syms = [s for s in tranches[t] if s in WR.columns]
            if not syms:
                rets.append(0.0); continue
            r = WR.loc[w, syms].fillna(0).mean()
            age = W.get_loc(w) - W.get_loc(t)
            r -= (COST / 1e4) * ((age == 0) + (age == H - 1))
            rets.append(r)
        out[w] = np.mean(rets) if rets else 0.0
    return pd.Series(out)


def st(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 52
    return f"CAGR {(eq.iloc[-1] ** (1 / yrs) - 1) * 100:6.1f}%  Sharpe {r.mean() / r.std() * np.sqrt(52):5.2f}  maxDD {(eq / eq.cummax() - 1).min() * 100:6.1f}%"


def top_in_hot(feat, k_clusters=3, n=10):
    def sel(w):
        g = CL[CL.week == w].nlargest(k_clusters, feat)
        m = ST[(ST.week == w)]
        lab = labels[w]
        members = lab[lab.isin(g.c)].index
        return list(m[m.sym.isin(members)].nlargest(n, "r13").sym)
    return sel


def top_plain(n=10):
    return lambda w: list(ST[ST.week == w].nlargest(n, "r13").sym)


def early_themes(n=10):
    def sel(w):
        g = CL[CL.week == w]
        g = g[(g.accel > 0) & (g.d_breadth > 0)]
        g = g[g.c_r13 < g.c_r13.quantile(0.8)] if len(g) > 5 else g
        g = g.nlargest(3, "d_breadth")
        lab = labels[w]; members = lab[lab.isin(g.c)].index
        m = ST[(ST.week == w) & ST.sym.isin(members)]
        return list(m.nlargest(n, "r4").sym)
    return sel


print("\n3. BOOKS (4-week hold, 4 staggered tranches, 25 bps per side)")
books = {"top 10 by own 13w return (no clusters)": top_plain(),
         "hottest 3 clusters by c_r4 -> top 10 by own 13w": top_in_hot("c_r4"),
         "hottest 3 clusters by c_r13 -> top 10 by own 13w": top_in_hot("c_r13"),
         "broadest 3 clusters (breadth) -> top 10 by own 13w": top_in_hot("breadth"),
         "early themes (accelerating, breadth rising, not yet top) -> top 10 by own 4w": early_themes()}
ew = WR.where(U).mean(axis=1)
for nm, sel in books.items():
    r = book(sel)
    print(f"  {nm:62s} dev {st(r[r.index < SPLIT])} | holdout {st(r[r.index >= SPLIT])}")
e = ew[(ew.index >= START)]
print(f"  {'equal-weight universe':62s} dev {st(e[e.index < SPLIT])} | holdout {st(e[e.index >= SPLIT])}")

# 4. snapshot
w = max(labels)
g = CL[CL.week == w].sort_values("c_r13", ascending=False)
lab = labels[w]
print(f"\n4. SNAPSHOT {w.date()}: hottest clusters by 13-week return")
for _, c in g.head(12).iterrows():
    mem = lab[lab == c.c].index
    top = ST[(ST.week == w) & ST.sym.isin(mem)].nlargest(12, "r13")
    print(f"  cluster {int(c.c):3d} n {int(c.n):3d} | 4w {c.c_r4 * 100:+5.1f}% 13w {c.c_r13 * 100:+6.1f}% breadth {c.breadth:.2f} "
          f"accel {c.accel * 100:+5.1f} | " + ", ".join(f"{s} ({names[s][:22]})" for s in top.sym))
CL.to_parquet(os.environ.get("OUT_CL", "/tmp/theme_clusters.parquet"), index=False)
