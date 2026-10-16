"""
research/theme_filing_flow.py -- can company filings spot a theme BEFORE its prices run? (Oct 16, 2026)

Same weekly price clusters as research/theme_clusters.py (k-means on 26 weeks of
market-residual weekly returns, price >= $3, >= $1M/day). For each cluster at week t:
  pos_share   share of members with a POSITIVE LLM-judged 8-K in the last 8 weeks
              (contract win, partnership, guidance raise, good preliminary results,
              regulatory approval, positive clinical, acquisition; direction >= 1)
  pos_chg     pos_share minus the same share over weeks t-16..t-8 (is good news spreading?)
  neg_share   same for direction <= -1
  quiet_good  pos_chg high while the cluster's 13-week return is still below the median
Coverage: LLM judgements exist only for small caps ($30M-$2B) with study items, 2017-2026.
Tests: Spearman IC of each vs the cluster's next 4 / 13 / 26-week excess return (t on
non-overlapping blocks), dev 2017-2023 / holdout 2024-2026; a 26-week book buying the
members with a recent positive filing inside the top-3 'quiet_good' clusters.
"""
import glob, os, sys, warnings
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
FX = {}
for h in (4, 13, 26):
    Fh = (d1c.shift(-(h + 1)) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 20)
    FX[h] = Fh.sub(Fh.where(U).mean(axis=1), axis=0)
WR = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 3)
START, SPLIT, LOOK, COST = pd.Timestamp("2017-01-06"), pd.Timestamp("2024-01-01"), 26, 25.0

POS = {"contract_win", "partnership", "guidance_raise", "prelim_results_good", "regulatory_approval",
       "clinical_positive", "acquisition"}
J, E = [], []
for d in [f"{P}/llm"] + sorted(glob.glob(f"{P}/llm_full/part*")) + sorted(glob.glob(f"{P}/llm_hist/part*")):
    if os.path.exists(f"{d}/judgements.parquet"):
        J.append(pd.read_parquet(f"{d}/judgements.parquet", columns=["accession", "event_type", "direction"]))
        E.append(pd.read_parquet(f"{d}/events.parquet", columns=["accession", "symbol", "accepted_true"]))
ev = pd.concat(E).drop_duplicates("accession").merge(pd.concat(J).drop_duplicates("accession"), on="accession")
sh = (pd.to_datetime(ev.accepted_true) + pd.Timedelta(hours=8)).dt.normalize()
ev["week"] = sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")


def grid(m):
    g = ev[m & ev.symbol.isin(set(S)) & ev.week.isin(set(W))].groupby(["week", "symbol"]).size().unstack()
    return g.reindex(index=W, columns=S).fillna(0)


gp = grid(ev.event_type.isin(POS) & (ev.direction >= 1))
gn = grid(ev.direction <= -1)
POS8, POS8p = gp.rolling(8, min_periods=1).sum(), gp.shift(8).rolling(8, min_periods=1).sum()
NEG8 = gn.rolling(8, min_periods=1).sum()
COV = grid(ev.direction.notna()).rolling(52, min_periods=1).sum() > 0      # name is covered by the reader
print("positive events:", int(gp.to_numpy().sum()), "negative:", int(gn.to_numpy().sum()), flush=True)

rows, members = [], {}
for i, wk in enumerate(W):
    if wk < START or i < LOOK + 16:
        continue
    u = U.iloc[i].fillna(False).to_numpy()
    X = resid.iloc[i - LOOK + 1:i + 1].to_numpy().T[u]
    ok = np.isfinite(X).sum(axis=1) >= LOOK - 3
    cols = np.nonzero(u)[0][ok]
    if len(cols) < 200:
        continue
    X = np.nan_to_num(X[ok]); X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    lab = KMeans(n_clusters=max(10, len(cols) // 40), n_init=2, random_state=0).fit_predict(X)
    df = pd.DataFrame({"c": lab, "sym": S[cols], "r13": r13.iloc[i].to_numpy()[cols],
                       "cov": COV.iloc[i].to_numpy()[cols], "pos": POS8.iloc[i].to_numpy()[cols] > 0,
                       "posp": POS8p.iloc[i].to_numpy()[cols] > 0, "neg": NEG8.iloc[i].to_numpy()[cols] > 0,
                       **{f"fx{h}": FX[h].iloc[i].to_numpy()[cols] for h in FX}})
    cv = df[df["cov"]]
    g = cv.groupby("c").agg(n_cov=("sym", "size"), pos_share=("pos", "mean"), pos_prev=("posp", "mean"),
                            neg_share=("neg", "mean"))
    g = g.join(df.groupby("c").agg(n=("sym", "size"), c_r13=("r13", "mean"),
                                   **{f"c_fx{h}": (f"fx{h}", "mean") for h in FX}))
    g = g[(g.n >= 8) & (g.n_cov >= 4)]
    g["pos_chg"] = g.pos_share - g.pos_prev
    g["quiet_good"] = g.pos_chg.rank(pct=True) * (g.c_r13 < g.c_r13.median())
    g["week"] = wk
    rows.append(g.reset_index())
    members[wk] = df[["c", "sym", "pos"]]
CL = pd.concat(rows, ignore_index=True)
print(f"cluster-weeks {len(CL):,}, weeks {CL.week.nunique()}", flush=True)

wk_all = sorted(CL.week.unique())
print("\nIC of filing-flow features vs the cluster's next excess return (t on non-overlapping blocks)")
for h in (4, 13, 26):
    keep = set(wk_all[::h]); col = f"c_fx{h}"
    for feat in ("pos_share", "pos_chg", "neg_share", "quiet_good"):
        ic = CL.dropna(subset=[col]).groupby("week").apply(lambda g: g[feat].rank().corr(g[col].rank()))
        out = []
        for per in ("dev", "holdout"):
            x = ic[[(w < SPLIT) == (per == "dev") for w in ic.index]].dropna()
            xn = x[[w in keep for w in x.index]]
            t = xn.mean() / xn.std() * np.sqrt(len(xn)) if len(xn) > 2 else np.nan
            out.append(f"{per} IC {x.mean():+.3f} (t {t:+.2f}, n {len(xn)})")
        print(f"  {h:2d}w {feat:10s} " + " | ".join(out))


def book(H=26, n=10):
    tr, out = {}, {}
    for w in wk_all:
        g = CL[CL.week == w].nlargest(3, "quiet_good")
        m = members[w]
        pick = m[m.c.isin(g.c) & m.pos].sym.tolist()[:n]
        if len(pick) < n:
            pick += m[m.c.isin(g.c) & ~m.sym.isin(pick)].sym.tolist()[: n - len(pick)]
        tr[w] = pick
        rets = []
        for t in [t for t in tr if 0 <= W.get_loc(w) - W.get_loc(t) < H]:
            syms = [s for s in tr[t] if s in WR.columns]
            r = WR.loc[w, syms].fillna(0).mean() if syms else 0.0
            age = W.get_loc(w) - W.get_loc(t)
            rets.append(r - (COST / 1e4) * ((age == 0) + (age == H - 1)))
        out[w] = np.mean(rets)
    return pd.Series(out)


def st(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 52
    return f"CAGR {(eq.iloc[-1] ** (1 / yrs) - 1) * 100:6.1f}%  Sharpe {r.mean() / r.std() * np.sqrt(52):5.2f}  maxDD {(eq / eq.cummax() - 1).min() * 100:6.1f}%"


print("\nBOOKS: members with a recent positive filing inside the 3 'quiet_good' clusters (good news spreading, price not yet hot)")
for H in (13, 26):
    r = book(H)
    print(f"  H{H}: dev {st(r[r.index < SPLIT])} | holdout {st(r[r.index >= SPLIT])}")
ew = WR.where(U).mean(axis=1); ew = ew[ew.index >= START]
print(f"  equal-weight universe: dev {st(ew[ew.index < SPLIT])} | holdout {st(ew[ew.index >= SPLIT])}")
