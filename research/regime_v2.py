"""
research/regime_v2.py -- pass 3b: can a richer regime diagnosis tell us WHICH
kind of stock to own next week (or to sit in SPY)?

Regime features (known at the formation Friday close t; z-scored on an
expanding window, so no look-ahead):
  disp4        cross-sectional dispersion of weekly returns (4-wk mean)
  breadth40    % of universe above its 40-wk average
  breadth_chg  4-wk change in breadth40
  corr8        average stock correlation proxy (8 wks)
  mkt_vol8     SPY 8-wk realised vol
  vol_trend    mkt_vol8 vs its 26-wk mean
  spy_trend    SPY vs its 40-wk average
  spy_52w      SPY 52-wk return
  credit13     HYG minus TLT, 13-wk return
  size13       IWM minus SPY, 13-wk return
  mom_spread   12-1 return gap, top vs bottom decile (momentum crowding)
  momfx8       live-proxy factor excess, trailing 8 wks (shadow #2's input)

Candidates (top-20 baskets, weekly, buffer 40, Mon close -> Mon close, 15 bps):
  live_proxy, momentum_12_1, near_52w_high, low_vol, high_vol, reversal_1w, SPY

Stage A (dev 2017-2023): slope of next-4-wk candidate return minus SPY on each
  feature, Newey-West t (lag 3). Count |t|>2 vs the number expected by chance.
Stage B (pre-registered designs, dev evaluation 2020-2023 where walk-forward
  predictions exist; pick by dev Sharpe; holdout scored ONCE):
  B0 always live_proxy           B1 always SPY
  B2 ridge on all features, walk-forward, hold the candidate with the highest
     predicted return over SPY (SPY if none > 0)
  B3 best single feature from Stage A: tercile rule fixed on dev data
  B4 shadow #2 rule (live_proxy only after an 8-wk lag)  [reference]

    python research/regime_v2.py <research-data dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import backtest as B  # noqa: E402

DATA = sys.argv[1]
OUT = Path("research/results"); OUT.mkdir(parents=True, exist_ok=True)
DEV = (pd.Timestamp("2017-01-06"), pd.Timestamp("2023-12-29"))
DEV_EVAL = pd.Timestamp("2020-01-03")
HOLD = (pd.Timestamp("2024-01-05"), pd.Timestamp("2026-09-18"))
COST, RIDGE = 15.0, 10.0

p = B.load_panel(DATA, "2026-10-02")
f = B.features(p)
C = p.df("close")
U = B.universe(p, f, 5.0, 2e6)
U500 = U & (f["adv"].where(U).rank(axis=1, ascending=False) <= 500)
d1c = p.df("d1_close")
wk_ret = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)
spy_r = wk_ret["SPY"]

# ── regime features ──────────────────────────────────────────────────────────
r1 = (C / C.shift(1) - 1).where(U).where(lambda x: x.abs() < 1.5)
feat = pd.DataFrame(index=p.weeks)
feat["disp4"] = r1.std(axis=1).rolling(4).mean()
above40 = (C > C.rolling(40, min_periods=30).mean()).astype(float).where(U)
feat["breadth40"] = above40.mean(axis=1)
feat["breadth_chg"] = feat["breadth40"] - feat["breadth40"].shift(4)
ew1 = r1.mean(axis=1)
feat["corr8"] = ew1.rolling(8).var() / r1.rolling(8, min_periods=6).var().mean(axis=1)
spy1 = C["SPY"] / C["SPY"].shift(1) - 1
feat["mkt_vol8"] = spy1.rolling(8).std() * np.sqrt(52)
feat["vol_trend"] = feat["mkt_vol8"] - feat["mkt_vol8"].rolling(26).mean()
feat["spy_trend"] = C["SPY"] / C["SPY"].rolling(40).mean() - 1
feat["spy_52w"] = C["SPY"] / C["SPY"].shift(52) - 1
feat["credit13"] = (C["HYG"] / C["HYG"].shift(13)) - (C["TLT"] / C["TLT"].shift(13))
feat["size13"] = (C["IWM"] / C["IWM"].shift(13)) - (C["SPY"] / C["SPY"].shift(13))
mrk = f["m12_1"].where(U).rank(axis=1, pct=True)
feat["mom_spread"] = f["m12_1"].where(mrk > 0.9).mean(axis=1) - f["m12_1"].where(mrk < 0.1).mean(axis=1)
rk = lambda x, m: B.xrank(x, m)
proxy_base = (0.35 * rk(f["rs_live"], U) + 0.25 * rk(f["trend"], U) + 0.20 * rk(f["hi52"], U)) / 0.8
top = proxy_base.where(U & wk_ret.notna()).rank(axis=1, pct=True) > 0.9
FXlive = wk_ret.where(top).mean(axis=1) - wk_ret.where(U).mean(axis=1)
feat["momfx8"] = FXlive.rolling(8).sum().shift(1)          # same timing as shadow #2
feat = feat.astype(float)
FEATS = list(feat.columns)
Z = (feat - feat.expanding(52).mean()) / feat.expanding(52).std()   # uses data <= t only

# ── candidate baskets (top-500 universe = deployable, fair vs SPY) ───────────
def scores(m):
    return {"live_proxy": (0.35 * rk(f["rs_live"], m) + 0.25 * rk(f["trend"], m) + 0.20 * rk(f["hi52"], m)) / 0.8,
            "momentum_12_1": f["rs_live"], "near_52w_high": f["hi52"], "low_vol": -f["vol12"],
            "high_vol": f["vol12"], "reversal_1w": -f["r1w"]}

weeks = p.weeks[(p.weeks >= DEV[0]) & (p.weeks <= HOLD[1])]
S = scores(U500)
CAND = list(S) + ["SPY"]
ret = {k: B.simulate(p, S[k].loc[weeks], U500.loc[weeks], n=20, buffer=40, k=1, cost_bps=COST).reindex(weeks)
       for k in S}
ret["SPY"] = spy_r.reindex(weeks)
R = pd.DataFrame(ret)
EX = R.sub(R["SPY"], axis=0)                         # candidate minus SPY, per formation week

# ── STAGE A ──────────────────────────────────────────────────────────────────
def nw_slope(y, x, lag=3):
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 60:
        return np.nan, np.nan
    yv, xv = d.iloc[:, 0].values, d.iloc[:, 1].values
    X = np.column_stack([np.ones(len(xv)), xv])
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    e = yv - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S0 = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S0 += w * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S0 @ XtXi))
    return b[1], b[1] / se[1]

fwd4 = EX.rolling(4).sum().shift(-3)                 # next 4 formation weeks
dev_m = (Z.index >= DEV[0]) & (Z.index <= DEV[1])
tt, bb = {}, {}
for c in CAND[:-1]:
    tt[c], bb[c] = {}, {}
    for z in FEATS:
        y = fwd4[c][(fwd4.index >= DEV[0]) & (fwd4.index <= DEV[1])]
        b_, t_ = nw_slope(y, Z[z].reindex(y.index))
        tt[c][z], bb[c][z] = t_, b_ * 100
T = pd.DataFrame(tt)
print("STAGE A (dev 2017-2023): Newey-West t of next-4-wk (candidate - SPY) on each regime feature (z-scored)")
print(T.round(2).to_string())
n_tests = T.size
print(f"\n|t|>2: {(T.abs() > 2).sum().sum()} of {n_tests} tests (about {0.05 * n_tests:.1f} expected by chance)")
print("Slope, % per 4 wks per 1 sd of feature:")
print(pd.DataFrame(bb).round(2).to_string())
T.to_csv(OUT / "regime_v2_stageA_t.csv")

# ── STAGE B ──────────────────────────────────────────────────────────────────
def assemble(choice):
    out, prev = [], None
    for w in weeks:
        k = choice.get(w)
        k = k if isinstance(k, str) else "SPY"
        rr = R.at[w, k]
        if not np.isfinite(rr):
            rr = R.at[w, "SPY"]
        if prev is not None and k != prev:
            rr -= 2 * COST / 1e4
        out.append(rr)
        prev = k
    return pd.Series(out, index=weeks)

# B2 ridge walk-forward: at formation t, train on rows whose outcome is known (<= t-1)
Zw = Z.reindex(weeks)
pred = pd.DataFrame(index=weeks, columns=CAND[:-1], dtype=float)
for i, w in enumerate(weeks):
    # outcome of formation week x is known at Mon close x+2 <= entry (Mon close w+1) iff x <= w-1
    X = Zw.loc[weeks[:i]].dropna()
    if len(X) < 104 or Zw.loc[w].isna().any():
        continue
    xm = np.column_stack([np.ones(len(X)), X.values])
    xt = np.concatenate([[1.0], Zw.loc[w].values])
    pen = RIDGE * np.eye(xm.shape[1]); pen[0, 0] = 0
    for c in CAND[:-1]:
        y = EX[c].reindex(X.index)
        ok = y.notna().values
        beta = np.linalg.solve(xm[ok].T @ xm[ok] + pen, xm[ok].T @ y.values[ok])
        pred.at[w, c] = xt @ beta
ridge_choice = pred.apply(lambda r: (r.idxmax() if r.notna().any() and r.max() > 0 else "SPY"), axis=1)

# B3 best single feature. Selection AND tercile rule fitted only on data before the
# window it is judged on: 2017-2019 for the dev comparison, 2017-2023 for the holdout.
def make_b3(fit_end):
    tt_ = {}
    for c in CAND[:-1]:
        for z in FEATS:
            y = fwd4[c][(fwd4.index >= DEV[0]) & (fwd4.index <= fit_end - pd.Timedelta(weeks=4))]
            tt_[(c, z)] = abs(nw_slope(y, Z[z].reindex(y.index))[1])
    c_best, z_best = max(tt_, key=lambda k: np.nan_to_num(tt_[k]))
    fit = (Z.index >= DEV[0]) & (Z.index <= fit_end)
    qs = Z[z_best][fit].quantile([1 / 3, 2 / 3]).values
    terc = pd.cut(Z[z_best], [-np.inf, qs[0], qs[1], np.inf], labels=["low", "mid", "high"])
    fw = weeks[weeks <= fit_end - pd.Timedelta(weeks=1)]
    dm = EX.loc[fw].groupby(terc.reindex(fw), observed=False).mean()
    rule = {t: (dm.loc[t].idxmax() if dm.loc[t].max() > 0 else "SPY") for t in dm.index}
    return terc.reindex(weeks).map(rule).astype(object), z_best, c_best, rule

b3_dev, z3d, c3d, rule3d = make_b3(DEV_EVAL - pd.Timedelta(weeks=1))
b3_hold, z3h, c3h, rule3h = make_b3(DEV[1])
print(f"\nB3 (fit 2017-2019): feature '{z3d}' (via {c3d}); tercile -> holding {rule3d}")
print(f"B3 (refit 2017-2023, used for holdout): feature '{z3h}' (via {c3h}); tercile -> holding {rule3h}")
single_choice = b3_dev

designs = {
    "B0 always live_proxy": pd.Series("live_proxy", index=weeks),
    "B1 always SPY": pd.Series("SPY", index=weeks),
    "B2 ridge, all 12 features": ridge_choice,
    "B3 single best feature": single_choice,
    "B4 shadow #2 rule (reference)": (feat["momfx8"].reindex(weeks) <= 0).map({True: "live_proxy", False: "SPY"}),
}
series = {k: assemble(v) for k, v in designs.items()}
ev = (weeks >= DEV_EVAL) & (weeks <= DEV[1])
dev_tbl = pd.DataFrame({k: {**B.stats(s[ev]), "% weeks not SPY": (designs[k][ev] != "SPY").mean() * 100}
                        for k, s in series.items()}).T.sort_values("Sharpe", ascending=False)
print(f"\nSTAGE B dev evaluation ({DEV_EVAL.date()} -> {DEV[1].date()}, where walk-forward predictions exist):")
print(dev_tbl.round(2).to_string())
mix = ridge_choice[ev].value_counts(normalize=True).mul(100).round(0)
print("B2 ridge holdings mix in dev (%):", mix.to_dict())

eligible = [k for k in dev_tbl.index if not k.startswith(("B1", "B4"))]
pick = eligible[0]
hm = (weeks >= HOLD[0]) & (weeks <= HOLD[1])
if pick.startswith("B3"):
    series[pick] = assemble(b3_hold)          # holdout uses the rule refit on all dev data
rows = {f"PICK  {pick}": B.stats(series[pick][hm]),
        "B0 always live_proxy": B.stats(series["B0 always live_proxy"][hm]),
        "B1 always SPY": B.stats(series["B1 always SPY"][hm])}
print(f"\n=== HOLDOUT (2024-01 -> 2026-09), pre-registered pick: {pick}")
print(pd.DataFrame(rows).T.round(2).to_string())
if pick.startswith("B2"):
    print("B2 holdings mix in holdout (%):", ridge_choice[hm].value_counts(normalize=True).mul(100).round(0).to_dict())
pd.DataFrame(rows).T.to_csv(OUT / "regime_v2_holdout.csv")
