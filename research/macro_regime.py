"""
research/macro_regime.py -- does macro/financial regime predict when momentum
works? Tested on 60 years of the momentum factor (Ken French UMD).

Monthly, as of each month end t, using only data PUBLIC by then:
  term          10y - 3m Treasury (monthly avg)            GS10, TB3MS
  term_chg3     3-month change in term
  rate_chg12    12-month change in 10y yield
  ff_chg6       6-month change in fed funds
  credit        Baa - Aaa corporate spread                 BAA, AAA
  credit_chg3   3-month change in credit
  cpi_yoy       CPI y/y, lagged 1 month (released mid next month)
  cpi_trend     3-month change in cpi_yoy
  unemp_chg6    6-month change in unemployment, lagged 1 month
  ip_yoy        industrial production y/y, lagged 1 month
  claims_yoy    4-wk avg initial claims, y/y %, as of t-7 days
  nfci          Chicago Fed financial conditions, as of t-7 days
  mkt_24m       market 24-month return (Cooper et al. market state)
  mkt_vol6      market 6-month realised vol
  umd_vol6      momentum 6-month realised vol (Barroso & Santa-Clara)
  umd_12m       momentum factor's own trailing 12-month return

Target: next-month UMD return (long winners / short losers).
Periods: DEV 1972-2004 (fit), VALID 2005-2016 (select), HOLDOUT 2017-2026 (score once).

    python research/macro_regime.py <research-data dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1]) / "macro"
OUT = Path("research/results"); OUT.mkdir(parents=True, exist_ok=True)
DEV = ("1972-01-31", "2004-12-31")
VAL = ("2005-01-31", "2016-12-31")
HOLD = ("2017-01-31", "2026-08-31")

rd = lambda n: pd.read_parquet(D / f"{n}.parquet").set_index("date")["value"].sort_index()
fac = pd.read_parquet(D / "french_factors.parquet").set_index("date")
mom = pd.read_parquet(D / "french_momentum.parquet").set_index("date")["Mom"]
months = pd.date_range("1960-01-31", "2026-08-31", freq="ME")

def month_end_last(s, lag_days=0):
    s = s.copy(); s.index = s.index + pd.Timedelta(days=lag_days)
    return s.resample("ME").last().reindex(months).ffill(limit=2)

def monthly_obs(s, lag_months=0):
    m = s.resample("ME").last()
    return m.shift(lag_months).reindex(months)

umd_m = (1 + mom).resample("ME").prod().reindex(months) - 1
mkt_d = fac["Mkt-RF"] + fac["RF"]
mkt_m = (1 + mkt_d).resample("ME").prod().reindex(months) - 1

X = pd.DataFrame(index=months)
gs10, tb3, ff = monthly_obs(rd("fred_GS10")), monthly_obs(rd("fred_TB3MS")), monthly_obs(rd("fred_FEDFUNDS"))
X["term"] = gs10 - tb3
X["term_chg3"] = X["term"] - X["term"].shift(3)
X["rate_chg12"] = gs10 - gs10.shift(12)
X["ff_chg6"] = ff - ff.shift(6)
X["credit"] = monthly_obs(rd("fred_BAA")) - monthly_obs(rd("fred_AAA"))
X["credit_chg3"] = X["credit"] - X["credit"].shift(3)
cpi = monthly_obs(rd("fred_CPIAUCSL"), 1)
X["cpi_yoy"] = cpi / cpi.shift(12) - 1
X["cpi_trend"] = X["cpi_yoy"] - X["cpi_yoy"].shift(3)
un = monthly_obs(rd("fred_UNRATE"), 1)
X["unemp_chg6"] = un - un.shift(6)
ip = monthly_obs(rd("fred_INDPRO"), 1)
X["ip_yoy"] = ip / ip.shift(12) - 1
cl = rd("fred_ICSA").rolling(4).mean()
cl_m = month_end_last(cl, lag_days=7)
X["claims_yoy"] = cl_m / cl_m.shift(12) - 1
X["nfci"] = month_end_last(rd("fred_NFCI"), lag_days=7)
X["mkt_24m"] = (1 + mkt_m).rolling(24).apply(np.prod, raw=True) - 1
X["mkt_vol6"] = mkt_d.rolling(126).std().resample("ME").last().reindex(months) * np.sqrt(252)
X["umd_vol6"] = mom.rolling(126).std().resample("ME").last().reindex(months) * np.sqrt(252)
X["umd_12m"] = (1 + umd_m).rolling(12).apply(np.prod, raw=True) - 1
FEATS = list(X.columns)
Z = (X - X.expanding(60).mean()) / X.expanding(60).std()       # expanding z: data <= t only
y = umd_m.shift(-1)                                             # next-month momentum return

per = {"DEV 1972-2004": DEV, "VALID 2005-2016": VAL, "HOLDOUT 2017-2026": HOLD}
sl = lambda s, a, b: s.loc[a:b]

def ols_t(yv, xv):
    d = pd.concat([yv, xv], axis=1).dropna()
    if len(d) < 36:
        return np.nan, np.nan
    Y, x = d.iloc[:, 0].values, d.iloc[:, 1].values
    A = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(A, Y, rcond=None)[0]
    e = Y - A @ b; XtXi = np.linalg.inv(A.T @ A)
    S0 = (A * e[:, None]).T @ (A * e[:, None])
    G = (A[1:] * e[1:, None]).T @ (A[:-1] * e[:-1, None]); S0 += 0.5 * (G + G.T)   # NW lag 1
    se = np.sqrt(np.diag(XtXi @ S0 @ XtXi))
    return b[1] * 100, b[1] / se[1]

print(f"Momentum factor (UMD) monthly mean: " + " | ".join(
    f"{k} {sl(umd_m, *v).mean()*100:+.2f}% (t {sl(umd_m, *v).mean()/sl(umd_m, *v).std()*np.sqrt(len(sl(umd_m, *v))):.1f})"
    for k, v in per.items()))

# ── STAGE A: each feature, each period (holdout shown only AFTER selection below) ─
rows = []
for z in FEATS:
    r = {"feature": z}
    for k in ("DEV 1972-2004", "VALID 2005-2016"):
        b, t = ols_t(sl(y, *per[k]), sl(Z[z], *per[k]))
        r[k + " slope%"], r[k + " t"] = b, t
    rows.append(r)
A = pd.DataFrame(rows).set_index("feature")
A["survives"] = (A["DEV 1972-2004 t"].abs() > 2) & (np.sign(A["DEV 1972-2004 t"]) == np.sign(A["VALID 2005-2016 t"])) \
                & (A["VALID 2005-2016 t"].abs() > 1)
print("\nSTAGE A: next-month momentum return on each regime feature (slope % per 1 sd, Newey-West t)")
print(A.round(2).to_string())
print(f"Features significant in DEV (|t|>2): {(A['DEV 1972-2004 t'].abs()>2).sum()} of {len(FEATS)} "
      f"(~{0.05*len(FEATS):.1f} by chance). Survive into VALID: {A.survives.sum()}")
A.to_csv(OUT / "macro_stageA.csv")

# Macro quadrants (growth x inflation), the classic regime map
growth_up = X["ip_yoy"] > X["ip_yoy"].rolling(12).mean()
infl_up = X["cpi_trend"] > 0
quad = pd.Series(np.select([growth_up & ~infl_up, growth_up & infl_up, ~growth_up & infl_up],
                           ["Goldilocks (growth up, inflation down)", "Reflation (both up)",
                            "Stagflation (growth down, inflation up)"], "Deflation (both down)"), index=months)
qt = {}
for k in ("DEV 1972-2004", "VALID 2005-2016"):
    yy, qq = sl(y, *per[k]), sl(quad, *per[k])
    qt[k] = yy.groupby(qq).mean() * 100
    qt[k + " n"] = yy.groupby(qq).size()
print("\nMomentum next-month return % by macro quadrant:")
print(pd.DataFrame(qt).round(2).to_string())

# ── STAGE B: strategies on the momentum factor ──────────────────────────────
def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {"months": len(r), "ann%": ((eq.iloc[-1]) ** (12 / len(r)) - 1) * 100,
            "vol%": r.std() * np.sqrt(12) * 100, "Sharpe": r.mean() / r.std() * np.sqrt(12),
            "maxDD%": (eq / eq.cummax() - 1).min() * 100, "worst_month%": r.min() * 100}

nxt = umd_m.shift(-1)
w = {}
w["S0 always momentum"] = pd.Series(1.0, index=months)
tv = sl(X["umd_vol6"], *DEV).median()                            # target vol fixed from DEV
w["S1 volatility-managed (B&SC)"] = (tv / X["umd_vol6"]).clip(0, 2)
w["S2 market state: off after 24m market loss"] = (X["mkt_24m"] > 0).astype(float)
w["S3 off in stress (NFCI > 0)"] = (X["nfci"] <= 0).astype(float)
# ridge, walk-forward: fit on months whose outcome is known (< t), predict t
pred = pd.Series(np.nan, index=months)
Zf = Z[FEATS]
for i, t in enumerate(months):
    if t < pd.Timestamp("1977-01-31"):
        continue
    trn = Zf.index[:i]
    d = pd.concat([Zf.loc[trn], nxt.loc[trn]], axis=1).dropna()
    if len(d) < 60 or Zf.loc[t].isna().any():
        continue
    A_ = np.column_stack([np.ones(len(d)), d.iloc[:, :-1].values]); pen = 12.0 * np.eye(A_.shape[1]); pen[0, 0] = 0
    beta = np.linalg.solve(A_.T @ A_ + pen, A_.T @ d.iloc[:, -1].values)
    pred.loc[t] = np.concatenate([[1.0], Zf.loc[t].values]) @ beta
w["S4 ridge on all 16 features (on/off)"] = (pred > 0).astype(float).where(pred.notna())
w["S5 vol-managed + ridge on/off"] = w["S1 volatility-managed (B&SC)"] * w["S4 ridge on all 16 features (on/off)"]
surv = list(A.index[A.survives])
if surv:
    sgn = np.sign(A.loc[surv, "DEV 1972-2004 t"])
    comp = (Z[surv] * sgn).mean(axis=1)
    w[f"S6 surviving features ({', '.join(surv)}) > 0"] = (comp > 0).astype(float).where(comp.notna())

res = {k: {pk: stats((v * nxt).loc[pv[0]:pv[1]]) for pk, pv in per.items() if pk != "HOLDOUT 2017-2026"}
       for k, v in w.items()}
tbl = pd.DataFrame({(k, pk): s for k, d in res.items() for pk, s in d.items()}).T
print("\nSTAGE B: momentum-factor strategies (DEV fit, VALID select)")
print(tbl.round(2).to_string())
val = tbl.xs("VALID 2005-2016", level=1)
cands = [k for k in val.index if not k.startswith("S0")]
pick = val.loc[cands, "Sharpe"].idxmax()
print(f"\n=== Pre-registered pick (best VALID Sharpe): {pick}")
hold = {k: stats((w[k] * nxt).loc[HOLD[0]:HOLD[1]]) for k in dict.fromkeys([pick, "S0 always momentum"])}
print("HOLDOUT 2017-2026 (scored once):")
print(pd.DataFrame(hold).T.round(2).to_string())
pd.DataFrame(hold).T.to_csv(OUT / "macro_holdout.csv")
print("\nHoldout Stage-A check for surviving features (diagnostic, not used for selection):")
for z in surv:
    print(f"  {z}: holdout slope {ols_t(sl(y, *HOLD), sl(Z[z], *HOLD))[0]:+.2f}% t {ols_t(sl(y, *HOLD), sl(Z[z], *HOLD))[1]:+.2f}")
w_df = pd.DataFrame(w); w_df.to_csv(OUT / "macro_weights_monthly.csv")
