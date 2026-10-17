"""
research/stack_v2.py -- the strategy stack, rebuilt honestly (Oct 17, 2026 audit).

Fixes vs strategy_stack.py / hedged_stack.py / shadow #7:
  * ETF-trend sleeve capped at 1x gross (the research book ran 3.2x on average with no financing);
    crypto sleeve capped at 1x gross too.
  * idle cash inside a sleeve earns the T-bill rate; borrowing costs 6%/yr.
  * Sharpe on EXCESS returns (minus the T-bill rate), not total returns.
  * hedged books: long book - beta x IWM (beta fitted on dev), as in shadow #7.
Weights re-chosen on dev 2017-2023 (max excess Sharpe, long-only, sum 1), holdout 2024-26 once.
Adds year-by-year returns, a deflated Sharpe ratio (Bailey & Lopez de Prado 2014) for the number of
designs tried in this research program, and leverage via account-level borrowing.
T-bill: yearly average 3-month rates (approximate, hard-coded; 2025-26 estimated).
"""
import contextlib, io, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import hedged_stack as HS
S = HS.S
SPLIT = S.SPLIT
TBILL = {2016: 0.0032, 2017: 0.0093, 2018: 0.0194, 2019: 0.0206, 2020: 0.0037, 2021: 0.0005, 2022: 0.0202,
         2023: 0.0507, 2024: 0.0497, 2025: 0.0420, 2026: 0.0380}
FIN = 0.06

def rf_daily(idx):
    return pd.Series([TBILL[d.year] / 252 for d in idx], index=idx)

def rf_weekly(idx):
    return pd.Series([TBILL[d.year] / 52 for d in idx], index=idx)

# --- ETF trend, gross <= 1, idle cash earns T-bills ---------------------------------------
px, ret = S.px, S.ret
vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
me = [px.index[px.index <= m][-1] for m in px.resample("ME").last().index if (px.index <= m).any()]
w = pd.DataFrame(0.0, index=px.index, columns=px.columns); prev = pd.Series(0.0, index=px.columns)
cost = pd.Series(0.0, index=px.index)
for i, d in enumerate(me[:-1]):
    sig = np.sign(px.loc[d] / px.shift(63).loc[d] - 1).clip(lower=0)
    tgt = (sig * (0.10 / vol.loc[d]).clip(upper=2.0)).fillna(0)
    tgt = tgt / max((tgt != 0).sum(), 1) * 4
    if tgt.sum() > 1:
        tgt = tgt / tgt.sum()                       # the fix: no borrowing inside the sleeve
    w.loc[(w.index > d) & (w.index <= me[i + 1])] = tgt.values
    cost.loc[px.index[px.index > d][0]] = (tgt - prev).abs().sum() * 5 / 1e4; prev = tgt
g = w.sum(axis=1)
etf_d = (w * ret.fillna(0)).sum(axis=1) - cost + (1 - g).clip(lower=0) * rf_daily(w.index)
ETF = S.weekly_from_daily(etf_d)

# --- crypto trend, gross <= 1 ----------------------------------------------------------------
cp, cr = S.cp, S.cr
cv = cr.rolling(30, min_periods=20).std() * np.sqrt(365)
sig = (cp > cp.rolling(50, min_periods=50).mean()).astype(float).shift(1)
wc = (sig * (0.40 / cv.shift(1)).clip(upper=1.5)).where(cp.notna())
wc = wc.div(wc.notna().sum(axis=1).clip(lower=1), axis=0).fillna(0)
gc = wc.sum(axis=1); wc = wc.div(gc.clip(lower=1), axis=0); gc = wc.sum(axis=1)
cry_d = (wc * cr.fillna(0)).sum(axis=1) - wc.diff().abs().sum(axis=1) * 25 / 1e4 + (1 - gc) * rf_daily(wc.index) * 252 / 365
CRY = S.weekly_from_daily(cry_d[cry_d.index >= "2021-06-01"])

# --- hedged stock books (as shadow #7: beta on dev) -------------------------------------------
R, mk = HS.R, HS.mk
dev = R.index < SPLIT
def hedge(col):
    X = np.c_[np.ones(dev.sum()), mk.loc[dev, "IWM"].to_numpy()]
    b = np.linalg.lstsq(X, R.loc[dev, col].to_numpy(), rcond=None)[0][1]
    return R[col] - b * mk.IWM, b
SLOW_h, bS = hedge("SLOW"); CON_h, bC = hedge("CONTRACT")
books = {"SLOW_h": SLOW_h, "CONTRACT_h": CON_h, "ETFTREND": ETF.reindex(R.index), "CRYPTO": CRY.reindex(R.index)}
for extra in ("PEAD", "INSIDER"):
    if extra in R.columns:
        books[extra + "_h"] = hedge(extra)[0]
A = pd.DataFrame(books)
A["CRYPTO"] = A["CRYPTO"].fillna(rf_weekly(A.index))      # cash before crypto trading starts
A = A.dropna()
RF = rf_weekly(A.index)
X = A.sub(RF, axis=0)                                      # excess returns
d, h = X[X.index < SPLIT], X[X.index >= SPLIT]

def st(r, ex):
    """r total weekly returns, ex excess weekly returns."""
    eq = (1 + r).cumprod(); yrs = len(r) / 52
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, Sh=ex.mean() / ex.std() * np.sqrt(52),
                vol=r.std() * np.sqrt(52) * 100, DD=(eq / eq.cummax() - 1).min() * 100)

def line(nm, r):
    ex = r - RF.reindex(r.index)
    a, b = st(r[r.index < SPLIT], ex[ex.index < SPLIT]), st(r[r.index >= SPLIT], ex[ex.index >= SPLIT])
    return (f"  {nm:40s} dev {a['CAGR']:6.1f}%/yr exSh {a['Sh']:5.2f} vol {a['vol']:5.1f}% DD {a['DD']:6.1f}% | "
            f"hold {b['CAGR']:6.1f}%/yr exSh {b['Sh']:5.2f} DD {b['DD']:6.1f}%")

print(f"betas: SLOW {bS:.3f}, CONTRACT {bC:.3f}; ETF sleeve gross: mean {g.mean():.2f} (was 3.2 uncapped)")
print("\nSLEEVES (total returns; Sharpe on excess over T-bills)")
for c in A.columns:
    print(line(c, A[c]))
print("\nCORRELATION (dev)"); print(d.corr().round(2).to_string())

def opt(cols):
    mu, cov = d[cols].mean().to_numpy(), d[cols].cov().to_numpy()
    f = lambda x: -(x @ mu) / np.sqrt(x @ cov @ x)
    o = minimize(f, np.ones(len(cols)) / len(cols), bounds=[(0, 1)] * len(cols),
                 constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}])
    return pd.Series(o.x, index=cols)

live4 = ["SLOW_h", "CONTRACT_h", "ETFTREND", "CRYPTO"]
mixes = {"shadow #7 weights (frozen)": pd.Series({"SLOW_h": 0.145, "CONTRACT_h": 0.353, "ETFTREND": 0.294, "CRYPTO": 0.208}),
         "live 4, re-optimised on dev": opt(live4),
         "live 4, no crypto": opt(["SLOW_h", "CONTRACT_h", "ETFTREND"])}
if "PEAD_h" in A.columns:
    mixes["live 4 + PEAD_h + INSIDER_h"] = opt([c for c in A.columns])
print("\nMIXES (1x, each sleeve <= 1x gross)")
MIX = {}
for nm, wts in mixes.items():
    r = A[wts.index] @ wts; MIX[nm] = r
    print(line(nm, r) + "\n      " + ", ".join(f"{k} {v:.3f}" for k, v in wts.items() if v > 0.004))

print("\nYEAR BY YEAR (re-optimised live 4, 1x; total return %)")
r = MIX["live 4, re-optimised on dev"]
yr = (1 + r).groupby(r.index.year).prod() - 1
print("  " + "  ".join(f"{y}: {v * 100:+.1f}" for y, v in yr.items()))

# deflated Sharpe: probability the holdout Sharpe beats the best of N trials under the null
def dsr(ex, n_trials, var_sr):
    sr = ex.mean() / ex.std(); T = len(ex); g3, g4 = skew(ex), kurtosis(ex, fisher=False)
    emax = np.sqrt(var_sr) * ((1 - 0.5772) * norm.ppf(1 - 1 / n_trials) + 0.5772 * norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - emax) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return norm.cdf(z), sr * np.sqrt(52), emax * np.sqrt(52)
exh = (r - RF.reindex(r.index))[r.index >= SPLIT]; exa = r - RF.reindex(r.index)
for n in (100, 500, 2000):
    p, sr, e = dsr(exa, n, (0.5 / np.sqrt(52)) ** 2)        # trial Sharpe dispersion ~0.5 annualised
    print(f"  deflated Sharpe, full 2017-26, N={n:5d} trials: Sharpe {sr:.2f} vs expected max of noise {e:.2f} -> P(real) {p:.3f}")

print("\nLEVERAGE (account-level borrowing at 6%/yr)")
for nm in ("live 4, re-optimised on dev", "live 4, no crypto"):
    base = MIX[nm]
    for k in (1.0, 1.5, 2.0, 2.5):
        lr = k * base - (k - 1) * FIN / 52 - 0.0 * RF          # borrowed part costs 6%; base already earns cash on idle
        print(line(f"{nm[:22]} x{k:.1f}", lr))
    # vol targeting (8-week), average leverage ~2 on dev
    rv = base.rolling(8, min_periods=8).std().shift(1); raw = 1 / rv
    devm = base.index < SPLIT
    k = 2.0 / raw[devm].mean(); lev = (k * raw).clip(0.5, 3).fillna(2.0); k *= 2.0 / lev[devm].mean()
    lev = (k * raw).clip(0.5, 3).fillna(2.0)
    print(line(f"{nm[:22]} vol-target ~2x", lev * base - (lev - 1).clip(lower=0) * FIN / 52) +
          f"  avg lev {lev[devm].mean():.2f}/{lev[~devm].mean():.2f}")
pd.DataFrame(MIX).to_csv(f"{S.P}/results/stack_v2_weekly.csv")
for nm, wts in mixes.items():
    print(f"WEIGHTS {nm}: " + json if False else "", end="")
