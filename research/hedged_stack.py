"""
research/hedged_stack.py -- do the stock books diversify better once their shared MARKET
exposure is hedged out? (Oct 16, 2026)

Each US-stock book (SLOW, PEAD, INSIDER, CONTRACT, LIVEFIX) is regressed on weekly IWM and
SPY returns over DEV 2017-2023; the hedged book = book - b_spy*SPY - b_iwm*IWM (short index
futures/ETFs; cost of carry ignored, short-index financing ~ cash rate). Then the same
dev-chosen mixes as research/strategy_stack.py, now able to hold hedged books, the raw
market (SPY) as its own sleeve, ETF trend and crypto.
"""
import os, sys, io, contextlib
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import strategy_stack as S

R = S.R.copy()
mk = S.weekly_from_daily(S.ret[["SPY", "IWM"]].fillna(0)["SPY"]).rename("SPY").to_frame()
mk["IWM"] = S.weekly_from_daily(S.ret["IWM"].fillna(0))
mk = mk.reindex(R.index)
dev = R.index < S.SPLIT
H = pd.DataFrame(index=R.index)
print("BETAS (dev) and hedged-book stats")
HEDGE = os.environ.get("HEDGE", "IWM")          # IWM | SPY | BOTH (BOTH embeds a size bet -- see note)
idx = ["SPY", "IWM"] if HEDGE == "BOTH" else [HEDGE]
print(f"hedge instrument(s): {idx}")
for c in [x for x in ("SLOW", "PEAD", "INSIDER", "CONTRACT", "DOD", "LIVEFIX") if x in R.columns]:
    X = np.c_[np.ones(dev.sum()), mk.loc[dev, idx].to_numpy()]
    b = np.linalg.lstsq(X, R.loc[dev, c].to_numpy(), rcond=None)[0]
    H[c + "_h"] = R[c] - sum(b[j + 1] * mk[ix] for j, ix in enumerate(idx))
    print(f"  {c:9s} " + " ".join(f"beta {ix} {b[j + 1]:+.2f}" for j, ix in enumerate(idx)) +
          f" | alpha {b[0] * 52 * 100:+.1f}%/yr | " + S.line(c + " hedged", H[c + "_h"]).strip())
ALL = pd.concat([H, R[["ETFTREND", "CRYPTO"]], mk[["SPY"]]], axis=1).dropna()
d, h = ALL[ALL.index < S.SPLIT], ALL[ALL.index >= S.SPLIT]
print("\nCORRELATION of hedged books (dev)"); print(d.corr().round(2).to_string())
cols = list(ALL.columns)
mu, cov = d.mean().to_numpy(), d.cov().to_numpy()
neg = lambda x: -(x @ mu) / np.sqrt(x @ cov @ x)
print("\nMIXES (dev-chosen weights; weights sum to 1, long-only in each sleeve)")
MIX = {}
for nm, allow in (("hedged books + ETF + crypto + SPY", cols),
                  ("hedged books + ETF + crypto (no SPY)", [c for c in cols if c != "SPY"]),
                  ("hedged books + ETF (no crypto, no SPY)", [c for c in cols if c not in ("SPY", "CRYPTO")])):
    bnds = [(0, 1) if c in allow else (0, 0) for c in cols]
    x0 = np.array([1.0 if c in allow else 0 for c in cols]); x0 /= x0.sum()
    o = minimize(neg, x0, bounds=bnds, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}])
    r = ALL @ o.x; MIX[nm] = r
    print(S.line(nm, r) + "\n      weights " + ", ".join(f"{c} {v:.2f}" for c, v in zip(cols, o.x) if v > 0.005))
FIN = 0.06 / 52
print("\nLEVERAGE (Kelly from dev; financing 6%/yr on the borrowed part)")
for nm, r in MIX.items():
    dd = r[r.index < S.SPLIT]; k = (dd.mean() - 0.04 / 52) / dd.var()
    for lev in sorted({2.0, 3.0, round(k / 2, 2)}):
        print(S.line(f"{nm[:26]} x{lev:.2f}", lev * r - max(lev - 1, 0) * FIN))
