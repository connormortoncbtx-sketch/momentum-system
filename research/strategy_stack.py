"""
research/strategy_stack.py -- stack the surviving books (+ the disadvantage fixes) and push
Sharpe through diversification; then size with leverage (Oct 16, 2026).

Weekly return series (calendar weeks ending Friday), 2017-01 .. 2026-09:
  SLOW      slow 12-1 momentum, hold while top-100, NO stop (the 10% stop failed the daily-bar check)
  LIVEFIX   live-style weekly composite with ALL disadvantage fixes (research/avoid_stack.py)
  CONTRACT  contract-win book chosen on 2017-23 (research/contract_filters.py: all contract
            wins + dev-adopted filters, 10-day hold, 1/vol weights, 25% cap)
  ETFTREND  cross-asset ETF trend, 63-day, long-only, vol-targeted (research/trend_test.py)
  CRYPTO    crypto trend, 50-day MA, long/cash, vol-targeted (2021-06+; cash before)
Mixes chosen on DEV 2017-2023 only: equal weight, inverse-vol, and max-Sharpe (long-only,
weights sum to 1). Scored once on HOLDOUT 2024-2026. Leverage: full / half Kelly from DEV,
financing 6%/yr on the borrowed part.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(__file__))
P = os.environ.get("PANEL", "research_data")
SPLIT, START = pd.Timestamp("2024-01-01"), pd.Timestamp("2017-01-06")


def weekly_from_formation(s):
    """avoid_stack series are labelled by formation Friday t and earn Mon close t+3 -> Mon close t+10."""
    s = s.copy(); s.index = s.index + pd.Timedelta(days=7)
    return s


def weekly_from_daily(r):
    return (1 + r).resample("W-FRI").prod() - 1


import avoid_stack as A  # noqa: E402
SER = {"SLOW": weekly_from_formation(A.sim("SLOW", [])),   # no stop: daily-bar check (research/stop_daily_check.py) showed it hurts
       "LIVEFIX": weekly_from_formation(A.sim("LIVE", ["no_spike", "no_topvol", "no_late", "no_distress", "no_merger",
                                                       "no_offering", "no_badnews", "stop10", "hold50"]))}
cb = pd.read_csv(f"{P}/results/contract_filters_daily.csv", index_col=0, parse_dates=True)["best"]
SER["CONTRACT"] = weekly_from_daily(cb)

D = f"{P}/trend"
e = pd.read_parquet(f"{D}/etf_daily.parquet")
px = e.pivot_table(index=pd.to_datetime(e.timestamp).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None),
                   columns="symbol", values="close").sort_index()
ret = px.pct_change(); vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
me = [px.index[px.index <= m][-1] for m in px.resample("ME").last().index if (px.index <= m).any()]
w = pd.DataFrame(0.0, index=px.index, columns=px.columns); prev = pd.Series(0.0, index=px.columns)
cost = pd.Series(0.0, index=px.index)
for i, d in enumerate(me[:-1]):
    sig = np.sign(px.loc[d] / px.shift(63).loc[d] - 1).clip(lower=0)
    tgt = (sig * (0.10 / vol.loc[d]).clip(upper=2.0)).fillna(0)
    tgt = tgt / max((tgt != 0).sum(), 1) * 4
    w.loc[(w.index > d) & (w.index <= me[i + 1])] = tgt.values
    cost.loc[px.index[px.index > d][0]] = (tgt - prev).abs().sum() * 5 / 1e4; prev = tgt
SER["ETFTREND"] = weekly_from_daily((w * ret).sum(axis=1) - cost)
c = pd.read_parquet(f"{D}/crypto_daily.parquet")
cp = c.pivot_table(index=pd.to_datetime(c.timestamp).dt.normalize().dt.tz_localize(None), columns="symbol", values="close").sort_index()
cr = cp.pct_change(); cv = cr.rolling(30, min_periods=20).std() * np.sqrt(365)
sig = (cp > cp.rolling(50, min_periods=50).mean()).astype(float).shift(1)
wc = (sig * (0.40 / cv.shift(1)).clip(upper=1.5)).where(cp.notna())
wc = wc.div(wc.notna().sum(axis=1).clip(lower=1), axis=0).fillna(0)
rc = (wc * cr).sum(axis=1) - wc.diff().abs().sum(axis=1) * 25 / 1e4
SER["CRYPTO"] = weekly_from_daily(rc[rc.index >= "2021-06-01"])

R = pd.DataFrame(SER)
R = R[(R.index >= START) & (R.index <= "2026-09-25")]
R["CRYPTO"] = R["CRYPTO"].fillna(0.0)
R = R.dropna()
dev, hold = R[R.index < SPLIT], R[R.index >= SPLIT]


def st(r):
    eq = (1 + r).cumprod(); yrs = len(r) / 52
    cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100.0
    return dict(CAGR=cagr, Sharpe=r.mean() / r.std() * np.sqrt(52), vol=r.std() * np.sqrt(52) * 100,
                maxDD=(eq / eq.cummax() - 1).min() * 100, wk=((1 + cagr / 100) ** (1 / 52) - 1) * 100)


def line(name, r):
    d, h = st(r[r.index < SPLIT]), st(r[r.index >= SPLIT])
    return (f"  {name:34s} dev {d['CAGR']:6.1f}%/yr Sh {d['Sharpe']:5.2f} DD {d['maxDD']:6.1f}% | "
            f"holdout {h['CAGR']:6.1f}%/yr Sh {h['Sharpe']:5.2f} DD {h['maxDD']:6.1f}% ({h['wk']:+.2f}%/wk)")


pd.set_option("display.width", 220)
print(f"weeks: dev {len(dev)}, holdout {len(hold)}")
print("\nCOMPONENTS")
for k in R.columns:
    print(line(k, R[k]))
print("\nCORRELATION (dev / holdout)")
print(dev.corr().round(2).to_string()); print(hold.corr().round(2).to_string())

cols = list(R.columns)
mixes = {"equal weight": np.ones(len(cols)) / len(cols)}
iv = 1 / dev.std(); mixes["inverse vol"] = (iv / iv.sum()).to_numpy()
mu, cov = dev.mean().to_numpy(), dev.cov().to_numpy()
neg_sh = lambda x: -(x @ mu) / np.sqrt(x @ cov @ x)
opt = minimize(neg_sh, np.ones(len(cols)) / len(cols), bounds=[(0, 1)] * len(cols),
               constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}])
mixes["max Sharpe (dev)"] = opt.x
no_crypto = [c != "CRYPTO" for c in cols]
x0 = np.where(no_crypto, 1.0, 0.0); x0 /= x0.sum()
opt2 = minimize(neg_sh, x0, bounds=[(0, 1) if nc else (0, 0) for nc in no_crypto],
                constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}])
mixes["max Sharpe, no crypto (dev)"] = opt2.x
print("\nMIXES (weights chosen on dev)")
MIX = {}
for nm, wts in mixes.items():
    r = R[cols] @ wts; MIX[nm] = r
    print(line(nm, r) + "\n      weights " + ", ".join(f"{c} {v:.2f}" for c, v in zip(cols, wts)))

FIN = 0.06 / 52
print("\nLEVERAGE on the max-Sharpe mix (Kelly from dev; financing 6%/yr on borrowed part)")
for nm in ("max Sharpe (dev)", "max Sharpe, no crypto (dev)", "inverse vol"):
    r = MIX[nm]; d = r[r.index < SPLIT]
    k_full = (d.mean() - 0.04 / 52) / d.var()
    for k in sorted({1.0, 2.0, 3.0, round(k_full / 2, 2), round(k_full, 2)}):
        lr = k * r - max(k - 1, 0) * FIN
        print(line(f"{nm} x{k:.2f}", lr))
print("\ntarget 3.00%/week; goal 1.00%/week")
