"""
research/flow_regime.py -- options, credit, macro and short-volume ("dark pool")
regime features vs which top-500 basket beats SPY over the next 4 weeks.

Features at formation Friday t (public by then; z-scored on an expanding window):
  vix, vix_term (VIX/VIX3M), vix_front (VIX9D/VIX), skew (CBOE SKEW),
  putcall_eq (CBOE equity put/call 4-wk avg; ends 2019),
  credit (Baa-10y), curve (10y-3m), real10_chg13 (10y real yield 13-wk change),
  claims_yoy (lag 7d), nfci (lag 7d),
  short_ratio (FINRA off-exchange short volume / total, 4-wk avg -- the input
  behind DIX-style "dark pool" indexes), short_ratio_chg (vs its 26-wk mean)

A feature "survives" only if |t| > 2 with the same sign in BOTH halves of dev
(2017-2020, 2021-2023). Survivors are then scored once on the 2024-26 holdout.

    python research/flow_regime.py <research-data dir>
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RD = sys.argv[1]
M = Path(RD) / "macro"
OUT = Path("research/results")
g = {"__name__": "x", "__file__": str(Path(__file__).parent / "regime_v2.py")}
src = (Path(__file__).parent / "regime_v2.py").read_text()
sys.argv = ["x", RD]
with contextlib.redirect_stdout(io.StringIO()):
    exec(src[:src.index("# ── STAGE B")], g)
EX, CAND, nw, p = g["EX"], g["CAND"], g["nw_slope"], g["p"]
weeks = p.weeks

rd = lambda n, col="value": pd.read_parquet(M / f"{n}.parquet").set_index("date")[col].sort_index()
def at_fri(s, lag_days=0):
    s = s.copy(); s.index = s.index + pd.Timedelta(days=lag_days)
    return s.reindex(s.index.union(weeks)).ffill(limit=10).reindex(weeks)

F = pd.DataFrame(index=weeks)
vix = at_fri(rd("cboe_VIX"))
F["vix"] = vix
F["vix_term"] = vix / at_fri(rd("cboe_VIX3M"))
F["vix_front"] = at_fri(rd("cboe_VIX9D")) / vix
F["skew"] = at_fri(rd("cboe_SKEW"))
pc = pd.read_parquet(M / "cboe_pc_equity.parquet").set_index("date")["p_c_ratio"].sort_index()
F["putcall_eq"] = at_fri(pc.rolling(20).mean())
F["credit"] = at_fri(rd("fred_BAA10Y"))
F["curve"] = at_fri(rd("fred_T10Y3M"))
F["real10_chg13"] = at_fri(rd("fred_DFII10")) - at_fri(rd("fred_DFII10")).shift(13)
cl = at_fri(rd("fred_ICSA").rolling(4).mean(), 7)
F["claims_yoy"] = cl / cl.shift(52) - 1
F["nfci"] = at_fri(rd("fred_NFCI"), 7)
try:
    fs = pd.read_parquet(M / "finra_short.parquet").set_index("date").sort_index()
    sr = (fs.short_volume.rolling(20).sum() / fs.total_volume.rolling(20).sum())
    F["short_ratio"] = at_fri(sr)
    F["short_ratio_chg"] = F["short_ratio"] - F["short_ratio"].rolling(26).mean()
except FileNotFoundError:
    print("FINRA short volume not available yet -- skipping those two features")
F = F.astype(float)
Z = (F - F.expanding(min_periods=52).mean()) / F.expanding(min_periods=52).std()

fwd4 = EX.rolling(4).sum().shift(-3)
halves = {"2017-20": ("2017-01-06", "2020-12-25"), "2021-23": ("2021-01-01", "2023-12-01"),
          "holdout": ("2024-01-05", "2026-08-21")}
rows = []
for z in F.columns:
    for c in CAND[:-1]:
        r = {"feature": z, "basket": c}
        for k, (a, b) in halves.items():
            y = fwd4[c].loc[a:b]
            s, t = nw(y, Z[z].reindex(y.index))
            r[k + " t"] = t
            r[k + " slope%"] = s * 100 if np.isfinite(s) else np.nan
        rows.append(r)
T = pd.DataFrame(rows)
dev_cols = ["2017-20 t", "2021-23 t"]
T["survives"] = (T[dev_cols].abs() > 2).all(axis=1) & (np.sign(T["2017-20 t"]) == np.sign(T["2021-23 t"]))
T.to_csv(OUT / "flow_regime_stageA.csv", index=False)

piv = T.pivot(index="feature", columns="basket", values="2017-20 t").round(1)
piv2 = T.pivot(index="feature", columns="basket", values="2021-23 t").round(1)
print("t-stats, next-4-wk (basket - SPY) on feature, 2017-2020:"); print(piv.to_string())
print("\nsame, 2021-2023:"); print(piv2.to_string())
n = T[dev_cols].notna().all(axis=1).sum()
print(f"\nTests with both halves available: {n}. |t|>2 in 2017-20: {(T['2017-20 t'].abs()>2).sum()}, "
      f"in 2021-23: {(T['2021-23 t'].abs()>2).sum()} (~{0.05*n:.0f} each by chance). "
      f"Same-sign |t|>2 in BOTH halves: {int(T.survives.sum())}")
surv = T[T.survives]
if len(surv):
    print("\nSURVIVORS, scored once on holdout 2024-2026:")
    print(surv[["feature", "basket", "2017-20 t", "2021-23 t", "holdout t", "holdout slope%"]].round(2).to_string(index=False))
else:
    print("\nNo feature/basket link survives both halves of dev; nothing to score on the holdout.")
cov = F.notna().groupby(F.index.year).mean().round(2)
print("\nFeature coverage by year (share of weeks with data):"); print(cov.loc[2017:].to_string())
