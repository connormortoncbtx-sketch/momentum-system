"""
research/v3_search.py -- restructured core, chosen on DEV (2017-2023) only.
The single best design by dev Sharpe is then scored ONCE on HOLDOUT (2024-01 -> 2026-09).
"""
import itertools, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import backtest as B

DATA = sys.argv[1]
OUT = Path("research/results"); OUT.mkdir(parents=True, exist_ok=True)
DEV = ("2017-01-06", "2023-12-29")
HOLD = ("2024-01-05", "2026-09-18")   # last formation week with a full Mon->Mon return

p = B.load_panel(DATA, "2026-10-02")
f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
rk = lambda x: B.xrank(x, U)
signals = {
    "live_proxy": (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80,
    "m12_1": f["m12_1"],
    "rs_live": f["rs_live"],
    "m12_1+hi52": rk(f["m12_1"]) + rk(f["hi52"]),
    "m12_1+reversal": rk(f["m12_1"]) + rk(-f["r1w"]),
    "m12_1+hi52+reversal": rk(f["m12_1"]) + rk(f["hi52"]) + rk(-f["r1w"]),
}
grid = list(itertools.product(signals, (2, 4, 8), (20, 30), (1, 2), ("equal", "invvol")))
if "--holdout-only" in sys.argv and (OUT / "v3_dev_grid.csv").exists():
    grid = []
rows = []
t0 = time.time()
for sig, k, n, bmult, wt in grid:
    s = signals[sig].loc[DEV[0]:DEV[1]]
    r = B.simulate(p, s, U.loc[s.index], n=n, buffer=n * bmult, k=k, cost_bps=15, weighting=wt, vol=f["vol12"])
    st = B.stats(r)
    rows.append({"signal": sig, "k": k, "n": n, "buffer": n * bmult, "weights": wt, **st})
print(f"{len(grid)} dev runs in {(time.time()-t0)/60:.1f} min")
if rows:
    dev = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    dev.to_csv(OUT / "v3_dev_grid.csv", index=False)
else:
    dev = pd.read_csv(OUT / "v3_dev_grid.csv")

spy = p.df("d1_close")["SPY"]; spy_r = (spy.shift(-2) / spy.shift(-1) - 1)
print("\nSPY dev:", {k: round(v, 2) for k, v in B.stats(spy_r.loc[DEV[0]:DEV[1]]).items()})
print("\nTop 10 designs on DEV (2017-2023):")
print(dev.head(10).round(2).to_string(index=False))
print("\nSpread of dev Sharpe across all designs:", dev.Sharpe.describe().round(2).to_dict())

best = dev.iloc[0]
s = signals[best.signal].loc[HOLD[0]:HOLD[1]]
hold = B.simulate(p, s, U.loc[s.index], n=int(best.n), buffer=int(best.buffer), k=int(best.k),
                  cost_bps=15, weighting=best.weights, vol=f["vol12"])
print("\n=== HOLDOUT (2024-01 -> 2026-09), pre-registered pick:", dict(best[["signal","k","n","buffer","weights"]]))
print("v3   :", {k: round(v, 2) for k, v in B.stats(hold).items()}, "| missing-return fills:", B.simulate.missing)
print("SPY  :", {k: round(v, 2) for k, v in B.stats(spy_r.loc[HOLD[0]:HOLD[1]]).items()})
print("by year v3:", B.by_year(hold).round(1).to_dict(), " SPY:", B.by_year(spy_r.loc[HOLD[0]:HOLD[1]].dropna()).round(1).to_dict())
pd.DataFrame({"v3": hold, "SPY": spy_r.reindex(hold.index)}).to_csv(OUT / "v3_holdout_weekly.csv")
