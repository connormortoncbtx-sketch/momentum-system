"""
research/pead.py -- post-earnings-announcement drift, built on the weekly panel.

An event becomes usable at the Friday close of the week containing d1 (the
2nd session after the filing), so it can be traded at the following Monday
close -- the system's normal entry. No look-ahead.

Stage 1  event study (dev years): abnormal return vs SPY by surprise quintile.
Stage 2  portfolio grid on dev 2017-2023; single pick scored once on holdout.

    python research/pead.py <research-data dir> [--holdout-only]
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import backtest as B  # noqa: E402

DATA = sys.argv[1]
OUT = Path("research/results"); OUT.mkdir(parents=True, exist_ok=True)
DEV = ("2017-01-06", "2023-12-29")
HOLD = ("2024-01-05", "2026-09-18")

p = B.load_panel(DATA, "2026-10-02")
f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
import os
SFX = ""
if os.environ.get("UNIV") == "top500":
    # large caps: 500 most-traded names (4-wk avg $ volume) among the base universe
    U = U & (f["adv"].where(U).rank(axis=1, ascending=False) <= 500)
    SFX = "_top500"
ev = pd.read_parquet(f"{DATA}/earnings_events.parquet")
ev = ev.dropna(subset=["sue"])
ev["avail_week"] = pd.to_datetime(ev["d1"]) + pd.to_timedelta(4 - pd.to_datetime(ev["d1"]).dt.weekday, unit="D")
ev = ev[ev.symbol.isin(p.symbols) & ev.avail_week.isin(p.weeks)]
print(f"events usable: {len(ev):,}  symbols: {ev.symbol.nunique():,}  "
      f"{ev.avail_week.min():%Y-%m} -> {ev.avail_week.max():%Y-%m}")

# ── STAGE 1: event study ─────────────────────────────────────────────────────
d1c, cl = p.df("d1_close"), p.df("close")
widx = {w: i for i, w in enumerate(p.weeks)}
spy_d1c, spy_cl = d1c["SPY"], cl["SPY"]
H = (1, 4, 8, 12)
rows = []
for e in ev.itertuples():
    i = widx[e.avail_week]
    if i + 1 >= len(p.weeks):
        continue
    entry = d1c.iat[i + 1, d1c.columns.get_loc(e.symbol)]
    se = spy_d1c.iat[i + 1]
    if not np.isfinite(entry) or entry <= 0:
        continue
    r = {"symbol": e.symbol, "week": e.avail_week, "sue": e.sue, "r2_abn": e.r2_abn, "gap": e.gap,
         "in_universe": bool(U.iat[i, U.columns.get_loc(e.symbol)])}
    for h in H:
        j = i + h
        if j < len(p.weeks):
            x = cl.iat[j, cl.columns.get_loc(e.symbol)]
            r[f"abn_{h}w"] = (x / entry - 1) - (spy_cl.iat[j] / se - 1) if np.isfinite(x) else np.nan
    rows.append(r)
es = pd.DataFrame(rows)
es = es[es.in_universe]
for c in [f"abn_{h}w" for h in H]:
    es[c] = es[c].clip(-1, 3)
es["q"] = es.groupby(es.week.dt.year)["sue"].transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop") + 1)

def study(d):
    t = d.groupby("q")[[f"abn_{h}w" for h in H]].mean() * 100
    t["n"] = d.groupby("q").size()
    return t

dev_es = es[(es.week >= DEV[0]) & (es.week <= DEV[1])]
print("\nSTAGE 1 (dev 2017-2023): mean abnormal return vs SPY (%), by surprise quintile (5 = best reaction)")
print(study(dev_es).round(2).to_string())
t5 = dev_es[dev_es.q == 5]
for h in H:
    x = t5[f"abn_{h}w"].dropna()
    print(f"  Q5 {h:>2}w: mean {x.mean()*100:+.2f}%  t={x.mean()/x.std()*np.sqrt(len(x)):.1f}  "
          f"(round-trip cost at 15bps = 0.30%)")
es.to_parquet(OUT / f"pead_event_study{SFX}.parquet", index=False)

# ── STAGE 2: portfolio grid ──────────────────────────────────────────────────
def event_score(col, max_age):
    """Score matrix: value of `col` for each symbol's latest event within max_age weeks."""
    m = pd.DataFrame(np.nan, index=p.weeks, columns=p.symbols)
    for e in ev.itertuples():
        i = widx[e.avail_week]
        v = getattr(e, col)
        for a in range(max_age):
            if i + a < len(p.weeks):
                m.iat[i + a, m.columns.get_loc(e.symbol)] = v   # later events overwrite
    return m

rk = lambda x: B.xrank(x, U)
cache = {}
def scores(name, age):
    key = (name, age)
    if key in cache:
        return cache[key]
    if name == "sue":
        s = event_score("sue", age)
    elif name == "r2_abn":
        s = event_score("r2_abn", age)
    elif name == "sue+m12_1":
        s = rk(event_score("sue", age)) + rk(f["m12_1"]).where(event_score("sue", age).notna())
    elif name == "sue_gap_up":                       # only reactions that gapped up and held
        s = event_score("sue", age).where(event_score("gap", age) > 0)
    cache[key] = s
    return s

holdout_only = "--holdout-only" in sys.argv
grid = list(itertools.product(["sue", "r2_abn", "sue+m12_1", "sue_gap_up"], (4, 8, 13), (1, 2, 4), (20, 30), ("equal", "invvol")))
rows = []
if not holdout_only:
    t0 = time.time()
    for name, age, k, n, wt in grid:
        s = scores(name, age).loc[DEV[0]:DEV[1]]
        mask = U.loc[s.index] & s.notna() & (s > (0 if name in ("sue", "r2_abn", "sue_gap_up") else -np.inf))
        r = B.simulate(p, s, mask, n=n, buffer=2 * n, k=k, cost_bps=15, weighting=wt, vol=f["vol12"])
        rows.append({"score": name, "max_age_w": age, "k": k, "n": n, "weights": wt, **B.stats(r)})
    print(f"\n{len(grid)} dev portfolio runs in {(time.time()-t0)/60:.1f} min")
    dev = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    dev.to_csv(OUT / f"pead_dev_grid{SFX}.csv", index=False)
else:
    dev = pd.read_csv(OUT / f"pead_dev_grid{SFX}.csv")

spy_r = (d1c["SPY"].shift(-2) / d1c["SPY"].shift(-1) - 1)
print("\nSPY dev:", {k: round(v, 2) for k, v in B.stats(spy_r.loc[DEV[0]:DEV[1]]).items()})
print("Top 10 PEAD designs on DEV:")
print(dev.head(10).round(2).to_string(index=False))
print("Dev Sharpe spread:", dev.Sharpe.describe().round(2).to_dict())

best = dev.iloc[0]
s = scores(best.score, int(best.max_age_w)).loc[HOLD[0]:HOLD[1]]
mask = U.loc[s.index] & s.notna() & (s > (0 if best.score in ("sue", "r2_abn", "sue_gap_up") else -np.inf))
B.simulate.missing = 0
hold = B.simulate(p, s, mask, n=int(best.n), buffer=2 * int(best.n), k=int(best.k), cost_bps=15,
                  weighting=best.weights, vol=f["vol12"])
print("\n=== HOLDOUT (2024-01 -> 2026-09), pre-registered pick:",
      dict(best[["score", "max_age_w", "k", "n", "weights"]]))
print("PEAD :", {k: round(v, 2) for k, v in B.stats(hold).items()}, "| missing fills:", B.simulate.missing)
print("SPY  :", {k: round(v, 2) for k, v in B.stats(spy_r.loc[HOLD[0]:HOLD[1]]).items()})
print("by year PEAD:", B.by_year(hold).round(1).to_dict(),
      " SPY:", B.by_year(spy_r.loc[HOLD[0]:HOLD[1]].dropna()).round(1).to_dict())
pd.DataFrame({"pead": hold, "SPY": spy_r.reindex(hold.index)}).to_csv(OUT / f"pead_holdout_weekly{SFX}.csv")
