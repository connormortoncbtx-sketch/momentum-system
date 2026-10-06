"""
research/insiders.py -- insider open-market purchases as a stock-picking signal.

A purchase is usable from the Friday close of the week containing
(filing_date + 1 day): conservative for after-hours filings, no look-ahead.

Signals over a trailing window of L weeks (per symbol, per formation week):
  n_insiders   distinct insiders buying
  rel_value    total $ bought / 4-wk average daily dollar volume
  officer_buy  $ bought by officers (CEO/CFO/etc.), log-scaled
Stage 1 event study on dev, Stage 2 portfolio grid on dev, single holdout score.

    python research/insiders.py <research-data dir> [--holdout-only]
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
HOLD = ("2024-01-05", "2026-07-03")   # SEC insider data ends 2026-06-30

p = B.load_panel(DATA, "2026-10-02")
f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
import os
SFX = ""
if os.environ.get("UNIV") == "top500":
    # large caps: 500 most-traded names (4-wk avg $ volume) among the base universe
    U = U & (f["adv"].where(U).rank(axis=1, ascending=False) <= 500)
    SFX = "_top500"
ib = pd.read_parquet(f"{DATA}/insider_buys.parquet")
ib = ib[(ib.value_usd > 0) & ib.symbol.isin(p.symbols)]
avail = ib.filing_date + pd.Timedelta(days=1)
ib["week"] = avail + pd.to_timedelta((4 - avail.dt.weekday) % 7, unit="D")
ib = ib[ib.week.isin(p.weeks)]
print(f"usable purchases: {len(ib):,}  symbols: {ib.symbol.nunique():,}")

# weekly aggregates (symbol x week)
g = ib.groupby(["week", "symbol"])
wk_val = g.value_usd.sum().unstack().reindex(index=p.weeks, columns=p.symbols)
wk_off = ib[ib.is_officer].groupby(["week", "symbol"]).value_usd.sum().unstack().reindex(index=p.weeks, columns=p.symbols)
# distinct insiders over a window needs owner identities, not weekly counts
owners = ib.groupby(["symbol", "week"]).owner_cik.apply(set)


def trailing(L):
    val = wk_val.fillna(0).rolling(L, min_periods=1).sum()
    off = wk_off.fillna(0).rolling(L, min_periods=1).sum()
    # distinct insiders buying within the trailing L weeks
    arr = np.zeros((len(p.weeks), len(p.symbols)))
    widx = {w: i for i, w in enumerate(p.weeks)}
    col = {c: j for j, c in enumerate(p.symbols)}
    per_sym = {}
    for (sym, wk), s_ in owners.items():
        per_sym.setdefault(sym, []).append((widx[wk], s_))
    for sym, lst in per_sym.items():
        j = col[sym]
        lst.sort()
        touched = sorted({i for i0, _ in lst for i in range(i0, min(i0 + L, len(p.weeks)))})
        for i in touched:
            window = set()
            for k, s_ in lst:
                if i - L < k <= i:
                    window |= s_
            arr[i, j] = len(window)
    n = pd.DataFrame(arr, index=p.weeks, columns=p.symbols)
    rel = val / (f["adv"].replace(0, np.nan))
    return {"n_insiders": n.where(val > 0), "rel_value": rel.where(val > 0),
            "officer_buy": np.log1p(off).where(off > 0)}


# ── STAGE 1: event study on cluster buys ─────────────────────────────────────
d1c, cl = p.df("d1_close"), p.df("close")
spy_d1c, spy_cl = d1c["SPY"], cl["SPY"]
sig4 = trailing(4)
H = (1, 4, 8, 12, 26)
rows = []
ni = sig4["n_insiders"]
widx = {w: i for i, w in enumerate(p.weeks)}
first_week = ib.groupby("symbol").week.apply(list)
for sym, weeks in first_week.items():
    j = p.symbols.get_loc(sym)
    last = -99
    for wk in sorted(set(weeks)):
        i = widx[wk]
        if i - last < 8 or i + 1 >= len(p.weeks):       # one event per 8 weeks per symbol
            continue
        last = i
        if not U.iat[i, j]:
            continue
        entry, se = d1c.iat[i + 1, j], spy_d1c.iat[i + 1]
        if not np.isfinite(entry):
            continue
        r = {"symbol": sym, "week": wk, "n": ni.iat[i, j], "rel": sig4["rel_value"].iat[i, j],
             "officer": bool(np.isfinite(sig4["officer_buy"].iat[i, j]))}
        for h in H:
            k = i + h
            if k < len(p.weeks):
                x = cl.iat[k, j]
                r[f"abn_{h}w"] = np.clip((x / entry - 1) - (spy_cl.iat[k] / se - 1), -1, 3) if np.isfinite(x) else np.nan
        rows.append(r)
es = pd.DataFrame(rows)
dev_es = es[(es.week >= DEV[0]) & (es.week <= DEV[1])]
cols = [f"abn_{h}w" for h in H]
print("\nSTAGE 1 (dev): mean abnormal return vs SPY (%) after insider purchases")
grp = {"single insider": dev_es[dev_es.n == 1], "cluster (2+ insiders)": dev_es[dev_es.n >= 2],
       "cluster 3+": dev_es[dev_es.n >= 3], "officer bought": dev_es[dev_es.officer],
       "top-quintile $ vs volume": dev_es[dev_es.rel >= dev_es.rel.quantile(0.8)]}
t = pd.DataFrame({k: v[cols].mean() * 100 for k, v in grp.items()}).T
t["n"] = [len(v) for v in grp.values()]
print(t.round(2).to_string())
for k, v in grp.items():
    x = v["abn_8w"].dropna()
    print(f"  {k:<26} 8w t-stat {x.mean()/x.std()*np.sqrt(len(x)):.1f}")
es.to_parquet(OUT / f"insider_event_study{SFX}.parquet", index=False)

# ── STAGE 2: portfolio grid ──────────────────────────────────────────────────
rk = lambda x: B.xrank(x, U)
cache = {}
def score(name, L):
    if (name, L) not in cache:
        s = trailing(L) if L != 4 else sig4
        if name == "n_insiders":
            sc = s["n_insiders"] + 0.01 * rk(s["rel_value"]).fillna(0)          # ties broken by size
        elif name == "rel_value":
            sc = s["rel_value"]
        elif name == "cluster_only":
            sc = s["rel_value"].where(s["n_insiders"] >= 2)
        elif name == "officer_buy":
            sc = s["officer_buy"]
        elif name == "rel_value+m12_1":
            sc = rk(s["rel_value"]) + rk(f["m12_1"]).where(s["rel_value"].notna())
        cache[(name, L)] = sc
    return cache[(name, L)]

grid = list(itertools.product(["n_insiders", "rel_value", "cluster_only", "officer_buy", "rel_value+m12_1"],
                              (4, 8, 13), (2, 4), (20, 30), ("equal", "invvol")))
if "--holdout-only" not in sys.argv:
    rows = []
    t0 = time.time()
    for name, L, k, n, wt in grid:
        s = score(name, L).loc[DEV[0]:DEV[1]]
        r = B.simulate(p, s, U.loc[s.index] & s.notna(), n=n, buffer=2 * n, k=k, cost_bps=15,
                       weighting=wt, vol=f["vol12"])
        rows.append({"score": name, "window_w": L, "k": k, "n": n, "weights": wt, **B.stats(r)})
    print(f"\n{len(grid)} dev runs in {(time.time()-t0)/60:.1f} min")
    dev = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    dev.to_csv(OUT / f"insider_dev_grid{SFX}.csv", index=False)
else:
    dev = pd.read_csv(OUT / f"insider_dev_grid{SFX}.csv")

spy_r = (d1c["SPY"].shift(-2) / d1c["SPY"].shift(-1) - 1)
print("\nSPY dev:", {k: round(v, 2) for k, v in B.stats(spy_r.loc[DEV[0]:DEV[1]]).items()})
print("Top 10 insider designs on DEV:")
print(dev.head(10).round(2).to_string(index=False))
print("Dev Sharpe spread:", dev.Sharpe.describe().round(2).to_dict())

# A design must be investable: at least 90% of dev weeks with a portfolio.
# (Sparse signals, e.g. large-cap officer buys, otherwise win on a few cherry weeks.)
dev = dev[dev.weeks >= 0.9 * dev.weeks.max()].sort_values("Sharpe", ascending=False)
print("Eligible (>=90% weeks invested) top 3:")
print(dev.head(3).round(2).to_string(index=False))
best = dev.iloc[0]
s = score(best.score, int(best.window_w)).loc[HOLD[0]:HOLD[1]]
B.simulate.missing = 0
hold = B.simulate(p, s, U.loc[s.index] & s.notna(), n=int(best.n), buffer=2 * int(best.n), k=int(best.k),
                  cost_bps=15, weighting=best.weights, vol=f["vol12"])
print("\n=== HOLDOUT (2024-01 -> 2026-09), pre-registered pick:",
      dict(best[["score", "window_w", "k", "n", "weights"]]))
print("INSIDER:", {k: round(v, 2) for k, v in B.stats(hold).items()}, "| missing fills:", B.simulate.missing)
print("SPY    :", {k: round(v, 2) for k, v in B.stats(spy_r.loc[HOLD[0]:HOLD[1]]).items()})
print("by year:", B.by_year(hold).round(1).to_dict() if len(hold) else {},
      " SPY:", B.by_year(spy_r.loc[HOLD[0]:HOLD[1]].dropna()).round(1).to_dict())
pd.DataFrame({"insider": hold, "SPY": spy_r.reindex(hold.index)}).to_csv(OUT / f"insider_holdout_weekly{SFX}.csv")
