"""
research/intraday_rules.py -- do intraday exit/entry rules add value?

Uses 5-min bars for the live-momentum proxy's weekly top picks (ranks 1-10
traded, equal weight). For each pick and each rule, walk the real path:

  entry   mon_open | mon_close (live design) | tue_open
  stop    hard stop s: first bar with low <= entry*(1-s); fill = min(stop, bar open)
  trail   activates when a bar's high >= entry*(1+a); then exits at the first bar whose
          low <= (running high before that bar)*(1-tr); fill = min(level, bar open)
  partial at activation sell half at max(entry*(1+a), bar open); trail the rest
  else    exit at Friday's final close
Costs: 15 bps per side (2 legs per trade). Choose best on dev (2017-2023) by
weekly-basket Sharpe; score that and the live-design baseline once on holdout.

    python research/intraday_rules.py <research-data dir>
"""
import glob
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = sys.argv[1]
OUT = Path("research/results"); OUT.mkdir(parents=True, exist_ok=True)
COST = 2 * 15 / 1e4
DEV_END, HOLD_START = pd.Timestamp("2023-12-29"), pd.Timestamp("2024-01-05")

bars = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/intraday_5m_*.parquet")))
picks = pd.read_csv("research/intraday_picks.csv", parse_dates=["trade_week"])
picks = picks[picks["rank"] <= 10]
bars = bars.merge(picks[["trade_week", "symbol", "rank"]], on=["trade_week", "symbol"])
bars = bars.sort_values(["trade_week", "symbol", "ts"])
bars["day"] = bars.ts.dt.normalize()
print(f"{bars.groupby(['trade_week','symbol']).ngroups:,} traded symbol-weeks, "
      f"{bars.trade_week.nunique()} weeks")

# ── precompute per-path arrays ───────────────────────────────────────────────
paths = []
for (wk, sym), g in bars.groupby(["trade_week", "symbol"], sort=False):
    days = g.day.values
    d0 = days[0]
    first_day = days == d0
    if first_day.all() or first_day.sum() < 10:        # need Monday + at least one more session
        continue
    o, h, l, c = (g[x].values.astype(float) for x in ("o", "h", "l", "c"))
    i_mc = int(first_day.sum())                        # first bar after Monday close
    paths.append({"wk": wk, "sym": sym, "o": o, "h": h, "l": l, "c": c, "i_mc": i_mc,
                  "mon_open": o[0], "mon_close": c[i_mc - 1], "tue_open": o[i_mc]})
print(f"usable paths: {len(paths):,}")


def first_true(mask):
    idx = np.flatnonzero(mask)
    return idx[0] if len(idx) else None


def trade_return(pth, entry, stop, trail, partial):
    if entry == "mon_open":
        e, start = pth["mon_open"], 0          # entry at the open of bar 0; bar 0 range applies
    elif entry == "mon_close":
        e, start = pth["mon_close"], pth["i_mc"]
    else:
        e, start = pth["tue_open"], pth["i_mc"]
    o, h, l, c = pth["o"][start:], pth["h"][start:], pth["l"][start:], pth["c"][start:]
    if len(c) == 0 or not np.isfinite(e) or e <= 0:
        return np.nan
    exit_px = c[-1]
    exit_i = len(c)

    # hard stop
    if stop is not None:
        sp = e * (1 - stop)
        i = first_true(l <= sp)
        if i is not None:
            exit_i, exit_px = i, min(sp, o[i])

    part_ret = None
    if trail is not None:
        a, tr = trail
        act_px = e * (1 + a)
        ia = first_true(h[:exit_i] >= act_px)        # must activate before the stop fires
        if ia is not None:
            if partial:
                part_ret = max(act_px, o[ia]) / e - 1  # half sold at activation
            hwm_prev = np.maximum.accumulate(h)          # running high incl. bar i
            hwm_prev = np.concatenate([[h[ia]], hwm_prev[ia:-1]]) if ia < len(h) - 1 else np.array([h[ia]])
            seg_l, seg_o = l[ia + 1:], o[ia + 1:]
            lvl = hwm_prev[1:len(seg_l) + 1] * (1 - tr) if len(seg_l) else np.array([])
            j = first_true(seg_l <= lvl) if len(seg_l) else None
            if j is not None and ia + 1 + j < exit_i:
                exit_i = ia + 1 + j
                exit_px = min(lvl[j], seg_o[j])
            elif j is None and exit_i == len(c):
                exit_px = c[-1]
    r = exit_px / e - 1
    if part_ret is not None:
        r = 0.5 * part_ret + 0.5 * r
    return r - COST


STOPS = [None, 0.05, 0.07, 0.10, 0.15]
TRAILS = [None, (0.08, 0.04), (0.10, 0.05), (0.17, 0.08), (0.25, 0.10)]
grid = [(e, s, t, pa) for e in ("mon_open", "mon_close", "tue_open") for s in STOPS for t in TRAILS
        for pa in ((False, True) if t else (False,))]
wk_index = pd.DatetimeIndex([pth["wk"] for pth in paths])
t0 = time.time()
res = {}
for combo in grid:
    res[combo] = np.array([trade_return(pth, *combo) for pth in paths])
print(f"{len(grid)} rule sets x {len(paths):,} paths in {(time.time()-t0)/60:.1f} min")


def summarize(r, mask):
    s = pd.Series(r[mask], index=wk_index[mask]).dropna()
    weekly = s.groupby(level=0).mean()
    eq = (1 + weekly).cumprod()
    yrs = len(weekly) / 52
    return {"trades": len(s), "mean_trade%": s.mean() * 100, "win%": (s > 0).mean() * 100,
            "weeks": len(weekly), "CAGR%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
            "Sharpe": weekly.mean() / weekly.std() * np.sqrt(52),
            "maxDD%": (eq / eq.cummax() - 1).min() * 100}


dev_m = wk_index <= DEV_END
hold_m = wk_index >= HOLD_START
label = lambda c: f"{c[0]} | stop {c[1] or '-'} | trail {c[2] or '-'}{' +half' if c[3] else ''}"
rows = [{"rule": label(c), **summarize(res[c], dev_m)} for c in grid]
dev = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
dev.to_csv(OUT / "intraday_rules_dev.csv", index=False)
base = ("mon_close", None, None, False)
live_like = ("mon_close", 0.10, (0.17, 0.08), True)
print("\nDEV (2017-2023), top 12 rule sets:")
print(dev.head(12).round(2).to_string(index=False))
print("\nDEV reference rows:")
print(dev[dev.rule.isin([label(base), label(live_like)])].round(2).to_string(index=False))
print("\nDEV, best rule per entry timing (no stop/trail):")
print(dev[dev.rule.str.endswith("stop - | trail -")].round(2).to_string(index=False))

best = grid[[label(c) for c in grid].index(dev.iloc[0].rule)]
print(f"\n=== HOLDOUT (2024-01 -> 2026-09): pre-registered pick = {label(best)}")
hold_rows = [{"rule": "PICK  " + label(best), **summarize(res[best], hold_m)},
             {"rule": "BASE  " + label(base), **summarize(res[base], hold_m)},
             {"rule": "LIVE~ " + label(live_like), **summarize(res[live_like], hold_m)}]
print(pd.DataFrame(hold_rows).round(2).to_string(index=False))
pd.DataFrame(hold_rows).to_csv(OUT / "intraday_rules_holdout.csv", index=False)
