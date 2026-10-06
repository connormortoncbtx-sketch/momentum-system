"""
research/regime.py -- does diagnosing the market regime tell us WHICH stocks
will win, and does acting on it beat always-on?

Regime replica: pipeline/02_regime.py rebuilt on weekly ETF closes
(5d->1w, 20d->4w, 50d->10w, 200d->40w). VIX is not in Alpaca: proxy =
SPY 8-week realised vol (annualised, %) + 3 pts. Same 5 dimensions, weights
and label thresholds as live.

Factors (weekly, Monday close -> next Monday close, base universe):
  excess return of the top decile by each score over the equal-weight universe.

Stage 1 (dev 2017-2023): factor excess by regime label; factor momentum;
         momentum-crash state (SPY 2-yr return < 0).
Stage 2: always-on vs five regime-aware strategies x two universes.
         Pick best dev Sharpe; score that and the baseline ONCE on holdout.

    python research/regime.py <research-data dir>
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
HOLD = (pd.Timestamp("2024-01-05"), pd.Timestamp("2026-09-18"))
COST = 15.0

p = B.load_panel(DATA, "2026-10-02")
f = B.features(p)
C = p.df("close")
U = B.universe(p, f, 5.0, 2e6)
U500 = U & (f["adv"].where(U).rank(axis=1, ascending=False) <= 500)

# ── regime replica ───────────────────────────────────────────────────────────
r = lambda s, n: C[s] / C[s].shift(n) - 1
clip = lambda x, a=-1, b=1: x.clip(a, b)
def blend(s, fn, ft, sn, st):
    return (clip(r(s, fn) / ft) + clip(r(s, sn) / st)) / 2

above = lambda s, n: (C[s] > C[s].rolling(n).mean())
pm = lambda b: np.where(b, 1.0, -1.0)

dd = C["SPY"] / C["SPY"].rolling(52, min_periods=20).max() - 1
trend = clip(0.30 * blend("SPY", 1, .015, 4, .05) + 0.30 * blend("SPY", 1, .015, 10, .08)
             + 0.10 * pm(above("SPY", 10)) + 0.08 * pm(above("SPY", 40)) + clip(dd / -0.10) * -0.15)
iwm_rel = (clip((r("IWM", 1) - r("SPY", 1)) / 0.01) + clip((r("IWM", 4) - r("SPY", 4)) / 0.03)) / 2
breadth = clip(0.25 * blend("QQQ", 1, .015, 4, .05) + 0.35 * blend("IWM", 1, .015, 4, .05) + 0.20 * iwm_rel
               + 0.05 * pm(above("QQQ", 10)) + 0.05 * pm(above("IWM", 10)))
vix = r("SPY", 1).rolling(8).std() * np.sqrt(52) * 100 + 3
vix_s = pd.cut(vix, [-np.inf, 13, 17, 21, 27, 35, np.inf], labels=[1, .5, 0, -.5, -.8, -1]).astype(float)
vix_trend = vix - vix.rolling(4).mean()
credit = (clip((r("HYG", 1) - r("SPY", 1)) / 0.005) + clip((r("HYG", 4) - r("SPY", 4)) / 0.02)) / 2
sentiment = clip(vix_s * 0.5 + clip(-vix_trend / 5, -0.5, 0.5) + credit * 0.3)
off1, off4 = (r("XLK", 1) + r("QQQ", 1)) / 2, (r("XLK", 4) + r("QQQ", 4)) / 2
def1 = (r("XLU", 1) + r("XLP", 1) + r("XLV", 1)) / 3
def4 = (r("XLU", 4) + r("XLP", 4) + r("XLV", 4)) / 3
rotation = clip(0.70 * (clip((off1 - def1) / 0.012) + clip((off4 - def4) / 0.04)) / 2
                + 0.30 * clip((r("XLK", 4) - r("XLU", 4)) / 20 / 0.002))
tlt = (clip(-(r("TLT", 1) - r("SPY", 1)) / 0.012) + clip(-(r("TLT", 4) - r("SPY", 4)) / 0.04)) / 2
gld = (clip(-(r("GLD", 1) - r("SPY", 1)) / 0.01) + clip(-(r("GLD", 4) - r("SPY", 4)) / 0.03)) / 2
safety = clip(0.60 * tlt + 0.40 * gld)
composite = 0.25 * trend + 0.25 * breadth + 0.25 * sentiment + 0.15 * rotation + 0.10 * safety
label = pd.Series(np.select([composite >= .35, composite >= .05, composite >= -.20, composite >= -.45],
                            ["risk_on", "trending_mixed", "choppy_neutral", "risk_off_mild"], "risk_off_severe"),
                  index=composite.index).where(composite.notna())
LABELS = ["risk_on", "trending_mixed", "choppy_neutral", "risk_off_mild", "risk_off_severe"]

# ── factors ──────────────────────────────────────────────────────────────────
d1c = p.df("d1_close")
wk_ret = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)
spy_r = wk_ret["SPY"]
rk = lambda x, m: B.xrank(x, m)
def scores(m):
    return {
        "momentum_12_1": f["rs_live"],
        "live_proxy": (0.35 * rk(f["rs_live"], m) + 0.25 * rk(f["trend"], m) + 0.20 * rk(f["hi52"], m)) / 0.8,
        "near_52w_high": f["hi52"],
        "low_vol": -f["vol12"],
        "high_vol": f["vol12"],
        "reversal_1w": -f["r1w"],
    }
S = scores(U)
ew = wk_ret.where(U).mean(axis=1)
fx = {}
for name, s in S.items():
    top = s.where(U & wk_ret.notna()).rank(axis=1, pct=True) > 0.9
    fx[name] = wk_ret.where(top).mean(axis=1) - ew
FX = pd.DataFrame(fx)

# ── STAGE 1 ──────────────────────────────────────────────────────────────────
dev_w = (FX.index >= DEV[0]) & (FX.index <= DEV[1])
print("STAGE 1 (dev): weekly factor excess over the universe (%), by replica regime label")
t = FX[dev_w].groupby(label[dev_w]).mean().reindex(LABELS) * 100
t["weeks"] = label[dev_w].value_counts().reindex(LABELS)
print(t.round(3).to_string())
tt = FX[dev_w].groupby(label[dev_w]).agg(lambda x: x.mean() / x.std() * np.sqrt(len(x))).reindex(LABELS)
print("\nt-stats:"); print(tt.round(2).to_string())
print("\nAll weeks (dev), mean %:", (FX[dev_w].mean() * 100).round(3).to_dict())

print("\nFactor momentum (dev): next-week factor excess % when trailing 8-wk factor excess > 0 vs <= 0")
fm = {}
for c in FX:
    trail = FX[c].rolling(8).sum().shift(1)          # known at formation (excludes this week's outcome)
    on, off = FX[c][dev_w & (trail > 0)], FX[c][dev_w & (trail <= 0)]
    fm[c] = {"after_up %": on.mean() * 100, "t_up": on.mean() / on.std() * np.sqrt(len(on)),
             "after_down %": off.mean() * 100, "t_down": off.mean() / off.std() * np.sqrt(len(off)),
             "n_up/down": f"{len(on)}/{len(off)}"}
print(pd.DataFrame(fm).T.round(3).to_string())
bear = (C["SPY"] / C["SPY"].shift(104) - 1) < 0
print("\nMomentum-crash state (SPY 2-yr return < 0), dev weeks in state:", int(bear[dev_w].sum()),
      "| momentum excess % in bear:", round(FX.momentum_12_1[dev_w & bear].mean() * 100, 3),
      "| not bear:", round(FX.momentum_12_1[dev_w & ~bear].mean() * 100, 3))

# ── STAGE 2 ──────────────────────────────────────────────────────────────────
ROT = ["live_proxy", "near_52w_high", "low_vol", "reversal_1w"]
period = (FX.index >= DEV[0]) & (FX.index <= HOLD[1])
weeks = FX.index[period]

def basket(name, m):
    s = scores(m)[name].loc[weeks]
    return B.simulate(p, s, m.loc[weeks], n=20, buffer=40, k=1, cost_bps=COST).reindex(weeks)

def assemble(choice: pd.Series, series: dict) -> pd.Series:
    """choice[t] = key of series to hold in week t ('SPY' allowed). Switch costs 2x on change."""
    out, prev = [], None
    for w in weeks:
        k = choice.get(w)
        k = k if isinstance(k, str) else "SPY"
        rr = series[k].get(w, np.nan)
        if not np.isfinite(rr):
            rr = spy_r.get(w, 0.0)
        if prev is not None and k != prev:
            rr -= 2 * COST / 1e4
        out.append(rr)
        prev = k
    return pd.Series(out, index=weeks)

# label -> best factor on dev (for S4), and labels where live_proxy factor > 0 (for S1)
dev_means = FX[dev_w].groupby(label[dev_w]).mean()
on_labels = [l for l in LABELS if l in dev_means.index and dev_means.loc[l, "live_proxy"] > 0]
best_by_label = {l: (dev_means.loc[l, ROT].idxmax() if dev_means.loc[l, ROT].max() > 0 else "SPY")
                 for l in LABELS if l in dev_means.index}
print(f"\nS1 on-labels (live_proxy factor > 0 in dev): {on_labels}")
print(f"S4 label -> factor (dev): {best_by_label}")

results, series_store = [], {}
for uname, m in [("base", U), ("top500", U500)]:
    ser = {k: basket(k, m) for k in ROT}
    ser["SPY"] = spy_r.reindex(weeks)
    fm8 = FX["live_proxy"].rolling(8).sum().shift(1)
    fm13 = FX["live_proxy"].rolling(13).sum().shift(1)
    trail13 = FX[ROT].rolling(13).sum().shift(1)
    strategies = {
        "S0 always-on live_proxy": pd.Series("live_proxy", index=weeks),
        "S1 your labels: on/off -> SPY": label.reindex(weeks).map(lambda l: "live_proxy" if l in on_labels else "SPY"),
        "S2a factor momentum 8w -> SPY": (fm8.reindex(weeks) > 0).map({True: "live_proxy", False: "SPY"}),
        "S2b factor momentum 13w -> SPY": (fm13.reindex(weeks) > 0).map({True: "live_proxy", False: "SPY"}),
        # added after Stage 1 (dev-only finding: momentum factor mean-reverts week to week)
        "S2c contrarian: on after 8w lag": (fm8.reindex(weeks) <= 0).map({True: "live_proxy", False: "SPY"}),
        "S3 rotate to best trailing factor": trail13.reindex(weeks).fillna(-1e9).idxmax(axis=1),
        "S4 your labels choose the factor": label.reindex(weeks).map(best_by_label),
        "S5 crash rule (bear -> SPY)": bear.reindex(weeks).map({True: "SPY", False: "live_proxy"}),
    }
    for sname, choice in strategies.items():
        s = assemble(choice, ser)
        key = f"{uname} | {sname}"
        series_store[key] = s
        results.append({"design": key, **B.stats(s[(s.index >= DEV[0]) & (s.index <= DEV[1])]),
                        "% weeks invested in stocks": (choice[(choice.index <= DEV[1])] != "SPY").mean() * 100})
dev_tbl = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
dev_tbl.to_csv(OUT / "regime_dev.csv", index=False)
print("\nSTAGE 2 (dev 2017-2023):")
print(dev_tbl.round(2).to_string(index=False))
print("SPY dev:", {k: round(v, 2) for k, v in B.stats(spy_r[(spy_r.index >= DEV[0]) & (spy_r.index <= DEV[1])]).items()})

best = dev_tbl.iloc[0].design
hold_rows = []
for key in dict.fromkeys([best, "base | S0 always-on live_proxy", "top500 | S0 always-on live_proxy"]):
    s = series_store[key]
    hold_rows.append({"design": ("PICK  " if key == best else "BASE  ") + key,
                      **B.stats(s[(s.index >= HOLD[0]) & (s.index <= HOLD[1])])})
hold_rows.append({"design": "SPY", **B.stats(spy_r[(spy_r.index >= HOLD[0]) & (spy_r.index <= HOLD[1])])})
print(f"\n=== HOLDOUT (2024-01 -> 2026-09), pre-registered pick: {best}")
print(pd.DataFrame(hold_rows).round(2).to_string(index=False))
pd.DataFrame(hold_rows).to_csv(OUT / "regime_holdout.csv", index=False)
lab_hold = label[(label.index >= HOLD[0]) & (label.index <= HOLD[1])].value_counts()
print("\nReplica label mix in holdout:", lab_hold.to_dict())
