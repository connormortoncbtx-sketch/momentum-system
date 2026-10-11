"""
research/post_filters.py -- post-scoring filter scan (Oct 2026)

Question: after a score picks a candidate pool, does ANY factor reliably tell
the good candidates from the bad ones?  (V1 filtered on prior-week return and
Monday's move, buying at Tuesday's open.)

Pools (each week, from the research universe: $5, $2M/day, no funds):
  live_tue   top 50 by the live momentum proxy, buy TUESDAY OPEN -> Friday close
             (V1 timing; Monday's move and Tuesday's gap are known at entry)
  live_mon   top 50 by the live momentum proxy, Monday close -> Friday close
  slow_4w    top 50 by 12-1 momentum, Monday close -> 4 weeks later
  slow_13w   top 50 by 12-1 momentum, Monday close -> 13 weeks later
Within each pool, candidates are split into quintiles by each factor; we record
each quintile's return minus the pool average. Dev 2017-2023, holdout 2024-26.
Newey-West t-stats. Then: top-10 by score after dropping one quintile (chosen
on dev) vs plain top-10, scored on the holdout.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
OUT = os.environ.get("OUT", "research/results")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
C, H, d1o, d1c, d2o = p.df("close"), p.df("high"), p.df("d1_open"), p.df("d1_close"), p.df("d2_open")
dv = p.df("dollar_vol")
rk = lambda x: B.xrank(x, U)
start, split = pd.Timestamp("2017-01-06"), pd.Timestamp("2024-01-01")
clip = lambda x: x.where(x.abs() < 3)

# ── forward returns (aligned to formation week t) ──────────────────────────
RET = {
    "live_tue": clip(C.shift(-1) / d2o.shift(-1) - 1),
    "live_mon": clip(C.shift(-1) / d1c.shift(-1) - 1),
    "slow_4w": clip(C.shift(-4) / d1c.shift(-1) - 1),
    "slow_13w": clip(C.shift(-13) / d1c.shift(-1) - 1),
}
proxy = (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80
SCORE = {"live_tue": proxy, "live_mon": proxy, "slow_4w": f["m12_1"], "slow_13w": f["m12_1"]}
LAG = {"live_tue": 0, "live_mon": 0, "slow_4w": 3, "slow_13w": 12}

# ── factors, all known at the decision time ────────────────────────────────
r1 = f["r1w"]
spy = C["SPY"]; spy_r = spy / spy.shift(1) - 1
cov = r1.rolling(52, min_periods=40).cov(spy_r) if False else None
beta = (r1.mul(spy_r, axis=0).rolling(52, min_periods=40).mean()
        - r1.rolling(52, min_periods=40).mean().mul(spy_r.rolling(52, min_periods=40).mean(), axis=0)) \
       .div(spy_r.rolling(52, min_periods=40).var(), axis=0)
resid = r1.sub(beta.mul(spy_r, axis=0))
F = {
    "prior_week_ret": r1,
    "prior_4wk_ret": f["r4w"],
    "prior_week_vs_spy": r1.sub(spy_r, axis=0),
    "dist_from_52w_high": 1 - f["hi52"],
    "trend_count": f["trend"] + 1e-6 * rk(f["rs_live"]),
    "m12_1": f["m12_1"], "m3_1": f["m3_1"],
    "vol_12wk": f["vol12"],
    "max_week_12wk (lottery)": r1.rolling(12, min_periods=10).max(),
    "beta_52wk": beta,
    "idio_vol_52wk": resid.rolling(52, min_periods=40).std(),
    "dollar_volume (size)": f["adv"],
    "volume_surge (last wk / 12wk)": dv / dv.rolling(12, min_periods=10).mean().shift(1),
    "price_level": f["price"],
    "close_vs_week_high": C / H,
    "friday_vs_prior_mean (stretch)": C / C.rolling(10).mean() - 1,
    # known only for the Tuesday-open entry (V1 timing):
    "monday_ret (Fri close->Mon close)": (d1c.shift(-1) / C - 1),
    "monday_open_gap": (d1o.shift(-1) / C - 1),
    "monday_intraday (open->close)": (d1c.shift(-1) / d1o.shift(-1) - 1),
    "tuesday_open_gap": (d2o.shift(-1) / d1c.shift(-1) - 1),
}
TUE_ONLY = {"monday_ret (Fri close->Mon close)", "monday_open_gap", "monday_intraday (open->close)", "tuesday_open_gap"}

# events
def event_grid(dates: pd.Series, syms: pd.Series, shift_weeks=0):
    wk = dates + pd.to_timedelta((4 - dates.dt.weekday) % 7, unit="D") - pd.Timedelta(days=7 * shift_weeks)
    a = np.zeros((len(p.weeks), len(p.symbols)), dtype=float)
    i = p.weeks.get_indexer(wk); j = p.symbols.get_indexer(syms); ok = (i >= 0) & (j >= 0)
    a[i[ok], j[ok]] = 1.0
    return pd.DataFrame(a, index=p.weeks, columns=p.symbols)
e = pd.read_parquet(f"{P}/earnings_events.parquet"); e["d0"] = pd.to_datetime(e.d0)
earn_this = event_grid(e.d0, e.symbol)                       # reaction during formation week
nxt = e[e.d0.dt.weekday >= 1]
earn_next = event_grid(nxt.d0, nxt.symbol, shift_weeks=1)   # reaction Tue-Fri of the holding week
ins = pd.read_parquet(f"{P}/insider_buys.parquet"); ins["filing_date"] = pd.to_datetime(ins.filing_date)
ins_wk = event_grid(ins.filing_date, ins.symbol).rolling(4, min_periods=1).max()
F["earnings_last_week (flag)"] = earn_this
F["earnings_in_hold_week (flag)"] = earn_next
F["insider_buy_last_4wk (flag)"] = ins_wk
FLAGS = {k for k in F if "(flag)" in k}


def nw_t(x, lag):
    x = x.dropna().to_numpy(); n = len(x)
    if n < 20 or x.std() == 0:
        return np.nan
    e_ = x - x.mean(); v = e_ @ e_ / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * (e_[l:] @ e_[:-l]) / n
    return x.mean() / np.sqrt(v / n) if v > 0 else np.nan


rows, pool_n = [], 50
for pool, ret in RET.items():
    sc = SCORE[pool]
    m = U & ret.notna() & sc.notna()
    inpool = m & (sc.where(m).rank(axis=1, ascending=False) <= pool_n)
    pool_mean = ret.where(inpool).mean(axis=1)
    for fname, fac in F.items():
        if fname in TUE_ONLY and pool != "live_tue":
            continue
        fv = fac.where(inpool)
        if fname in FLAGS:
            groups = {"flag=1": fv == 1, "flag=0": fv == 0}
        else:
            q = fv.rank(axis=1, pct=True).mul(5).clip(upper=4.999).floordiv(1)
            groups = {f"Q{k+1}": q == k for k in range(5)}
        series = {g: ret.where(msk & inpool).mean(axis=1) - pool_mean for g, msk in groups.items()}
        if fname not in FLAGS:
            series["Q5-Q1"] = series["Q5"] - series["Q1"]
        for g, s in series.items():
            s = s[s.index >= start]
            d, h = s[s.index < split], s[s.index >= split]
            share = (groups[g] & inpool).sum(axis=1).div(inpool.sum(axis=1)).mean() if g in groups else np.nan
            rows.append(dict(pool=pool, factor=fname, group=g, share=share,
                             dev=d.mean() * 100, dev_t=nw_t(d, LAG[pool]),
                             hold=h.mean() * 100, hold_t=nw_t(h, LAG[pool])))
    print("done", pool, flush=True)
res = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
res.to_csv(f"{OUT}/post_filters.csv", index=False)
