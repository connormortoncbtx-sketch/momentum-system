"""
research/kalshi/toxicity.py -- can a model tell, before the fact, which resting quotes are about to be picked off?
(Oct 26, 2026; Connor: "is there an angle to machine learning here?")

Why public trades, not our fills: the bot has ~80 independent city-days of fills -- any model on them memorises noise.
The public trade sample (research_data/kalshi/trades_sample.parquet, May 2024 - Oct 2026, ~3M taker trades in the
daily-high series) has ~100x more. Every taker trade is a fill for some maker; we score it from the MAKER's side.

Sample: maker ends up holding the favourite (>= 50c, what the bot buys), 1-48h before close, not block trades.
Labels (cents per contract, maker's view):
  m60     price 60 minutes later (last trade at or before t+60m, within 3h) minus the fill price -- the markout
  settle  result minus fill price (the money, noisier)
Features, all known at the moment of the fill (causal): minute of the hour, hour (ET), hours to close, days ahead,
held price, taker size, seconds since this market's previous trade, trades in this city-day in the last 10 / 60 min,
net taker flow against the maker's side in this market over 30 min, this market's price move over 5 / 30 min (maker's
side), city, weekday.
Model: gradient-boosted trees (sklearn HistGradientBoosting, small: depth 3, 200 iterations, strong min leaf), trained
on DEV (< 2025-10-01), predicting m60. Judged ONLY on the HOLDOUT (>= 2025-10-01): fills sorted into prediction
quintiles; realised m60 and settle per contract by quintile, ±2se clustered by city-day. A useful model = the
"most toxic" quintile clearly worse than the rest on BOTH labels, beyond what the minute-of-hour rule alone explains.
Also reported: the value of a filter that skips the worst 20% / 10% of predicted fills (edge of what remains).
Outputs (aggregates + the model for the bot, both public-data only): results/kalshi_toxicity.csv,
kalshi/toxicity_model.pkl.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")
pd.set_option("display.width", 220)
FEATS = ["minute", "hour_et", "h", "days_ahead", "held", "log_size", "log_gap", "ev_n10", "ev_n60", "flow30",
         "mv5", "mv30", "city", "dow"]


def load():
    t = pd.read_parquet(os.environ.get("TRADES", f"{RD}/kalshi/trades_sample.parquet"))
    m = pd.read_parquet(os.environ.get("MARKETS", f"{RD}/kalshi/markets.parquet"),
                        columns=["ticker", "event_ticker", "series", "result", "close_time"])
    t = t.merge(m, on="ticker")
    t = t[t.result.isin(["yes", "no"]) & (t.p > 0) & (t.p < 1) & ~t.is_block_trade.fillna(False).astype(bool)].copy()
    t["t"] = pd.to_datetime(t.t, utc=True)
    t["R"] = (t.result == "yes").astype(float)
    t["h"] = (pd.to_datetime(t.close_time, utc=True) - t.t).dt.total_seconds() / 3600
    return t.sort_values(["ticker", "t"]).reset_index(drop=True)


def features(t):
    # maker holds YES when the taker sold YES (taker_side 'no'): sign = +1 maker long YES, -1 maker long NO
    t["s"] = np.where(t.taker_side == "no", 1.0, -1.0)
    t["held"] = np.where(t.s > 0, t.p, 1 - t.p)
    et = t.t.dt.tz_convert("America/New_York")
    t["minute"] = t.t.dt.minute
    t["hour_et"] = et.dt.hour
    t["dow"] = et.dt.dayofweek
    mday = pd.to_datetime(t.ticker.str.split("-").str[1], format="%y%b%d", errors="coerce")
    t["days_ahead"] = (mday - et.dt.tz_localize(None).dt.normalize()).dt.days
    t["log_size"] = np.log1p(t.c)
    t["city"] = t.series.astype("category").cat.codes
    g = t.groupby("ticker", sort=False)
    t["log_gap"] = np.log1p(g.t.diff().dt.total_seconds().fillna(86400))
    ns = ((t.t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()).astype("int64")   # unit-safe seconds
    # this market's price moves (YES price), then turned to the maker's side
    out_m5, out_m30, flow = np.full(len(t), np.nan), np.full(len(t), np.nan), np.zeros(len(t))
    p60, p60_ok = np.full(len(t), np.nan), np.zeros(len(t), bool)
    for tk, idx in g.indices.items():
        ts, ps = ns.values[idx], t.p.values[idx]
        signed = (t.s.values[idx] * t.c.values[idx])                # + = taker sold YES (pushes YES down)
        cum = np.cumsum(signed)
        for back, arr in ((300, out_m5), (1800, out_m30)):
            j = np.searchsorted(ts, ts - back, side="right") - 1
            ok = j >= 0
            arr[idx[ok]] = ps[ok] - ps[j[ok]]
        j30 = np.searchsorted(ts, ts - 1800, side="left")
        flow[idx] = cum - np.where(j30 > 0, cum[np.maximum(j30 - 1, 0)], 0) - signed      # flow BEFORE this trade
        k = np.searchsorted(ts, ts + 3600, side="right") - 1
        ok = (k >= 0) & (ts[np.maximum(k, 0)] > ts) & (ts + 3 * 3600 >= ts[np.maximum(k, 0)])
        # markout price: last trade at or before t+60m; require a trade after the fill (else no information)
        p60[idx[ok]] = ps[k[ok]]
    t["mv5"] = t.s * out_m5 * 100
    t["mv30"] = t.s * out_m30 * 100
    f = -t.s * flow                                  # net taker contracts AGAINST the maker's side in the last 30 min
    t["flow30"] = np.sign(f) * np.log1p(np.abs(f))
    t["m60"] = t.s * (p60 - t.p) * 100
    t["settle"] = t.s * (t.R - t.p) * 100
    # city-day activity: trades in this event in the last 10 / 60 min
    for back, col in ((600, "ev_n10"), (3600, "ev_n60")):
        t[col] = 0.0
        for ev, idx in t.groupby("event_ticker", sort=False).indices.items():
            ts = np.sort(ns.values[idx]); order = np.argsort(ns.values[idx], kind="stable")
            cnt = np.arange(len(ts)) - np.searchsorted(ts, ts - back, side="left")
            v = np.empty(len(ts)); v[order] = cnt
            t.loc[t.index[idx], col] = v
    return t


def clustered(df, col):
    """per-contract mean of col and 2 s.e. clustered by city-day."""
    w = df.c
    g = df.assign(v=df[col] * w).groupby("event_ticker").agg(v=("v", "sum"), w=("c", "sum"))
    N = g.w.sum(); m = g.v.sum() / N; G = len(g)
    se = np.sqrt(((g.v - m * g.w) ** 2).sum() / N ** 2 * G / (G - 1)) if G > 2 else np.nan
    return m, 2 * se, G, N


def report(h, label, pred_col="pred"):
    rows = []
    h = h.copy()
    h["q"] = pd.qcut(h[pred_col].rank(method="first"), 5, labels=["Q1 safest", "Q2", "Q3", "Q4", "Q5 most toxic"])
    for q, g in h.groupby("q", observed=True):
        a = clustered(g, "m60"); b = clustered(g, "settle")
        rows.append(dict(model=label, group=str(q), city_days=a[2], contracts=a[3], m60_c=a[0], m60_2se=a[1],
                         settle_c=b[0], settle_2se=b[1]))
    for drop in (0.10, 0.20):
        cut = h[pred_col].quantile(1 - drop)
        keep = h[h[pred_col] < cut]
        a = clustered(keep, "m60"); b = clustered(keep, "settle")
        rows.append(dict(model=label, group=f"skip worst {drop:.0%}", city_days=a[2], contracts=a[3], m60_c=a[0],
                         m60_2se=a[1], settle_c=b[0], settle_2se=b[1]))
    a = clustered(h, "m60"); b = clustered(h, "settle")
    rows.append(dict(model=label, group="ALL", city_days=a[2], contracts=a[3], m60_c=a[0], m60_2se=a[1],
                     settle_c=b[0], settle_2se=b[1]))
    return rows


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    t = load()
    print(f"{len(t):,} trades, {t.ticker.nunique():,} markets", flush=True)
    t = features(t)
    s = t[(t.held >= 0.5) & (t.held <= 0.97) & (t.h >= 1) & (t.h <= 48) & t.m60.notna()].copy()
    dev, hold = s[s.t < SPLIT], s[s.t >= SPLIT]
    print(f"sample: {len(s):,} maker fills on the favourite (dev {len(dev):,} / holdout {len(hold):,}), "
          f"{s.event_ticker.nunique():,} city-days", flush=True)
    # clip the label so a handful of 50c jumps don't dominate the fit; weight by contracts (log-damped)
    y = dev.m60.clip(-30, 30)
    mdl = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=500,
                                        l2_regularization=1.0, categorical_features=[FEATS.index("city")],
                                        random_state=0)
    mdl.fit(dev[FEATS], y, sample_weight=np.log1p(dev.c))
    hold = hold.assign(pred=mdl.predict(hold[FEATS]) * -1)            # higher = more toxic (worse expected markout)
    dev = dev.assign(pred=mdl.predict(dev[FEATS]) * -1)
    # baseline: the minute-of-hour rule only (toxic = :45-:59)
    hold["pred_min"] = hold.minute.between(45, 59).astype(float) + np.random.default_rng(0).random(len(hold)) * 1e-6
    rows = report(hold, "GBM holdout") + report(hold, "minute rule holdout", "pred_min") + report(dev, "GBM dev (in-sample)")
    R = pd.DataFrame(rows)
    out = f"{RD}/results"; os.makedirs(out, exist_ok=True)
    R.to_csv(f"{out}/kalshi_toxicity.csv", index=False)
    print("\n" + R.round(2).to_string(index=False))
    # which features carry it (permutation importance on the holdout, m60)
    from sklearn.inspection import permutation_importance
    sub = hold.sample(min(len(hold), 60000), random_state=0)
    pi = permutation_importance(mdl, sub[FEATS], sub.m60.clip(-30, 30), n_repeats=3, random_state=0,
                                sample_weight=np.log1p(sub.c))
    imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)
    print("\npermutation importance (holdout):\n" + imp.round(4).to_string())
    os.makedirs(f"{RD}/kalshi", exist_ok=True)
    pickle.dump(dict(model=mdl, feats=FEATS, split=str(SPLIT), city_codes=dict(enumerate(
        t.series.astype("category").cat.categories))), open(f"{RD}/kalshi/toxicity_model.pkl", "wb"))
    g = R[(R.model == "GBM holdout")].set_index("group")
    print(f"::notice::toxicity GBM holdout m60 by quintile (safest..toxic): "
          + ", ".join(f"{g.at[q, 'm60_c']:+.1f}" for q in ["Q1 safest", "Q2", "Q3", "Q4", "Q5 most toxic"])
          + f" | settle: " + ", ".join(f"{g.at[q, 'settle_c']:+.1f}" for q in ["Q1 safest", "Q2", "Q3", "Q4", "Q5 most toxic"])
          + f" | skip worst 20%: m60 {g.at['skip worst 20%', 'm60_c']:+.2f} settle {g.at['skip worst 20%', 'settle_c']:+.2f}"
          + f" vs all {g.at['ALL', 'm60_c']:+.2f} / {g.at['ALL', 'settle_c']:+.2f} | top features: " + ", ".join(imp.index[:4]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc(); print(tb); print("::error::" + tb[-900:].replace("\n", " | ")); raise
