"""
research/kalshi/mm_perfill.py -- weather market-making edge measured the way the bot earns it (Oct 18, 2026).

The weather MM study (market_making.py) reported maker edge per CONTRACT (1.39c dev / 0.76c holdout) and simulated a
book that captures a fixed SHARE of every taker trade -- both weight big trades heavily. The longshot study showed that
weighting can flip a result. The bot rests a FIXED 5 contracts 2c behind the best bid, so here:
  A. maker edge at the touch, weighted four ways: per contract, per trade, per fixed-5 fill (min(c, 5)), per market
  B. the bot's deep fills: a sweep = trades in one market, same timestamp, same taker side. If it ran >= 2c past its
     first price, a quote 2c behind would have been hit: we fill min(5, contracts at or beyond our level) at
     first -/+ 2c. Edge per fill and per market, by period, price bucket and hours to close.
Bot filters applied: 1-48h before close, price 3-97c. dev < 2025-10-01 <= holdout.
"""
import os
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")
OFF, SIZE = 0.02, 5


def load():
    t = pd.read_parquet(f"{RD}/kalshi/trades_sample.parquet")
    m = pd.read_parquet(f"{RD}/kalshi/markets.parquet", columns=["ticker", "result", "close_time"])
    t = t.merge(m, on="ticker")
    t = t[t.result.isin(["yes", "no"]) & (t.p > 0) & (t.p < 1) & ~t.is_block_trade.fillna(False)]
    t["R"] = (t.result == "yes").astype(float)
    t["h"] = (pd.to_datetime(t.close_time, utc=True) - t.t).dt.total_seconds() / 3600
    t["period"] = np.where(t.t < SPLIT, "dev", "holdout")
    return t


def wstats(x, e, w, key="ticker"):
    out = {"n": len(x), "per_contract_c": (e * x.c).sum() / x.c.sum() * 100, "per_trade_c": e.mean() * 100,
           "per_fill5_c": (e * w).sum() / w.sum() * 100}
    pm = (e * w).groupby(x[key]).sum() / w.groupby(x[key]).sum()
    out["per_market_c"] = pm.mean() * 100
    ev = x[key].str.rsplit("-", n=1).str[0]
    pe = pm.groupby(pm.index.str.rsplit("-", n=1).str[0]).mean()
    out["t_market"] = pm.mean() / (pe.std() / np.sqrt(len(pe))) if len(pe) > 2 else np.nan
    return pd.Series(out)


def main():
    t = load()
    pd.set_option("display.width", 220)
    f = t[(t.h >= 1) & (t.h <= 48) & (t.p >= 0.03) & (t.p <= 0.97)].copy()
    f["e"] = np.where(f.taker_side == "yes", f.p - f.R, f.R - f.p)
    print("A. MAKER EDGE AT THE TOUCH (bot filters), cents")
    print(f.groupby("period").apply(lambda g: wstats(g, g.e, np.minimum(g.c, SIZE))).round(2).to_string())

    # B. sweeps
    g = t.sort_values(["ticker", "t"]).groupby(["ticker", "t", "taker_side"], sort=False)
    sw = g.agg(first=("p", "first"), lo=("p", "min"), hi=("p", "max"), R=("R", "first"), h=("h", "first"),
               period=("period", "first")).reset_index()
    # contracts at or beyond our level
    t2 = t.merge(sw[["ticker", "t", "taker_side", "first"]], on=["ticker", "t", "taker_side"])
    beyond = np.where(t2.taker_side == "no", t2.p <= t2["first"] - OFF + 1e-9, t2.p >= t2["first"] + OFF - 1e-9)
    q = t2[beyond].groupby(["ticker", "t", "taker_side"]).c.sum().rename("q_beyond")
    sw = sw.merge(q, on=["ticker", "t", "taker_side"])
    sw = sw[(sw.h >= 1) & (sw.h <= 48)]
    sw["px"] = np.where(sw.taker_side == "no", sw["first"] - OFF, sw["first"] + OFF)     # our fill price (YES terms)
    sw = sw[(sw.px >= 0.03) & (sw.px <= 0.97)]
    sw["e"] = np.where(sw.taker_side == "no", sw.R - sw.px, sw.px - sw.R)              # bought YES / sold YES at px
    sw["c"] = np.minimum(sw.q_beyond, SIZE)
    print(f"\nB. DEEP FILLS (quote {OFF*100:.0f}c behind, size {SIZE}): sweeps reaching our level")
    print(sw.groupby("period").apply(lambda g: wstats(g, g.e, g.c)).round(2).to_string())
    sw["pb"] = pd.cut(sw.px, [0, .1, .3, .5, .7, .9, 1], labels=["3-10", "10-30", "30-50", "50-70", "70-90", "90-97"])
    sw["hb"] = pd.cut(sw.h, [0, 3, 6, 12, 24, 48], labels=["1-3h", "3-6h", "6-12h", "12-24h", "24-48h"])
    for by in ("pb", "hb"):
        print(f"\nB by {by} (holdout)")
        print(sw[sw.period == "holdout"].groupby(by, observed=True).apply(lambda g: wstats(g, g.e, g.c)).round(2).to_string())
    h = sw[sw.period == "holdout"]
    days = (h.t.max() - h.t.min()).days or 1
    print(f"\nholdout deep fills in sample: {len(h):,} over {days} days in {h.ticker.nunique():,} sampled markets")
    a = wstats(h, h.e, h.c)
    print(f"::notice::weather MM per fill: deep fills holdout {a.per_fill5_c:+.2f}c per contract (fixed-5 fills), "
          f"per market {a.per_market_c:+.2f}c t{a.t_market:.1f}; touch per contract {wstats(f[f.period=='holdout'], f[f.period=='holdout'].e, np.minimum(f[f.period=='holdout'].c, SIZE)).per_contract_c:+.2f}c")


if __name__ == "__main__":
    main()
