"""
research/kalshi/obs_value.py -- what must a real-time station feed deliver to be worth building? (Oct 22, 2026)

obs_decided.py: with the precise 5-minute averages (official basis), buckets already decided still traded ~75c --
+22.6c/contract -- but at a 10-minute reading age. Here the same snipe (buy the certain side as a taker at the traded
price, after the taker fee; margin 1F) is scored as a function of the feed's DELAY and PRECISION, holdout 2025-26:
  delay      reading must be >= 1, 2, 5, 10, 20, 30, 60 minutes old at the trade
  precision  exact   : 5-minute average in F (what the official high is built from)
             wholeC  : the same reading rounded to whole degrees C then converted (what free 5-minute feeds carry)
             hourly  : only the :51-:56 readings (hourly METAR-like), exact value
Output per setting: contracts and city-days available at <= 97c, P&L per contract, wrong %, and $ per city-day at
1 contract per trade print (an upper bound on capacity: we'd compete with whoever is already doing this).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import weather_edge as W

pd.set_option("display.width", 220)
MARGIN = 1


def readings(kind):
    a = pd.read_parquet(W.D / "asos5avg.parquet")
    a["station"] = a.station.where(a.station.str.len() == 4, "K" + a.station)
    a = a[a.station.isin(W.TZ)]
    if kind == "wholeC":
        a = a.assign(tmpf=((a.tmpf - 32) / 1.8).round() * 1.8 + 32)
    if kind == "hourly":
        mi = a.valid_utc.dt.minute
        a = a[(mi >= 51) & (mi <= 56)]
    out = []
    for st, g in a.groupby("station"):
        utc_naive = g.valid_utc.dt.tz_localize(None)
        off = g.valid_utc.dt.tz_convert(W.TZ[st]).dt.tz_localize(None) - utc_naive
        g = g.assign(day=(utc_naive + off.min()).dt.normalize().values).sort_values("valid_utc")
        g["cmax"] = g.groupby("day").tmpf.cummax().round()
        out.append(g[["station", "valid_utc", "day", "cmax"]])
    return pd.concat(out, ignore_index=True).sort_values("valid_utc")


def main():
    m = W.market_intervals(pd.read_parquet(W.D / "markets.parquet"))
    t = pd.read_parquet(W.D / "trades_sample.parquet")
    t = t[(t.p > 0) & (t.p < 1) & ~t.is_block_trade.fillna(False)]
    t = t.merge(m[["ticker", "event_ticker", "station", "date", "lo", "hi", "y"]], on="ticker")
    t["t"] = pd.to_datetime(t.t, utc=True)
    PERIOD = os.environ.get("PERIOD", "hold")
    t = t[(t.date >= W.SPLIT) if PERIOD == "hold" else (t.date < W.SPLIT)]           # holdout 2025-26 or dev 2021-24
    rows = []
    for kind in ("exact", "wholeC", "hourly"):
        obs = readings(kind)
        for lag in (1, 2, 5, 10, 20, 30, 60):
            x = t.assign(tq=t.t - pd.Timedelta(minutes=lag)).sort_values("tq")
            x = pd.merge_asof(x, obs.rename(columns={"valid_utc": "tq"}), on="tq", by="station", direction="backward",
                              tolerance=pd.Timedelta("3h"))
            x = x[x.day == x.date]
            dead = np.isfinite(x.hi) & (x.cmax >= x.hi + 1 + MARGIN)
            sure = ~np.isfinite(x.hi) & np.isfinite(x.lo) & (x.cmax >= x.lo + MARGIN)
            g = x[dead | sure]
            side_yes = sure[dead | sure].values
            q = np.where(side_yes, g.p, 1 - g.p)
            win = np.where(side_yes, g.y == 1, g.y == 0)
            pnl = np.where(win, 1 - q, -q) - W.fee(q)
            cheap = q <= 0.97
            gc = g[cheap]
            cd = (pnl[cheap] * gc.c.clip(upper=1)).sum()
            n_cd = gc.event_ticker.nunique()
            rows.append(dict(precision=kind, delay_min=lag, prints_le97=int(cheap.sum()), city_days=n_cd,
                             contracts_le97=gc.c.sum(), avg_price_c=(q[cheap] * gc.c).sum() / gc.c.sum() * 100 if cheap.any() else np.nan,
                             pnl_c=(pnl[cheap] * gc.c).sum() / gc.c.sum() * 100 if cheap.any() else np.nan,
                             wrong_pct=(~win[cheap]).mean() * 100 if cheap.any() else np.nan,
                             usd_1lot_per_print=cd))
    R = pd.DataFrame(rows)
    print("HOLDOUT 2025-26 snipe of decided buckets (certain side, taker, <= 97c), by feed delay x precision")
    print(R.round(2).to_string(index=False))
    out = os.path.join(os.environ.get("RESEARCH_DATA", "research_data"), "results")
    os.makedirs(out, exist_ok=True)
    R.to_csv(f"{out}/kalshi_obs_value_{PERIOD}.csv", index=False)
    pick = R.set_index(["precision", "delay_min"]).pnl_c
    print("::notice::obs value (holdout, c/contract at <=97c): " + "; ".join(
        f"{k}: " + " ".join(f"{d}m {pick.get((k, d), np.nan):+.1f}" for d in (2, 5, 10, 20, 30)) for k in ("exact", "wholeC", "hourly")))


if __name__ == "__main__":
    main()
