"""
research/kalshi/obs_decided.py -- trading after the station has already decided a bucket (Oct 19, 2026).

A daily-high bucket is DECIDED early when the high so far already settles it:
  certain NO  : high so far >= bucket top + 1 + margin       ("between" and "less than" buckets)
  certain YES : high so far >= threshold + margin            ("greater than" buckets)
The high so far = rounded max of ASOS 5-minute averages (the reporting basis of the official daily max) since local
STANDARD-time midnight, using only readings at least LAG minutes old at the trade (real-time feeds lag).
Two questions on 3M weather trades (kalshi/trades_sample.parquet, 14 stations, 2021-2026):
  A  DEFENSE: how much maker-side volume trades in decided markets, and what does the maker earn there vs elsewhere?
     (maker side of every trade held to settlement, per contract and per market)
  B  SNIPING: buying the certain side as a taker at the traded price: profit = 1 - q - taker fee if right, -q - fee
     if wrong; how often is "certain" wrong; how many contracts per city-day were available, at what prices.
Dev = before 2025, holdout = 2025-2026. Margin 0 / 1 F; lag 10 / 30 minutes.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import weather_edge as W

D = W.D


def obs_cummax():
    a = pd.read_parquet(D / os.environ.get("OBS_FILE", "asos5avg.parquet"))
    a["station"] = a.station.where(a.station.str.len() == 4, "K" + a.station)
    out = []
    for st, g in a.groupby("station"):
        if st not in W.TZ:
            continue
        utc_naive = g.valid_utc.dt.tz_localize(None)
        off = g.valid_utc.dt.tz_convert(W.TZ[st]).dt.tz_localize(None) - utc_naive
        lst = utc_naive + off.min()                              # local STANDARD time (no DST)
        g = g.assign(day=lst.dt.normalize().values).sort_values("valid_utc")
        g["cmax"] = g.groupby("day").tmpf.cummax().round()
        out.append(g[["station", "valid_utc", "day", "cmax"]])
    return pd.concat(out, ignore_index=True).sort_values("valid_utc")


def main():
    m = W.market_intervals(pd.read_parquet(D / "markets.parquet"))
    t = pd.read_parquet(D / "trades_sample.parquet")
    t = t[(t.p > 0) & (t.p < 1) & ~t.is_block_trade.fillna(False)]
    t = t.merge(m[["ticker", "event_ticker", "station", "date", "lo", "hi", "y", "strike_type"]], on="ticker")
    t["t"] = pd.to_datetime(t.t, utc=True)
    obs = obs_cummax()
    pd.set_option("display.width", 220)
    rows_a, rows_b = [], []
    for lag in (10, 30):
        x = t.assign(tq=t.t - pd.Timedelta(minutes=lag)).sort_values("tq")
        x = pd.merge_asof(x, obs.rename(columns={"valid_utc": "tq"}), on="tq", by="station", direction="backward",
                          tolerance=pd.Timedelta("3h"))
        x = x[x.day == x.date]                                    # reading from the market's own (standard-time) day
        x["period"] = np.where(x.date < W.SPLIT, "dev", "hold")
        x["maker"] = np.where(x.taker_side == "yes", x.p - x.y, x.y - x.p)
        for margin in (0, 1):
            dead = np.isfinite(x.hi) & (x.cmax >= x.hi + 1 + margin)                 # certain NO
            sure = ~np.isfinite(x.hi) & np.isfinite(x.lo) & (x.cmax >= x.lo + margin) # certain YES ("greater")
            x["state"] = np.where(dead, "certain NO", np.where(sure, "certain YES", "open"))
            # A: maker economics by state
            for (per, stt), g in x.groupby(["period", "state"]):
                pm = (g.maker * g.c).groupby(g.ticker).sum() / g.c.groupby(g.ticker).sum()
                rows_a.append(dict(lag=lag, margin=margin, period=per, state=stt, trades=len(g), contracts_k=g.c.sum() / 1e3,
                                   maker_c=(g.maker * g.c).sum() / g.c.sum() * 100, maker_per_mkt_c=pm.mean() * 100,
                                   maker_on_losing_side_pct=((g.maker < 0) * g.c).sum() / g.c.sum() * 100))
            # B: snipe the certain side as a taker at the traded price
            for stt, side_yes in (("certain NO", False), ("certain YES", True)):
                g = x[x.state == stt]
                q = g.p if side_yes else 1 - g.p                       # price of the certain side
                win = (g.y == 1) if side_yes else (g.y == 0)
                fee = W.fee(q.values)
                pnl = np.where(win, 1 - q, -q) - fee
                for per in ("dev", "hold"):
                    k = (g.period == per).values
                    if k.sum() == 0:
                        continue
                    gg = g[k]; qq = q[k]; pp = pnl[k]; ww = win[k]
                    cheap = (qq <= 0.97).values
                    cd = gg[cheap].groupby("event_ticker").c.sum()
                    rows_b.append(dict(lag=lag, margin=margin, side=stt, period=per, trades=int(k.sum()),
                                       wrong_pct=(1 - ww.mean()) * 100, wrong_contracts_pct=((~ww) * gg.c).sum() / gg.c.sum() * 100,
                                       pnl_c_all=(pp * gg.c).sum() / gg.c.sum() * 100,
                                       trades_at_le97c=int(cheap.sum()),
                                       pnl_c_le97=(pp[cheap] * gg.c[cheap]).sum() / gg.c[cheap].sum() * 100 if cheap.any() else np.nan,
                                       city_days_with_le97=len(cd), med_contracts_le97_per_cityday=cd.median() if len(cd) else np.nan,
                                       mean_price_le97=(qq[cheap] * gg.c[cheap]).sum() / gg.c[cheap].sum() * 100 if cheap.any() else np.nan))
    A = pd.DataFrame(rows_a); B = pd.DataFrame(rows_b)
    os.makedirs("research_data/results", exist_ok=True)
    print("A. MAKER SIDE BY STATE (cents per contract to settlement)")
    print(A.round(2).to_string(index=False))
    print("\nB. SNIPING THE CERTAIN SIDE AS A TAKER (at traded prices, after taker fee)")
    print(B.round(2).to_string(index=False))
    A.to_csv(D.parent / "results" / "kalshi_obs_decided_A.csv", index=False)
    B.to_csv(D.parent / "results" / "kalshi_obs_decided_B.csv", index=False)


if __name__ == "__main__":
    main()
