"""
research/kalshi/model_lag.py -- does the market lag new forecast-model runs? (Oct 22, 2026)

Earlier tests: public forecasts are already in the price (weather_edge.py). Untested: HOW FAST they get in. If a new
NBM / GFS MOS run moves the forecast high and the market takes tens of minutes to follow, a bot that reads the run
the moment it is published can trade into the lag (the live bot now has sub-second order books).

Data: mos.parquet (GFS MOS and NBM text 'NBS' runs, station daytime max per target date), trades_sample.parquet
(every trade in ~5.9k sampled markets, May 2024 - Oct 2026), markets.parquet (bucket intervals, results).
  1. Forecast revisions: for each model run, the change in the forecast high vs the previous run of the same model
     for the same station and date (|change| >= 1F), for target dates 0-2 days out.
  2. Implied probability change per bucket: P(high in bucket) under Normal(forecast, SD) new minus old.
  3. Response curve: market YES price (last trade, <= 90 min old) at run time + k minutes, k = -60 .. +360. The
     through-origin slope of price change on implied change, by k: where the market has already moved by the time the
     run is public there is no lag to trade.
  4. Trade test: at entry = run time + delay (60..240 min; the text bulletins are public ~1-2h after the run time),
     if |implied change| >= threshold and the market has not yet moved half of it, take the implied side as a taker
     (last trade +/- 1c, plus fee), held to settlement and alternatively sold 2h later (-/+ 1c, plus fee).
     Settings chosen on dev (< 2025-10-01), scored once on holdout. t from per-city-day sums.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
import weather_edge as W

pd.set_option("display.width", 240)
SD = 2.5
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")
KS = [-60, -30, 0, 30, 60, 90, 120, 150, 180, 240, 300, 360]


def fee(p):
    return np.ceil(0.07 * 100 * p * (1 - p) * 100 - 1e-9) / 100 / 100


def prob(f, lo, hi):
    return norm.cdf((hi + 0.5 - f) / SD) - norm.cdf((lo - 0.5 - f) / SD)


class Prices:
    def __init__(self, t):
        t = t.sort_values("t")
        self.d = {k: (g.t.values.astype("datetime64[ns]"), g.p.values) for k, g in t[["ticker", "t", "p"]].groupby("ticker")}

    def at(self, tk, ts, max_age=pd.Timedelta("90min")):
        if tk not in self.d:
            return np.nan
        tt, pp = self.d[tk]
        x = np.datetime64(ts.tz_convert(None).to_datetime64(), "ns")
        i = np.searchsorted(tt, x, side="right") - 1
        if i < 0 or x - tt[i] > np.timedelta64(int(max_age.total_seconds()), "s"):
            return np.nan
        return pp[i]


def revisions(mos):
    mos = W.mos_max(mos)                                     # daytime max rows, 'date' = target local date
    mos = mos.sort_values(["model", "station", "date", "runtime"])
    mos["prev"] = mos.groupby(["model", "station", "date"]).n_x.shift()
    mos["lead_d"] = (mos.date - mos.runtime.dt.tz_localize(None).dt.normalize()).dt.days
    r = mos[mos.prev.notna() & (mos.n_x != mos.prev) & mos.lead_d.between(0, 2)]
    return r[["model", "station", "date", "runtime", "n_x", "prev"]]


def main():
    m = W.market_intervals(pd.read_parquet(W.D / "markets.parquet"))
    t = pd.read_parquet(W.D / "trades_sample.parquet")
    t = t[~t.is_block_trade.fillna(False)]
    m = m[m.ticker.isin(set(t.ticker))]
    P = Prices(t)
    R = revisions(pd.read_parquet(W.D / "mos.parquet"))
    m["date"] = m.date.dt.normalize()
    X = R.merge(m[["ticker", "event_ticker", "station", "date", "lo", "hi", "y", "close_time"]], on=["station", "date"])
    X["dp"] = prob(X.n_x, X.lo, X.hi) - prob(X.prev, X.lo, X.hi)
    X = X[X.dp.abs() >= 0.03].reset_index(drop=True)
    X["period"] = np.where(X.runtime < SPLIT, "dev", "hold")
    print(f"revisions {len(R):,}; bucket-revision pairs with |implied change| >= 3pts and a traded market: {len(X):,}")

    # 3. response curve
    for k in KS:
        X[f"P{k}"] = [P.at(tk, rt + pd.Timedelta(minutes=k)) for tk, rt in zip(X.ticker, X.runtime)]
    rows = []
    for (model, per), g in X.groupby(["model", "period"]):
        base = g.P0 if True else None
        for k in KS:
            ok = g[f"P{k}"].notna() & g.P0.notna() & g.P360.notna()
            d, dp = (g[f"P{k}"] - g.P0)[ok], g.dp[ok]
            dfin = (g.P360 - g.P0)[ok]
            rows.append(dict(model=model, period=per, k_min=k, n=int(ok.sum()), slope=(d * dp).sum() / (dp * dp).sum(),
                             share_of_6h=(d * dp).sum() / (dfin * dp).sum() if (dfin * dp).sum() else np.nan))
    C = pd.DataFrame(rows)
    print("\n3. RESPONSE: market price change since the run time, regressed on the implied change (slope; share of the "
          "6h move)\n" + C.pivot_table(index=["model", "k_min"], columns="period", values=["slope", "share_of_6h", "n"]).round(3).to_string())

    # 4. trade test
    res = []
    for model, gm in X.groupby("model"):
        for delay in (60, 90, 120, 150, 180, 240):
            ent = [rt + pd.Timedelta(minutes=delay) for rt in gm.runtime]
            pe = np.array([P.at(tk, e, pd.Timedelta("30min")) for tk, e in zip(gm.ticker, ent)])
            px = np.array([P.at(tk, e + pd.Timedelta("2h"), pd.Timedelta("60min")) for tk, e in zip(gm.ticker, ent)])
            p0 = gm.P0.values
            for th in (0.05, 0.10, 0.20):
                s = np.sign(gm.dp.values)
                moved = np.where(np.isfinite(p0), (pe - p0) * s / np.abs(gm.dp.values), 0)
                ok = np.isfinite(pe) & (np.abs(gm.dp.values) >= th) & (moved < 0.5) & (pe > 0.03) & (pe < 0.97)
                g = gm[ok].assign(pe=pe[ok], px=px[ok], s=s[ok])
                if len(g) < 20:
                    continue
                cost = np.where(g.s > 0, g.pe + 0.01, 1 - g.pe + 0.01) + fee(g.pe)
                pay = np.where(g.s > 0, g.y, 1 - g.y)
                g = g.assign(e_settle=pay - cost)
                ex = np.where(g.s > 0, g.px - 0.01, 1 - g.px - 0.01) - fee(g.px.fillna(0.5))
                g = g.assign(e_2h=np.where(g.px.notna(), ex - cost, np.nan))
                for per, h in g.groupby("period"):
                    for col in ("e_settle", "e_2h"):
                        hh = h[h[col].notna()]
                        cd = hh.groupby("event_ticker")[col].sum()
                        res.append(dict(model=model, delay=delay, th=th, period=per, exit=col, n=len(hh),
                                        city_days=len(cd), edge_c=hh[col].mean() * 100,
                                        t=cd.mean() / cd.std() * np.sqrt(len(cd)) if len(cd) > 2 and cd.std() > 0 else np.nan))
    T = pd.DataFrame(res)
    print("\n4. TRADE INTO THE LAG (taker, +1c spread + fee), cents per contract\n" +
          T.pivot_table(index=["model", "exit", "delay", "th"], columns="period", values=["edge_c", "t", "n"]).round(2).to_string())
    out = os.path.join(os.environ.get("RESEARCH_DATA", "research_data"), "results")
    os.makedirs(out, exist_ok=True)
    C.to_csv(f"{out}/kalshi_model_lag_response.csv", index=False); T.to_csv(f"{out}/kalshi_model_lag_trades.csv", index=False)
    d = T[T.period == "dev"]
    if len(d):
        b = d.sort_values("t", ascending=False).iloc[0]
        h = T[(T.period == "hold") & (T.model == b.model) & (T.delay == b.delay) & (T.th == b.th) & (T.exit == b.exit)]
        hs = f"holdout {h.edge_c.iloc[0]:+.2f}c t {h.t.iloc[0]:.1f} (n {h.n.iloc[0]})" if len(h) else "no holdout rows"
        sh = C[(C.period == "hold") & (C.k_min == 120)].set_index("model").share_of_6h.round(2).to_dict()
        print(f"::notice::model lag: share of 6h repricing done by run+2h (holdout) {sh}; dev-best trade {b.model} "
              f"delay {b.delay} th {b.th} {b.exit}: dev {b.edge_c:+.2f}c t {b.t:.1f} -> {hs}")


if __name__ == "__main__":
    main()
