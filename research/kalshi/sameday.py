"""
research/kalshi/sameday.py -- same-day Kalshi daily-high markets with live station readings.

At 12:00, 14:00 and 16:00 local time on the day: M = highest ASOS reading since local
midnight (public in real time), F = the morning NBM-text forecast high (as in
weather_edge.py). Model: final high = max(M, X), X ~ Normal(a + b*F + c*M, sigma_hour),
fitted per decision hour on dev events (interval-censored likelihood). Bucket
probability -> compare with the last hourly candle's bid/ask before the decision;
taker fee as in weather_edge.py. Dev = before 2025, holdout = 2025-2026 (scored once).
Kalshi's day is local STANDARD time; ASOS readings are rounded whole degrees F.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
import weather_edge as W

D = W.D
HOURS = (12, 14, 16)


def max_so_far(ev, asos):
    out = {h: np.full(len(ev), np.nan) for h in HOURS}
    for st, g in ev.groupby("station"):
        a = asos[asos.station == st]
        if a.empty:
            continue
        tz = W.TZ[st]
        # local STANDARD time: subtract DST hour when in effect
        loc = a.valid_utc.dt.tz_convert(tz)
        dst = loc.map(lambda x: x.dst().total_seconds() if x.dst() else 0.0)
        lst = loc.dt.tz_localize(None) - pd.to_timedelta(dst, unit="s")
        s = pd.Series(a.tmpf.values, index=lst.values).sort_index()
        for idx, d in zip(g.index, g.date):
            day = s[(s.index >= d) & (s.index < d + pd.Timedelta(days=1))]
            for h in HOURS:
                part = day[day.index < d + pd.Timedelta(hours=h)]
                if len(part) >= 6:
                    out[h][ev.index.get_loc(idx)] = np.round(part.max())
    for h in HOURS:
        ev[f"M{h}"] = out[h]
    return ev


def prob_interval(M, mu, sd, lo, hi):
    """P(max(M, X) in [lo-0.5, hi+0.5]) with X ~ N(mu, sd)."""
    L, H = lo - 0.5, hi + 0.5
    pX = norm.cdf((H - mu) / sd) - norm.cdf((L - mu) / sd)
    pAtMost = norm.cdf((H - mu) / sd)
    return np.where(M > H, 0.0, np.where(M >= L, pAtMost, pX))


def fit(e, h):
    F, M = e.F.values, e[f"M{h}"].values
    lo, hi = e.rlo.values, e.rhi.values

    def nll(w):
        mu = w[0] + w[1] * F + w[2] * M
        sd = np.exp(w[3])
        p = prob_interval(M, mu, sd, lo, hi)
        return -np.log(np.clip(p, 1e-9, 1)).sum()
    return minimize(nll, np.r_[0, 0.5, 0.5, np.log(2.0)], method="L-BFGS-B").x


def main():
    m = W.market_intervals(pd.read_parquet(D / "markets.parquet"))
    cd = pd.read_parquet(D / "candles.parquet")
    for c in ("bid", "ask"):
        cd[c] = cd[c].astype(float)
    asos = pd.read_parquet(D / os.environ.get("ASOS_FILE", "asos.parquet"))   # asos1min.parquet: 1-min data (audit redo)
    asos["station"] = asos.station.where(asos.station.str.len() == 4, "K" + asos.station)
    mos = W.mos_max(pd.read_parquet(D / "mos.parquet"))
    ev = m.groupby("event_ticker").agg(station=("station", "first"), date=("date", "first")).reset_index()
    ev = ev.merge(W.realised(m), on="event_ticker")
    ev = W.forecasts(ev, mos)
    ev["F"] = ev.f_morn_NBS.fillna(ev.f_morn_GFS)
    ev = max_so_far(ev.reset_index(drop=True), asos)
    print(f"events {len(ev):,}; with M at 14h: {ev.M14.notna().sum():,}")
    rows = []
    for h in HOURS:
        e = ev.dropna(subset=["F", f"M{h}"])
        e = e[np.isfinite(e.rlo) | np.isfinite(e.rhi)]
        fitted = []
        for st, g in e.groupby("station"):
            dev = g[g.date < W.SPLIT]
            if len(dev) < 150:
                continue
            w = fit(dev, h)
            fitted.append(g[["event_ticker"]].assign(mu=w[0] + w[1] * g.F + w[2] * g[f"M{h}"], sd=np.exp(w[3]),
                                                       M=g[f"M{h}"]))
        if not fitted:
            continue
        fe = pd.concat(fitted)
        mk = m.merge(fe, on="event_ticker")
        mk = mk.merge(ev[["event_ticker"]], on="event_ticker")
        # decision time in UTC (local standard time h:00)
        t = []
        for st, d in zip(mk.station, mk.date):
            loc = (pd.Timestamp(d) + pd.Timedelta(hours=h)).tz_localize(W.TZ[st])
            t.append((loc + (loc.dst() or pd.Timedelta(0))).tz_convert("UTC"))
        slot = f"h{h}"
        mk[f"t_{slot}"] = t
        mk = mk.merge(W.price_at(mk, cd, slot), on="ticker", how="left")
        mk = mk[(mk.open_time <= mk[f"t_{slot}"]) & (mk.close_time > mk[f"t_{slot}"])]
        mk["mid"] = (mk.bid + mk.ask) / 2
        mk["p_model"] = prob_interval(mk.M.values, mk.mu.values, mk.sd.values, mk.lo.values, mk.hi.values)
        mk = mk.dropna(subset=["mid", "p_model"])
        mk = mk[(mk.ask - mk.bid) <= 0.10]
        dv = mk.date < W.SPLIT
        print(f"\n=== {h}:00 local: {len(mk):,} quoted markets ({dv.sum():,} dev / {(~dv).sum():,} holdout)")
        for nm, part in (("DEV", mk[dv]), ("HOLDOUT", mk[~dv])):
            print(f"  {nm}: logloss market {W.logloss(part.mid, part.y):.4f} model {W.logloss(part.p_model, part.y):.4f}")
        for theta in (0.0, 0.03, 0.06, 0.10):
            for nm, part in (("DEV", mk[dv]), ("HOLDOUT", mk[~dv])):
                rows.append({"hour": h, "theta": theta, "split": nm, **W.trades(part, part.p_model.values, theta)})
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
