"""
research/kalshi/weather_edge.py
===============================
Does a public-forecast model beat Kalshi daily-high temperature markets after fees?

Inputs (research_data/kalshi/): markets.parquet, candles.parquet, mos.parquet
(see fetch_kalshi.py).

Steps
  1. Each market -> integer interval [lo, hi] of daily highs that pays YES.
     Each event -> realised interval of the high (from which markets paid).
  2. Forecast at two decision times, using only model runs already published:
       eve  = 21:00 local the day before,   morn = 09:00 local on the day
     GFS MOS and NBM text (NBS) daytime max for the target date.
  3. Model: high ~ Normal(a + b*forecast, sigma(day-of-year)), fitted per
     station x slot by interval-censored maximum likelihood on DEV events only.
  4. Market price at the decision time = last hourly candle's yes bid/ask.
  5. Score: calibration of the market, log loss market vs model vs blend, and a
     taker trading sim after Kalshi's fee (0.07 * P * (1-P) per contract,
     rounded up to the cent per order of 100 contracts).
DEV = events before 2025-01-01, HOLDOUT = 2025-01-01 on (scored once).
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

D = Path(os.environ.get("KDATA", "research_data/kalshi"))
SPLIT = pd.Timestamp("2025-01-01")
TZ = {"KNYC": "America/New_York", "KPHL": "America/New_York", "KDCA": "America/New_York",
      "KBOS": "America/New_York", "KEWR": "America/New_York", "KMIA": "America/New_York",
      "KMDW": "America/Chicago", "KAUS": "America/Chicago", "KHOU": "America/Chicago",
      "KDFW": "America/Chicago", "KSAT": "America/Chicago", "KDEN": "America/Denver",
      "KPHX": "America/Phoenix", "KLAX": "America/Los_Angeles", "KSAN": "America/Los_Angeles"}
SLOTS = {"eve": (-1, 21), "morn": (0, 9)}          # (day offset, local hour)
AVAIL = {"GFS": pd.Timedelta("4h30min"), "NBS": pd.Timedelta("2h")}   # run -> public
ORDER = 100                                          # contracts per order for fee rounding


def fee(p):
    """Kalshi taker fee per contract for an order of ORDER contracts at price p."""
    return np.ceil(0.07 * ORDER * p * (1 - p) * 100 - 1e-9) / 100 / ORDER


# ---------------------------------------------------------------- 1. intervals
def market_intervals(m):
    m = m.copy()
    t = m.title.fillna("")
    gt = t.str.extract(r"(?:>|over|above)\s*(-?\d+)")[0].astype(float)
    lt = t.str.extract(r"(?:<|under|below)\s*(-?\d+)")[0].astype(float)
    bw = t.str.extract(r"be\s+(-?\d+)\s*-\s*(-?\d+)\s*°")
    lo = np.full(len(m), np.nan); hi = np.full(len(m), np.nan)
    st = m.strike_type.values
    fl, cp = m.floor_strike.values, m.cap_strike.values
    for i in range(len(m)):
        if st[i] == "greater":
            lo[i], hi[i] = fl[i] + 1, np.inf
        elif st[i] == "less":
            lo[i], hi[i] = -np.inf, cp[i] - 1
        elif st[i] == "between":
            lo[i], hi[i] = fl[i], cp[i]
        elif not np.isnan(gt.iat[i]):
            lo[i], hi[i] = gt.iat[i] + 1, np.inf
        elif not np.isnan(lt.iat[i]):
            lo[i], hi[i] = -np.inf, lt.iat[i] - 1
        elif isinstance(bw.iat[i, 0], str):
            lo[i], hi[i] = float(bw.iat[i, 0]), float(bw.iat[i, 1])
    m["lo"], m["hi"] = lo, hi
    m["y"] = (m.result == "yes").astype(float)
    m["date"] = pd.to_datetime(m.event_ticker.str.extract(r"-(\d{2}[A-Z]{3}\d{2})$")[0],
                               format="%y%b%d", errors="coerce")
    bad = m.lo.isna() | m.date.isna() | ~m.result.isin(["yes", "no"])
    print(f"markets: {len(m):,}, unparsed/void dropped: {bad.sum():,}")
    return m[~bad]


def realised(m):
    """Interval [rlo, rhi] for each event's actual high, from all settled markets."""
    out = []
    for ev, g in m.groupby("event_ticker"):
        rlo, rhi = -np.inf, np.inf
        for r in g.itertuples(index=False):
            if r.y == 1:
                rlo, rhi = max(rlo, r.lo), min(rhi, r.hi)
            elif np.isinf(r.lo):          # "less than" paid NO -> high >= hi+1
                rlo = max(rlo, r.hi + 1)
            elif np.isinf(r.hi):          # "greater" paid NO  -> high <= lo-1
                rhi = min(rhi, r.lo - 1)
        out.append((ev, rlo, rhi))
    e = pd.DataFrame(out, columns=["event_ticker", "rlo", "rhi"])
    ok = e.rlo <= e.rhi
    print(f"events: {len(e):,}, inconsistent dropped: {(~ok).sum():,}")
    return e[ok]


# ------------------------------------------------------------- 2. forecasts
def mos_max(mos):
    mos = mos.copy()
    h = mos.ftime.dt.hour
    mos = mos[(h >= 18) | (h <= 6)]                              # daytime-max rows
    mos["date"] = (mos.ftime - pd.Timedelta("12h")).dt.tz_localize(None).dt.normalize()
    mos["avail"] = mos.runtime + mos.model.map(AVAIL)
    return mos.sort_values("avail")


def forecasts(ev, mos):
    """ev: event_ticker, station, date. Adds f_<slot>_<model> and decision times."""
    ev = ev.copy()
    for slot, (off, hr) in SLOTS.items():
        dec = [(pd.Timestamp(d) + pd.Timedelta(days=off, hours=hr)).tz_localize(TZ[s]).tz_convert("UTC")
               for d, s in zip(ev.date, ev.station)]
        ev[f"t_{slot}"] = dec
        for model in ("GFS", "NBS"):
            mm = mos[mos.model == model][["station", "date", "avail", "n_x"]]
            left = ev[["event_ticker", "station", "date", f"t_{slot}"]].rename(columns={f"t_{slot}": "avail"})
            left = left.sort_values("avail")
            j = pd.merge_asof(left, mm, on="avail", by=["station", "date"], direction="backward")
            ev = ev.merge(j[["event_ticker", "n_x"]].rename(columns={"n_x": f"f_{slot}_{model}"}),
                          on="event_ticker", how="left")
    return ev


# ------------------------------------------------------------------ 3. model
def design(df, slot):
    g, n = df[f"f_{slot}_GFS"], df[f"f_{slot}_NBS"]
    has_n = n.notna().astype(float)
    f = n.fillna(g)
    doy = df.date.dt.dayofyear.values * 2 * np.pi / 365.25
    X = np.column_stack([np.ones(len(df)), f, g.fillna(f) - f, has_n])
    S = np.column_stack([np.ones(len(df)), np.cos(doy), np.sin(doy)])
    return X, S


def fit(df, slot):
    X, S = design(df, slot)
    lo, hi = df.rlo.values - 0.5, df.rhi.values + 0.5

    def nll(w):
        mu = X @ w[:4]
        sd = np.exp(S @ w[4:])
        p = norm.cdf((hi - mu) / sd) - norm.cdf((lo - mu) / sd)
        return -np.log(np.clip(p, 1e-9, 1)).sum()

    w0 = np.r_[0, 1, 0, 0, np.log(3), 0, 0]
    return minimize(nll, w0, method="L-BFGS-B").x


def predict(w, df, slot):
    X, S = design(df, slot)
    return X @ w[:4], np.exp(S @ w[4:])


# --------------------------------------------------------------- 4. prices
def price_at(mk, cd, slot):
    q = mk[["ticker", f"t_{slot}"]].rename(columns={f"t_{slot}": "ts"})
    q["ts"] = pd.to_datetime(q.ts, utc=True).astype("datetime64[ns, UTC]")
    q = q.drop_duplicates("ticker").sort_values("ts")
    c = cd.dropna(subset=["bid", "ask"], how="all").copy()
    c["ts"] = c.ts.astype("datetime64[ns, UTC]")
    c = c.sort_values("ts")
    j = pd.merge_asof(q, c[["ticker", "ts", "bid", "ask", "volume"]].assign(cts=c.ts),
                      on="ts", by="ticker", direction="backward", tolerance=pd.Timedelta("12h"))
    # day volume up to decision, as a liquidity gauge
    return j.drop(columns="ts")


# --------------------------------------------------------------- 5. scoring
def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def trades(df, p, theta):
    ask, bid, y = df.ask.values, df.bid.values, df.y.values
    ey = p - ask - fee(ask)
    en = (1 - p) - (1 - bid) - fee(1 - bid)
    ey = np.where((ask > 0) & (ask < 1), ey, -1)
    en = np.where((bid > 0) & (bid < 1), en, -1)
    buy_y = (ey >= en) & (ey > theta)
    buy_n = (en > ey) & (en > theta)
    pnl = np.where(buy_y, y - ask - fee(ask), 0.0) + np.where(buy_n, (1 - y) - (1 - bid) - fee(1 - bid), 0.0)
    cost = np.where(buy_y, ask + fee(ask), 0.0) + np.where(buy_n, 1 - bid + fee(1 - bid), 0.0)
    t = df.assign(pnl=pnl, cost=cost, n=(buy_y | buy_n).astype(int))
    t = t[t.n == 1]
    if not len(t):
        return dict(n=0)
    day = t.groupby("date").pnl.sum()
    tstat = day.mean() / (day.std(ddof=1) / np.sqrt(len(day))) if len(day) > 2 else np.nan
    return dict(n=len(t), days=len(day), pnl_per=t.pnl.mean(), roi=t.pnl.sum() / t.cost.sum(),
                win=(t.pnl > 0).mean(), t=tstat, per_day=day.mean())


def main():
    m = pd.read_parquet(D / "markets.parquet")
    cd = pd.read_parquet(D / "candles.parquet")
    mos = mos_max(pd.read_parquet(D / "mos.parquet"))
    for c in ("bid", "ask"):
        cd[c] = cd[c].astype(float)
    m = market_intervals(m)
    ev = m.groupby("event_ticker").agg(station=("station", "first"), series=("series", "first"),
                                       date=("date", "first")).reset_index()
    ev = ev.merge(realised(m), on="event_ticker")
    ev = forecasts(ev, mos)
    m = m.merge(ev.drop(columns=["station", "series", "date"]), on="event_ticker")
    print(f"events with forecasts: eve {ev.f_eve_GFS.notna().sum():,} / morn {ev.f_morn_GFS.notna().sum():,} "
          f"of {len(ev):,}; NBS share {ev.f_morn_NBS.notna().mean():.0%}")

    res = []
    for slot in SLOTS:
        e = ev.dropna(subset=[f"f_{slot}_GFS"])
        e = e[np.isfinite(e.rlo) | np.isfinite(e.rhi)]
        mk = m[m.event_ticker.isin(e.event_ticker)].copy()
        mk = mk.merge(price_at(mk, cd, slot), on="ticker", how="left")
        mk = mk[(mk.open_time <= mk[f"t_{slot}"]) & (mk.close_time > mk[f"t_{slot}"])]
        mk["mid"] = np.where(mk.bid.notna() & mk.ask.notna(), (mk.bid + mk.ask) / 2, np.nan)
        fitted = []
        for st, es in e.groupby("station"):
            dev = es[es.date < SPLIT]
            if len(dev) < 150:
                print(f"  {slot} {st}: only {len(dev)} dev events, skipped")
                continue
            w = fit(dev, slot)
            mu, sd = predict(w, es, slot)
            fitted.append(es[["event_ticker"]].assign(mu=mu, sd=sd))
            print(f"  {slot} {st}: dev events {len(dev):,}, a={w[0]:+.2f} b={w[1]:.3f} "
                  f"sigma~{np.exp(w[4]):.2f}F")
        mk = mk.merge(pd.concat(fitted), on="event_ticker")
        mk["p_model"] = (norm.cdf((mk.hi + 0.5 - mk.mu) / mk.sd) - norm.cdf((mk.lo - 0.5 - mk.mu) / mk.sd))
        mk = mk.dropna(subset=["p_model", "mid"])
        mk = mk[(mk.ask - mk.bid) <= 0.10]                   # quoted, reasonably tight
        # blend: logistic on logit(mid), logit(model), fitted on dev
        lg = lambda x: np.log(np.clip(x, 1e-3, 1 - 1e-3) / (1 - np.clip(x, 1e-3, 1 - 1e-3)))
        Z = np.column_stack([np.ones(len(mk)), lg(mk.mid), lg(mk.p_model)])
        dv = (mk.date < SPLIT).values
        yy = mk.y.values
        b = minimize(lambda b: -(yy[dv] * (Z[dv] @ b) - np.log1p(np.exp(Z[dv] @ b))).sum(),
                     np.r_[0, 0.5, 0.5]).x
        mk["p_blend"] = 1 / (1 + np.exp(-(Z @ b)))
        print(f"\n=== {slot}: {len(mk):,} quoted markets ({dv.sum():,} dev / {(~dv).sum():,} holdout); "
              f"blend weights mkt {b[1]:.2f} model {b[2]:.2f}")
        for name, part in (("DEV", mk[dv]), ("HOLDOUT", mk[~dv])):
            print(f"  {name}: logloss market {logloss(part.mid, part.y):.4f}  model {logloss(part.p_model, part.y):.4f}"
                  f"  blend {logloss(part.p_blend, part.y):.4f}   brier mkt {((part.mid - part.y) ** 2).mean():.4f}"
                  f" model {((part.p_model - part.y) ** 2).mean():.4f} blend {((part.p_blend - part.y) ** 2).mean():.4f}")
        cal = mk[dv].assign(bin=pd.cut(mk[dv].mid, [0, .05, .15, .3, .5, .7, .85, .95, 1]))
        print("  market calibration (dev): " + "  ".join(
            f"{iv.left:.2f}-{iv.right:.2f}: {g.mid.mean():.3f}->{g.y.mean():.3f} (n={len(g)})"
            for iv, g in cal.groupby("bin", observed=True)))
        for pcol in ("p_model", "p_blend"):
            for theta in (0.0, 0.02, 0.05, 0.10):
                for name, part in (("DEV", mk[dv]), ("HOLDOUT", mk[~dv])):
                    r = trades(part, part[pcol].values, theta)
                    res.append(dict(slot=slot, prob=pcol, theta=theta, split=name, **r))
        mk.to_parquet(D / f"scored_{slot}.parquet", index=False)
    r = pd.DataFrame(res)
    pd.set_option("display.width", 200)
    print("\n" + r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    r.to_csv(D / "edge_results.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
