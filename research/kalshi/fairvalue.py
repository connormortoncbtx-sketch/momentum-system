"""
research/kalshi/fairvalue.py -- Phase 2 of the fair-value project (Nov 10, 2026): is the HOURLY gridded NBM a sharper
(or additional) source of truth than Kalshi's own price, and does it tell the bot which favourites not to buy?

Data (public): kalshi/nbm_tmax.parquet (every hourly NBM run: daytime-max mean and ensemble sd per station; built by
nbm_archive.py), kalshi/markets.parquet (bucket strikes + results), kalshi/candles.parquet (hourly bid/ask per market).
Overlap: Oct 2024 - Oct 2026, the series with both (NY, CHI, LAX, MIA, PHIL, DEN, AUS, DAL, BOS, PHX).

At each market-hour with a two-sided book, 12-36h before the end of the daytime window (D+1 00Z): the latest NBM run
public by then (run + 75 min). Model: official high Y = round(X), X ~ Normal(mean + bias[station], k*sd (+) c), bucket
probability from the strike type. bias/k/c fitted on DEV (< 2025-10-01) by log-loss on the outcomes; everything
judged on the HOLDOUT (>= 2025-10-01), standard errors by bootstrap over market days.
Questions:
 1. SHARPNESS: log-loss and Brier of market mid vs model, by hours-to-window-end.
 2. ADDED INFORMATION: logistic blend logit(p) = a*logit(mid) + b*logit(model) + c fitted on DEV; does it beat the
    mid on the HOLDOUT? (b > 0 and a lower holdout log-loss = the model knows something the price doesn't)
 3. THE BOT'S USE: among favourite prices the bot would hold (side >= 60c), settlement edge when the model AGREES
    (model prob >= mid) vs DISAGREES by > 5 points -- a filter "don't buy favourites the model says are rich".
 4. TAKER test: buy at the ask / sell at the bid when the model differs from the book by > 5/10 points, after fees.
Output: results/kalshi_fairvalue.csv + printed tables.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")
MON = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)


def mday(ev):
    d = ev.split("-")[-1]
    if len(d) != 7 or d[2:5] not in MON:
        return pd.NaT
    return pd.Timestamp(2000 + int(d[:2]), MON[d[2:5]], int(d[5:7]))


def load():
    nbm = pd.read_parquet(f"{RD}/kalshi/nbm_tmax.parquet")
    m = pd.read_parquet(f"{RD}/kalshi/markets.parquet")
    m = m[m.series.isin(nbm.series.unique()) & m.result.isin(["yes", "no"])].copy()
    m["day"] = m.event_ticker.map(mday)
    m = m.dropna(subset=["day"])
    m = m[m.day >= pd.Timestamp(nbm.day.min())]
    c = pd.read_parquet(f"{RD}/kalshi/candles.parquet", columns=["ticker", "ts", "bid", "ask"])
    c = c[c.ticker.isin(m.ticker)].dropna(subset=["bid", "ask"])
    c = c[(c.ask > c.bid) & (c.ask - c.bid <= 0.10) & (c.bid > 0) & (c.ask < 1)]
    c = c.merge(m[["ticker", "series", "station", "strike_type", "floor_strike", "cap_strike", "result", "day", "event_ticker"]], on="ticker")
    c["wend"] = (c.day + pd.Timedelta(days=1)).dt.tz_localize("UTC")
    c["lead"] = (c.wend - c.ts).dt.total_seconds() / 3600
    c = c[(c.lead > 12) & (c.lead <= 36)].copy()
    c["mid"] = (c.bid + c.ask) / 2
    c["R"] = (c.result == "yes").astype(float)
    # latest NBM run public by ts (run + 75 min), same series and target day
    nbm["avail"] = pd.to_datetime(nbm.run, utc=True) + pd.Timedelta(minutes=75)
    nbm["day"] = pd.to_datetime(nbm.day)
    nbm = nbm.sort_values("avail")
    c = c.sort_values("ts")
    out = []
    for (s, d), g in c.groupby(["series", "day"]):
        n = nbm[(nbm.series == s) & (nbm.day == d)]
        if n.empty:
            continue
        out.append(pd.merge_asof(g, n[["avail", "mean_f", "sd_f"]], left_on="ts", right_on="avail"))
    c = pd.concat(out, ignore_index=True).dropna(subset=["mean_f"])
    return c


def bucket_p(df, mu, sig):
    lo = np.where(df.strike_type == "between", df.floor_strike - 0.5,
                  np.where(df.strike_type == "greater", df.floor_strike + 0.5, -np.inf))
    hi = np.where(df.strike_type == "between", df.cap_strike + 0.5,
                  np.where(df.strike_type == "less", df.cap_strike - 0.5, np.inf))
    return np.clip(norm.cdf((hi - mu) / sig) - norm.cdf((lo - mu) / sig), 1e-4, 1 - 1e-4)


def model_p(df, params, stations):
    bias = dict(zip(stations, params[:len(stations)]))
    k, cc = np.exp(params[-2]), np.exp(params[-1])
    mu = df.mean_f.values + df.station.map(bias).values
    sig = np.sqrt((k * df.sd_f.fillna(df.sd_f.median()).values) ** 2 + cc ** 2)
    return bucket_p(df, mu, sig)


def ll(p, R):
    return -(R * np.log(p) + (1 - R) * np.log(1 - p))


def boot_diff(df, a, b, n=400, seed=0):
    """mean(a - b) per row and a 95% CI by bootstrapping market days."""
    rng = np.random.default_rng(seed)
    g = df.assign(d=a - b).groupby("day").d.agg(["sum", "count"])
    days = g.index.values; S, C = g["sum"].values, g["count"].values
    bs = [S[i].sum() / C[i].sum() for i in (rng.integers(0, len(days), len(days)) for _ in range(n))]
    return S.sum() / C.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    c = load()
    stations = sorted(c.station.unique())
    c["part"] = np.where(c.ts < SPLIT, "dev", "hold")
    dev, hold = c[c.part == "dev"], c[c.part == "hold"]
    print(f"{len(c):,} market-hours ({len(dev):,} dev / {len(hold):,} holdout), {c.event_ticker.nunique():,} city-days, "
          f"series {sorted(c.series.unique())}", flush=True)
    fit = dev.groupby(["ticker", (dev.lead // 6)]).head(1)          # thin to ~1 row per market per 6h of lead
    x0 = np.r_[np.zeros(len(stations)), 0.0, 0.0]
    res = minimize(lambda p: ll(model_p(fit, p, stations), fit.R.values).mean(), x0, method="L-BFGS-B")
    params = res.x
    print("fit: bias " + ", ".join(f"{s} {b:+.2f}F" for s, b in zip(stations, params[:-2]))
          + f" | k {np.exp(params[-2]):.2f} c {np.exp(params[-1]):.2f}F", flush=True)
    c["model"] = model_p(c, params, stations)
    c["mid_c"] = c.mid.clip(0.01, 0.99)
    # blend on DEV
    from sklearn.linear_model import LogisticRegression
    X = lambda d: np.c_[logit(d.mid_c), logit(d.model)]
    lr = LogisticRegression(C=1e6).fit(X(c[c.part == "dev"]), c[c.part == "dev"].R)
    c["blend"] = lr.predict_proba(X(c))[:, 1]
    print(f"blend on dev: logit(p) = {lr.coef_[0][0]:.2f}*logit(mid) + {lr.coef_[0][1]:.2f}*logit(model) + {lr.intercept_[0]:+.2f}")
    rows = []
    c["leadband"] = pd.cut(c.lead, [12, 18, 24, 30, 36], labels=["12-18h", "18-24h", "24-30h", "30-36h"])
    for part in ("dev", "hold"):
        for lb, g in [("ALL", c[c.part == part])] + [(str(k), v) for k, v in c[c.part == part].groupby("leadband", observed=True)]:
            Lm, Lmod, Lb = ll(g.mid_c.values, g.R.values), ll(g.model.values, g.R.values), ll(g.blend.values, g.R.values)
            d1 = boot_diff(g, Lmod, Lm); d2 = boot_diff(g, Lb, Lm)
            rows.append(dict(part=part, lead=lb, n=len(g), days=g.day.nunique(), ll_market=Lm.mean(), ll_model=Lmod.mean(),
                             ll_blend=Lb.mean(), model_minus_mkt=d1[0], ci_lo=d1[1], ci_hi=d1[2],
                             blend_minus_mkt=d2[0], blend_ci_lo=d2[1], blend_ci_hi=d2[2],
                             brier_mkt=((g.mid_c - g.R) ** 2).mean(), brier_model=((g.model - g.R) ** 2).mean()))
    T = pd.DataFrame(rows)
    print("\n1-2. LOG-LOSS (lower is better); model/blend minus market with 95% CI by market-day bootstrap "
          "(negative = beats the market):\n" + T.round(4).to_string(index=False))
    # 3. the bot's use: favourites the bot would hold
    h = c[c.part == "hold"].copy()
    h["fav_yes"] = h.mid >= 0.6
    h["fav_no"] = h.mid <= 0.4
    fav = pd.concat([h[h.fav_yes].assign(held_px=lambda d: d.mid, held_model=lambda d: d.model, win=lambda d: d.R),
                     h[h.fav_no].assign(held_px=lambda d: 1 - d.mid, held_model=lambda d: 1 - d.model, win=lambda d: 1 - d.R)])
    fav["gap"] = fav.held_model - fav.held_px
    fav["edge_c"] = (fav.win - fav.held_px) * 100
    out3 = []
    for lab, g in (("all favourites", fav), ("model agrees (gap >= 0)", fav[fav.gap >= 0]),
                   ("model mildly disagrees (-5..0 pts)", fav[(fav.gap < 0) & (fav.gap >= -0.05)]),
                   ("model disagrees by > 5 pts", fav[fav.gap < -0.05]), ("model disagrees by > 10 pts", fav[fav.gap < -0.10])):
        if len(g):
            m_, lo, hi = boot_diff(g, g.edge_c.values, np.zeros(len(g)))
            out3.append(dict(group=lab, rows=len(g), days=g.day.nunique(), edge_at_mid_c=m_, ci_lo=lo, ci_hi=hi))
    F3 = pd.DataFrame(out3)
    print("\n3. FAVOURITES (side >= 60c), HOLDOUT: settlement edge at the mid, by what the model says:\n" + F3.round(2).to_string(index=False))
    # 4. taker test
    out4 = []
    for thr in (0.05, 0.10):
        buy = h[h.model > h.ask + thr]; sell = h[h.model < h.bid - thr]
        fee = lambda p: 0.07 * p * (1 - p) * 100
        pnl = pd.concat([pd.DataFrame(dict(day=buy.day, v=(buy.R - buy.ask) * 100 - fee(buy.ask))),
                         pd.DataFrame(dict(day=sell.day, v=(sell.bid - sell.R) * 100 - fee(sell.bid)))])
        if len(pnl):
            m_, lo, hi = boot_diff(pnl, pnl.v.values, np.zeros(len(pnl)))
            out4.append(dict(threshold=thr, trades=len(pnl), days=pnl.day.nunique(), c_per_contract=m_, ci_lo=lo, ci_hi=hi))
    F4 = pd.DataFrame(out4)
    print("\n4. TAKER on model-vs-book gaps, HOLDOUT, after fees (each market-hour counted):\n" + F4.round(2).to_string(index=False))
    os.makedirs(f"{RD}/results", exist_ok=True)
    pd.concat([T.assign(table="logloss"), F3.assign(table="favourites"), F4.assign(table="taker")]).to_csv(
        f"{RD}/results/kalshi_fairvalue.csv", index=False)
    a = T[(T.part == "hold") & (T.lead == "ALL")].iloc[0]
    print(f"::notice::fair value HOLDOUT: log-loss market {a.ll_market:.4f} model {a.ll_model:.4f} blend {a.ll_blend:.4f}; "
          f"model-market {a.model_minus_mkt:+.4f} [{a.ci_lo:+.4f},{a.ci_hi:+.4f}], blend-market {a.blend_minus_mkt:+.4f} "
          f"[{a.blend_ci_lo:+.4f},{a.blend_ci_hi:+.4f}] | favourites the model dislikes >5pts: "
          + (f"{F3.iloc[3].edge_at_mid_c:+.1f}c vs all {F3.iloc[0].edge_at_mid_c:+.1f}c" if len(F3) > 3 else "n/a"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
