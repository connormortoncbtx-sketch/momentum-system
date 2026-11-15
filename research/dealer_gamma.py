"""
research/dealer_gamma.py -- dealer gamma exposure (GEX) of SPY / QQQ options at the start of each day, as a condor
filter (Nov 15, 2026; Connor: "test everything").

Idea: when dealers are net LONG gamma they hedge against moves (pinning); net SHORT gamma they hedge with moves
(trend days). Convention (as the industry GEX estimates): customers are net long puts and net short calls, so dealer
gamma = + call OI x gamma - put OI x gamma.
PRE-REGISTERED:
  GEX_d   sum over every contract expiring within 35 days of: sign x OI (start of day d = prior-day positions) x
          Black-Scholes gamma at the prior close, flat vol = prior VIX close (SPY) / 1.2 x VIX (QQQ), time to expiry
          from the prior close, x 100 x S^2 x 1% (dollar gamma per 1% move), in $bn
  z       GEX_d / its trailing 250-day standard deviation (per underlying)
  rule    skip the condor on the 10% most negative-z days (cut-off = earlier days only)
  pass    raises the Sharpe of the 2pm AND 3pm condor in BOTH 2020-23 and 2024-26, and adds to the range-so-far rule
          (range + GEX skip beats range skip alone in both periods for both slots)
Data: Theta Terminal (theta_run.yml); only the daily aggregates leave the runner -> results/dealer_gamma_daily.parquet.
MODE=probe: test the open-interest endpoint formats and stop.
"""
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import options_theta as OT  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
MODE = os.environ.get("MODE", "")
LIMIT = int(os.environ.get("LIMIT") or 0)
note = lambda m: print(f"::notice::{m}", flush=True)


def oi_day(und, d):
    ds = d.strftime("%Y%m%d")
    if OT.API == "v2":
        rows = OT.get_v2("/v2/bulk_hist/option/open_interest", root=und, exp=0, start_date=ds, end_date=ds, use_csv="false")
    else:
        rows = OT.get("/option/history/open_interest", symbol=und, expiration="*", date=ds)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    low = {c.lower(): c for c in df.columns}
    pick = lambda *n: next((low[x] for x in n if x in low), None)
    k, r, e, oi = pick("strike"), pick("right"), pick("expiration", "exp"), pick("open_interest", "oi")
    out = pd.DataFrame({"strike": pd.to_numeric(df[k], errors="coerce"), "right": df[r].astype(str).str.upper().str[0],
                        "exp": pd.to_datetime(df[e].astype(str).str.replace("-", "").str[:8], format="%Y%m%d", errors="coerce"),
                        "oi": pd.to_numeric(df[oi], errors="coerce")})
    if out.strike.median() > 5000:
        out["strike"] /= 1000
    return out.dropna()


def gamma(S, K, T, s):
    d1 = (np.log(S / K) + 0.5 * s * s * T) / (s * np.sqrt(T))
    return np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (S * s * np.sqrt(T))


def probe():
    for d in (dt.date(2019, 3, 8), dt.date(2024, 6, 12)):
        try:
            o = oi_day("SPY", d)
            note(f"probe OI SPY {d}: {len(o)} rows, exps {o.exp.nunique() if len(o) else 0}, sample {o.head(3).to_dict('records')}")
        except Exception as ex:
            note(f"probe OI SPY {d} failed: {str(ex)[:400]}")


def main():
    if MODE == "probe":
        return probe()
    O = pd.read_parquet(f"{RD}/results/options_series_daily.parquet", columns=["d", "und"]).drop_duplicates()
    O["d"] = pd.to_datetime(O.d)
    vix = pd.read_parquet(f"{RD}/macro/cboe_VIX.parquet").assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").value
    rows, done = [], 0
    for und in ("SPY", "QQQ"):
        bars = pd.read_parquet(f"{RD}/intraday_liquid/5m_{und}.parquet")
        bars["d"] = pd.to_datetime(bars.ts).dt.normalize()
        close = bars.groupby("d").close.last()
        days = sorted(O[O.und == und].d)
        if LIMIT:
            days = days[-LIMIT:]
        for d in days:
            prev = close.index[close.index < d]
            if not len(prev):
                continue
            pd_ = prev[-1]
            S = float(close[pd_])
            v = vix[vix.index <= pd_]
            if not len(v):
                continue
            sig = float(v.iloc[-1]) / 100 * (1.2 if und == "QQQ" else 1.0)
            try:
                o = oi_day(und, d.date())
            except Exception as ex:
                print(f"{und} {d.date()}: {str(ex)[:200]}", flush=True)
                continue
            if o.empty:
                continue
            o = o[(o.exp >= d) & (o.exp <= d + pd.Timedelta(days=35)) & (o.oi > 0)]
            # time from the prior close (16:00) to the expiry's 16:00 close, in years
            T = ((o.exp - pd_).dt.days.clip(lower=0) * 24 / (365 * 24)).clip(lower=1 / (365 * 4))
            g = gamma(S, o.strike.to_numpy(), T.to_numpy(), sig)
            dollar = o.oi.to_numpy() * g * 100 * S * S * 0.01 / 1e9
            sign = np.where(o.right.to_numpy() == "C", 1.0, -1.0)
            same = (o.exp == d).to_numpy()
            rows.append(dict(d=d, und=und, spot_prev=S, gex=float((sign * dollar).sum()),
                             gex_ex0dte=float((sign * dollar)[~same].sum()), gross=float(dollar.sum()),
                             call_gex=float(dollar[sign > 0].sum()), put_gex=float(dollar[sign < 0].sum()), n=len(o)))
            done += 1
            if done % 200 == 0:
                print(f"  {done} days ({und} {d.date()}: GEX {rows[-1]['gex']:+.2f}bn)", flush=True)
    D = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    D.to_parquet(f"{RD}/results/dealer_gamma_daily.parquet", index=False)
    note(f"dealer gamma: {len(D)} und-days, {D.d.min().date()}..{D.d.max().date()}; median GEX SPY "
         f"{D[D.und == 'SPY'].gex.median():+.2f}bn, negative share {(D.gex < 0).mean():.0%}")


if __name__ == "__main__":
    main()
