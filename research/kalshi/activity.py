"""
research/kalshi/activity.py -- does market ACTIVITY tell the bot when and where to deploy capital? (Nov 4, 2026)

Kalshi has no market cap; the equivalents are volume (contracts traded) and open interest. Three questions, public
data only, the bot's 11 daily-high series:
 1. BUSY vs QUIET: the bot's backtest fills (2c behind, 5-lot, favourite >= 60c, 6-48h before close, :45-:59 skipped;
    as depth_curve.py) cut by how busy the city-day was AT THE TIME OF THE FILL -- contracts traded across all its
    buckets in the previous 3 hours (causal: the bot could know it live). Terciles cut on the first half of the sample;
    edge per contract with 95% bands clustered by market day, first and second half separately.
    Also by the city-day's whole-window volume (known only afterwards -- descriptive).
 2. WHEN: edge per contract and share of all volume by hour of day (Central).
 3. CAPACITY TREND: median contracts traded per city-day, by city and month, over the last 12 months (from the markets
    endpoint -- every settled market, not a sample).
Output: results/kalshi_activity.csv (+ _hours, _trend).
"""
import concurrent.futures as cf
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mm_scan as M
import depth_curve as DC

RD = os.environ.get("RESEARCH_DATA", "research_data")
N_EV = int(os.environ.get("N_EV") or 80)
DC.N_EV = N_EV
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)


def band(x, unit="mday"):
    g = x.assign(v=x.e * x.c).groupby(unit).agg(v=("v", "sum"), w=("c", "sum"))
    N = g.w.sum()
    if N <= 0:
        return np.nan, np.nan, 0, 0
    m = g.v.sum() / N; G = len(g)
    se = np.sqrt(((g.v - m * g.w) ** 2).sum() / N ** 2 * G / (G - 1)) if G > 2 else np.nan
    return m, 1.96 * se, G, N


def trend():
    rows = []
    for s in DC.SERIES:
        mk = M.paged("/markets", "markets", max_pages=20, series_ticker=s, status="settled") + \
             M.paged("/historical/markets", "markets", max_pages=20, series_ticker=s)
        for m in mk:
            rows.append(dict(series=s, event=m.get("event_ticker"), close=m.get("close_time"),
                             vol=M.f(m.get("volume_fp", m.get("volume")))))
    T = pd.DataFrame(rows).drop_duplicates()
    T["close"] = pd.to_datetime(T.close, utc=True)
    T = T[T.close >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)]
    ev = T.groupby(["series", "event"]).agg(vol=("vol", "sum"), close=("close", "max")).reset_index()
    ev["month"] = ev.close.dt.strftime("%Y-%m")
    P = ev.pivot_table(index="series", columns="month", values="vol", aggfunc="median").round(0)
    return P


def main():
    with cf.ThreadPoolExecutor(4) as ex:
        mk = pd.concat([d for d in ex.map(DC.events, DC.SERIES) if len(d)], ignore_index=True)
    print(f"{mk.event.nunique()} city-days, {len(mk)} markets", flush=True)
    F, TR = [], []
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(DC.trades, m.ticker): m for m in mk.itertuples()}
        for i, fu in enumerate(cf.as_completed(futs)):
            m = futs[fu]
            try:
                t = fu.result()
                if len(t):
                    tt = t[(t.p > 0) & (t.p < 1)].copy()
                    tt["ts"] = pd.to_datetime(tt.t, utc=True, format="ISO8601"); tt["event"] = m.event
                    TR.append(tt[["event", "ts", "c"]])
                F.append(DC.sim(t, m))
            except Exception as e:
                print(f"::warning::{m.ticker}: {type(e).__name__}: {str(e)[:100]}")
            if i % 500 == 499:
                print(f"  {i + 1}/{len(mk)}", flush=True)
    F = pd.concat([f for f in F if len(f)], ignore_index=True)
    TR = pd.concat(TR, ignore_index=True).sort_values(["event", "ts"])
    # causal activity: contracts traded in the city-day in the 3h before each fill
    act, trs = [], dict(tuple(TR.groupby("event")))
    for ev, g in F.groupby("event"):
        tr = trs.get(ev)
        if tr is None:
            act.append(pd.Series(0.0, index=g.index)); continue
        ts = tr.ts.values.astype("datetime64[s]").astype("int64"); cum = np.concatenate([[0.0], tr.c.cumsum().values])
        q = g.ts.values.astype("datetime64[s]").astype("int64")
        hi = np.searchsorted(ts, q, side="left"); lo = np.searchsorted(ts, q - 3 * 3600, side="left")
        act.append(pd.Series(cum[hi] - cum[lo], index=g.index))
    F["act3h"] = pd.concat(act)
    tot = TR.groupby("event").c.sum()
    F["day_vol"] = F.event.map(tot)
    F["mday"] = F.event.str.split("-").str[1]
    F["hour_ct"] = F.ts.dt.tz_convert("America/Chicago").dt.hour
    half = F.ts.quantile(0.5)
    F["half"] = np.where(F.ts < half, "1st", "2nd")
    rows = []
    for col, lab in (("act3h", "busy in the last 3h (causal)"), ("day_vol", "whole city-day volume (hindsight)")):
        cuts = F.loc[F.half == "1st", col].quantile([1 / 3, 2 / 3]).to_numpy()
        F["t"] = np.searchsorted(cuts, F[col].to_numpy(), side="right")
        for k, name in enumerate(["quiet", "middle", "busy"]):
            x = F[F.t == k]
            m, hw, G, N = band(x)
            a = band(x[x.half == "1st"])[0]; b = band(x[x.half == "2nd"])[0]
            rows.append(dict(cut=lab, group=f"{name} ({'<' if k == 0 else '>=' if k == 2 else ''}"
                                               f"{cuts[0] if k == 0 else cuts[1] if k == 2 else cuts[0]:.0f})",
                             contracts=N, market_days=G, edge_c=m, ci=hw, first_half=a, second_half=b))
    R = pd.DataFrame(rows)
    H = []
    allv = TR.assign(h=TR.ts.dt.tz_convert("America/Chicago").dt.hour).groupby("h").c.sum()
    for h, x in F.groupby("hour_ct"):
        m, hw, G, N = band(x)
        H.append(dict(hour_ct=h, contracts=N, edge_c=m, ci=hw, first_half=band(x[x.half == "1st"])[0],
                      second_half=band(x[x.half == "2nd"])[0], share_of_volume=allv.get(h, 0) / allv.sum() * 100))
    H = pd.DataFrame(H)
    P = trend()
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/kalshi_activity.csv", index=False)
    H.to_csv(f"{RD}/results/kalshi_activity_hours.csv", index=False)
    P.to_csv(f"{RD}/results/kalshi_activity_trend.csv")
    print("\n1. BUSY vs QUIET (edge c/contract, 95% band by market day):\n" + R.round(2).to_string(index=False))
    print("\n2. BY HOUR (Central):\n" + H.round(2).to_string(index=False))
    print("\n3. MEDIAN CONTRACTS PER CITY-DAY BY MONTH:\n" + P.to_string())
    c = R[R.cut.str.startswith("busy")].set_index("group")
    print("::notice::activity: causal 3h terciles edge " + ", ".join(f"{g.split(' ')[0]} {r.edge_c:+.1f}c ±{r.ci:.1f} "
          f"(halves {r.first_half:+.1f}/{r.second_half:+.1f})" for g, r in c.iterrows()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
