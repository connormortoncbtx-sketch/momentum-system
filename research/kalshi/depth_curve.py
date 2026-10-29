"""
research/kalshi/depth_curve.py -- the marginal-contract curve past today's city-day cap (Oct 29, 2026).

Live fills stop at the cap (EVENT_CAP=30), so `kmm study`'s DEPTH table can't see what the 31st-60th contract of a
city-day would earn. This replays the public tape on WHOLE city-days of the bot's 11 series (every bucket, so depth is
counted the way the cap counts it): the bot's backtest rule (2c behind, 5-lot, sweeps; favourite >= FAV, 6-48h before
close, :45-:59 skipped), fills taken in time order across the city-day with NO cap, each fill tagged with the
contracts the backtest already holds in that city-day. Edge per contract by depth band, 95% bands clustered by city-day
and by market day. Calibration (kmm calibrate, Oct 27): real fills ~0.28x the backtest's, edge per contract about the
same -- so backtest depth D corresponds to roughly 0.28 x D real contracts; both are shown.
Public data only -> results/kalshi_depth_curve.csv.
"""
import concurrent.futures as cf
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mm_scan as M

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/kalshi/depth"; os.makedirs(OUT, exist_ok=True)
SERIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHLAX", "KXHIGHMIA", "KXHIGHPHIL", "KXHIGHDEN", "KXHIGHAUS", "KXHIGHTDAL",
          "KXHIGHTATL", "KXHIGHTBOS", "KXHIGHTLV"]
N_EV = int(os.environ.get("N_EV") or 40)               # city-days per series (most recent settled)
OFF, SIZE, FAV = 0.02, 5, float(os.environ.get("FAV") or 0.6)
RATIO = 0.28
pd.set_option("display.width", 200)


def events(s):
    rows = M.paged("/markets", "markets", max_pages=10, series_ticker=s, status="settled")
    d = pd.DataFrame([dict(ticker=m["ticker"], event=m.get("event_ticker"), close=m.get("close_time"), result=m.get("result"))
                      for m in rows if m.get("result") in ("yes", "no")])
    if d.empty:
        return d
    d["close"] = pd.to_datetime(d.close, utc=True)
    keep = d.groupby("event").close.max().sort_values().index[-N_EV:]
    return d[d.event.isin(keep)]


def trades(tk):
    p = f"{OUT}/{tk}.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    rows = M.paged("/markets/trades", "trades", max_pages=40, ticker=tk)
    t = pd.DataFrame([dict(ticker=tk, t=r.get("created_time"), p=M.f(r.get("yes_price_dollars")),
                           c=M.f(r.get("count_fp", r.get("count"))), side=r.get("taker_side"),
                           block=bool(r.get("is_block_trade"))) for r in rows])
    t.to_parquet(p, index=False)
    return t


def sim(t, m):
    t = t[(t.p > 0) & (t.p < 1) & ~t.block].copy()
    if t.empty:
        return t
    t["ts"] = pd.to_datetime(t.t, utc=True, format="ISO8601")
    sw = t.groupby(["ts", "side"]).p.agg(["max", "min"]).reset_index()
    sw["first"] = np.where(sw.side == "no", sw["max"], sw["min"])
    t = t.merge(sw[["ts", "side", "first"]], on=["ts", "side"])
    t["q"] = np.where(t.side == "no", np.where(t.p <= t["first"] - OFF + 1e-9, t.c, 0), np.where(t.p >= t["first"] + OFF - 1e-9, t.c, 0))
    f = t.groupby(["ts", "side"]).agg(first=("first", "first"), q=("q", "sum")).reset_index()
    f = f[f.q > 0]
    f["d"] = np.where(f.side == "no", 1, -1)
    f["px"] = np.where(f.d > 0, f["first"] - OFF, f["first"] + OFF)
    f["held"] = np.where(f.d > 0, f.px, 1 - f.px)
    f["h"] = (m.close - f.ts).dt.total_seconds() / 3600
    f = f[(f.px >= 0.03) & (f.px <= 0.97) & (f.held >= FAV) & (f.h >= 6) & (f.h <= 48) & ~f.ts.dt.minute.between(45, 59)]
    f["c"] = np.minimum(f.q, SIZE)
    R = 1.0 if m.result == "yes" else 0.0
    f["e"] = f.d * (R - f.px) * 100
    f["ticker"], f["event"] = m.ticker, m.event
    return f[["ts", "ticker", "event", "c", "e"]]


def band(x, unit):
    g = x.assign(v=x.e * x.c).groupby(unit).agg(v=("v", "sum"), w=("c", "sum"))
    N = g.w.sum(); m = g.v.sum() / N; G = len(g)
    se = np.sqrt(((g.v - m * g.w) ** 2).sum() / N ** 2 * G / (G - 1)) if G > 2 else np.nan
    return m, 1.96 * se, G, N


def main():
    with cf.ThreadPoolExecutor(4) as ex:
        mk = pd.concat([d for d in ex.map(events, SERIES) if len(d)], ignore_index=True)
    print(f"{mk.event.nunique()} city-days, {len(mk)} markets ({N_EV} most recent per series)", flush=True)
    F = []
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(trades, m.ticker): m for m in mk.itertuples()}
        for i, fu in enumerate(cf.as_completed(futs)):
            m = futs[fu]
            try:
                F.append(sim(fu.result(), m))
            except Exception as e:
                print(f"::warning::{m.ticker}: {type(e).__name__}: {str(e)[:100]}")
            if i % 300 == 299:
                print(f"  {i + 1}/{len(mk)}", flush=True)
    F = pd.concat([f for f in F if len(f)], ignore_index=True).sort_values(["event", "ts"])
    F["depth"] = F.groupby("event").c.cumsum() - F.c
    F["mday"] = F.event.str.split("-").str[1]
    edges = [0, 10, 20, 30, 40, 60, 80, 120, 1e9]
    labels = ["0-9", "10-19", "20-29", "30-39", "40-59", "60-79", "80-119", "120+"]
    F["band"] = pd.cut(F.depth, edges, right=False, labels=labels)
    rows = []
    for b, x in F.groupby("band", observed=True):
        m, hw, G, N = band(x, "event"); _, hwd, D, _ = band(x, "mday")
        rows.append(dict(depth_backtest=b, depth_real_approx=f"~{int(edges[labels.index(b)] * RATIO)}+",
                         contracts=N, city_days=G, edge_c=m, ci_cityday=hw, ci_marketday=hwd))
    R = pd.DataFrame(rows)
    per_ev = F.groupby("event").c.sum()
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/kalshi_depth_curve.csv", index=False)
    print(f"\nbacktest contracts per city-day (no cap): median {per_ev.median():.0f}, 75th pct {per_ev.quantile(.75):.0f}, "
          f"90th {per_ev.quantile(.9):.0f}  (x{RATIO} for real: {per_ev.median() * RATIO:.0f} / {per_ev.quantile(.75) * RATIO:.0f} / "
          f"{per_ev.quantile(.9) * RATIO:.0f})")
    print("\nEDGE BY DEPTH (c/contract; 95% bands by city-day and by market day):")
    print(R.round(2).to_string(index=False))
    print("::notice::depth curve (backtest depth: edge c): " + ", ".join(f"{r.depth_backtest} {r.edge_c:+.1f}" for r in R.itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
