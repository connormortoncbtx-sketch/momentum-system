"""
research/kalshi/weather_expand.py -- which OTHER temperature series would the bot earn on? (Oct 19, 2026)

Same measurement as mm_perfill.py (the bot's deep fills: a sweep that trades >= 2c past its first price would fill a
quote 2c behind; fill min(5, contracts at/beyond our level); held to settlement), applied to every daily high/low
temperature series on Kalshi. The 8 series the bot trades are included as the control.
  sample   per series, up to N_PER (default 60) settled markets from the last 365 days, spread across months
  trades   full history per market (up to 25 pages)
  filters  the bot's: 6-48h before close, fill price 3-97c
  report   per series: markets, deep fills / market, edge per fill (c/contract), per-market edge and t (clustered by
           event = city-day), recent half (last 180 days) vs older half, touch edge, typical daily volume
"""
import concurrent.futures as cf
import os, re, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mm_scan as M   # get / paged / rate limiter / Fail

RD = os.environ.get("RESEARCH_DATA", "research_data")
LIST = os.environ.get("SERIES_FILE")          # other mode: test the series listed in this file instead of temperature
TAG = "other" if LIST else "weather"
OUT = f"{RD}/kalshi/expand" + ("_other" if LIST else ""); os.makedirs(OUT, exist_ok=True)
WINDOWS = {"6-48h": (6, 48), "6h+": (6, 1e9)}   # the bot's current window, and quoting any time >= 6h before close
N_PER = int(os.environ.get("N_PER") or 60)
CURRENT = {"KXHIGHNY", "KXHIGHCHI", "KXHIGHLAX", "KXHIGHMIA", "KXHIGHPHIL", "KXHIGHDEN", "KXHIGHAUS", "KXHIGHTDAL"}
OFF, SIZE, MIN_H, MAX_H = 0.02, 5, 6, 48
NOW = pd.Timestamp.now(tz="UTC")


def series():
    S = M.series_list()
    if LIST:
        want = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), LIST)) if l.strip()]
        return S[S.ticker.isin(want)]
    S = S[S.ticker.str.match(r"^KX(HIGH|LOW)") & ~S.ticker.str.contains("NYD|^KXHIGHUS$|HOURLY", regex=True)]
    return S


def markets(s):
    rows = M.paged("/markets", "markets", max_pages=10, series_ticker=s, status="settled") + \
           M.paged("/historical/markets", "markets", max_pages=10, series_ticker=s)
    d = pd.DataFrame([dict(ticker=m["ticker"], close=m.get("close_time"), result=m.get("result"),
                           volume=M.f(m.get("volume_fp", m.get("volume")))) for m in rows if m.get("result") in ("yes", "no")])
    if not len(d):
        return d
    d = d.drop_duplicates("ticker")
    d["close"] = pd.to_datetime(d.close, utc=True, errors="coerce")
    return d[d.close >= NOW - pd.Timedelta(days=365)]


def trades(tk):
    rows = []
    for path in ("/markets/trades", "/historical/trades"):
        rows = M.paged(path, "trades", max_pages=25, ticker=tk)
        if rows:
            break
    return [dict(ticker=tk, t=r.get("created_time"), p=M.f(r.get("yes_price_dollars")), c=M.f(r.get("count_fp", r.get("count"))),
                 taker_side=r.get("taker_side"), block=bool(r.get("is_block_trade"))) for r in rows]


def scan(s):
    cache = f"{OUT}/{s}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    mk = markets(s)
    if len(mk) < 10:
        pd.DataFrame({"ticker": []}).to_parquet(cache, index=False); return None
    mk["month"] = mk.close.dt.strftime("%Y-%m")
    per = max(1, N_PER // mk.month.nunique())
    rng = np.random.default_rng(abs(hash(s)) % 2**32)
    smp = mk.assign(_r=rng.random(len(mk))).sort_values("_r").groupby("month").head(per).head(N_PER)
    tr = []
    for tk in smp.ticker:
        tr += trades(tk)
    if not tr:
        pd.DataFrame({"ticker": []}).to_parquet(cache, index=False); return None
    t = pd.DataFrame(tr).merge(smp[["ticker", "close", "result", "volume"]], on="ticker")
    t["series"] = s
    t.to_parquet(cache, index=False)
    return t


def deep_fills(t, min_h=MIN_H, max_h=MAX_H):
    t = t[(t.p > 0) & (t.p < 1) & t.c.notna() & ~t.block].copy()
    t["ts"] = pd.to_datetime(t.t, utc=True, format="ISO8601")
    t["R"] = (t.result == "yes").astype(float)
    t["h"] = (t.close - t.ts).dt.total_seconds() / 3600
    t = t.sort_values(["ticker", "ts"])
    key = ["ticker", "ts", "taker_side"]
    first = t.groupby(key, sort=False).p.transform("first")
    beyond = np.where(t.taker_side == "no", t.p <= first - OFF + 1e-9, t.p >= first + OFF - 1e-9)
    t["q"] = np.where(beyond, t.c, 0.0)
    sw = t.groupby(key, sort=False).agg(first=("p", "first"), q=("q", "sum"), R=("R", "first"), h=("h", "first"),
                                         close=("close", "first")).reset_index()
    sw = sw[(sw.q > 0) & (sw.h >= min_h) & (sw.h <= max_h)]
    sw["px"] = np.where(sw.taker_side == "no", sw["first"] - OFF, sw["first"] + OFF)
    sw = sw[(sw.px >= 0.03) & (sw.px <= 0.97)]
    sw["e"] = np.where(sw.taker_side == "no", sw.R - sw.px, sw.px - sw.R)
    sw["c"] = np.minimum(sw.q, SIZE)
    touch = t[(t.h >= min_h) & (t.h <= max_h) & (t.p >= 0.03) & (t.p <= 0.97)]
    te = np.where(touch.taker_side == "yes", touch.p - touch.R, touch.R - touch.p)
    return sw, float((te * np.minimum(touch.c, SIZE)).sum() / np.minimum(touch.c, SIZE).sum()) if len(touch) else np.nan


def summarise(s, t, win="6-48h"):
    sw, touch_c = deep_fills(t, *WINDOWS[win])
    mk = t.drop_duplicates("ticker")
    out = dict(series=s, window=win, current=s in CURRENT, markets=len(mk), med_mkt_volume=mk.volume.median(), touch_c=touch_c * 100)
    if len(sw) < 20:
        return dict(out, fills=len(sw))
    pm = (sw.e * sw.c).groupby(sw.ticker).sum() / sw.c.groupby(sw.ticker).sum()
    ev = pm.groupby(pm.index.str.rsplit("-", n=1).str[0]).mean()
    recent = sw.close >= NOW - pd.Timedelta(days=180)
    f = lambda x: (x.e * x.c).sum() / x.c.sum() * 100 if len(x) else np.nan
    return dict(out, fills=len(sw), fills_per_mkt=len(sw) / len(mk), edge_c=f(sw), per_market_c=pm.mean() * 100,
                t=pm.mean() / (ev.std() / np.sqrt(len(ev))) if len(ev) > 2 else np.nan,
                recent_c=f(sw[recent]), older_c=f(sw[~recent]), recent_fills=int(recent.sum()))


def main():
    S = series()
    print(f"temperature series: {len(S)}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(scan, s): s for s in S.ticker}
        for fu in cf.as_completed(futs):
            s = futs[fu]
            try:
                t = fu.result()
                if t is not None and len(t) and "p" in t:
                    for w in WINDOWS:
                        rows.append(summarise(s, t, w))
            except Exception as e:
                print(f"::warning::{s}: {type(e).__name__}: {str(e)[:150]}", flush=True)
    R = pd.DataFrame(rows).merge(S[[c for c in ("ticker", "title", "category", "frequency") if c in S]].rename(columns={"ticker": "series"}),
                                 on="series", how="left")
    R = R.sort_values(["window", "current", "per_market_c"], ascending=[True, False, False])
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/kalshi_{TAG}_expand.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 38)
    cols = ["series", "window", "category", "frequency", "title", "markets", "med_mkt_volume", "fills", "fills_per_mkt", "edge_c", "per_market_c", "t",
            "recent_c", "older_c", "recent_fills", "touch_c"]
    print(R[[c for c in cols if c in R]].round(2).to_string(index=False))
    cur = R[R.current & (R.window == "6-48h")]
    good = R[~R.current & (R.per_market_c > 1.5) & (R.t > 2.5) & (R.recent_c > 0.5) & (R.fills_per_mkt >= 5)]
    print(f"::notice::{TAG} expand: control (current 8) per-market edge median {cur.per_market_c.median() if len(cur) else float('nan'):.2f}c; "
          f"{len(good)} new series pass (per-market > 1.5c, t > 2.5, recent > 0.5c, >= 5 fills/market): "
          + ", ".join(f"{r.series}[{r.window}] {r.per_market_c:.1f}c t{r.t:.1f}" for r in good.head(25).itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
