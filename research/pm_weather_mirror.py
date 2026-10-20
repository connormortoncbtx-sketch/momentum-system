"""
research/pm_weather_mirror.py -- mirror a Polymarket weather specialist's buys onto the same Kalshi bucket (Oct 20, 2026).

The persistent $2-20k Polymarket wallet 0x213e1d... trades only same-day daily-high buckets (1-5pm ET, edge in the
leading bucket at 50-90c). Question: if we bought the SAME city / date / temperature range on Kalshi at the moment it
bought on Polymarket (or 1 / 5 minutes later), paying Kalshi's ask and taker fee, would it pay?
  match      Polymarket "highest temperature in <city> be between a-b F on <date>" -> the Kalshi daily-high market for
             that city and date covering exactly a..b (Kalshi bucket edges differ by city; non-exact -> skipped)
  price      Kalshi 1-minute candlesticks: yes ASK close at the minute of the Polymarket buy + lag (0 / 1 / 5 min)
  outcome    Kalshi's own result (stations can differ from Polymarket's source); agreement rate reported
Public Kalshi API only; Polymarket trades from research_data/polymarket (pm_copytrade2 cache).
"""
import datetime as dt, json, os, re, time, traceback, urllib.parse, urllib.request
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
API = "https://api.elections.kalshi.com/trade-api/v2"
WALLET = os.environ.get("WALLET", "0x213e1deb37695fb513e1110e6506220c37a75e0f")
SERIES = {"Los Angeles": "KXHIGHLAX", "New York City": "KXHIGHNY", "Seattle": "KXHIGHTSEA", "Houston": "KXHIGHTHOU",
          "San Francisco": "KXHIGHTSFO", "Atlanta": "KXHIGHTATL", "Miami": "KXHIGHMIA", "Austin": "KXHIGHAUS",
          "Denver": "KXHIGHDEN", "Chicago": "KXHIGHCHI", "Philadelphia": "KXHIGHPHIL", "Dallas": "KXHIGHTDAL",
          "Boston": "KXHIGHTBOS", "Las Vegas": "KXHIGHTLV"}
LAGS = (0, 1, 5)
_last = [0.0]


def get(path, **q):
    for i in range(5):
        w = 0.12 - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        _last[0] = time.time()
        try:
            url = f"{API}{path}" + ("?" + urllib.parse.urlencode(q) if q else "")
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            if "404" in str(e):
                return {}
            time.sleep(2 * (i + 1))
    return {}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def event_markets(series, day):
    ev = f"{series}-{day:%y}{day:%b}".upper() + f"{day:%d}"
    d = get("/markets", event_ticker=ev, limit=100)
    out = []
    for m in d.get("markets", []):
        lo, hi = f(m.get("floor_strike")), f(m.get("cap_strike"))
        st = m.get("strike_type")
        if st == "between" and np.isfinite(lo) and np.isfinite(hi):
            out.append(dict(ticker=m["ticker"], lo=lo, hi=hi, result=m.get("result")))
    return out


def ask_at(series, ticker, ts):
    """{lag: yes ask (dollars)} from 1-minute candles around ts."""
    d = get(f"/series/{series}/markets/{ticker}/candlesticks", start_ts=ts - 120, end_ts=ts + 7 * 60, period_interval=1)
    cs = d.get("candlesticks", [])
    out = {}
    for L in LAGS:
        tgt = ts + L * 60
        best = None
        for c in cs:
            if c.get("end_period_ts", 0) >= tgt:
                a = c.get("yes_ask") or {}
                v = f(a.get("close_dollars")) if a.get("close_dollars") is not None else f(a.get("close")) / 100
                if np.isfinite(v) and 0 < v < 1:
                    best = v
                break
        if best is not None:
            out[L] = best
    return out


def main():
    T = pd.read_parquet(f"{RD}/polymarket/w_{WALLET}.parquet")
    T = T[T.side == "BUY"].copy()
    rx = T.title.str.extract(r"highest temperature in (.+?) be between (\d+)-(\d+)°F on ([A-Z][a-z]+ \d+)\?")
    T["city"], T["a"], T["b"] = rx[0], pd.to_numeric(rx[1]), pd.to_numeric(rx[2])
    T["ts"] = T.timestamp.astype(int)
    T["day"] = pd.to_datetime(rx[3] + " " + pd.to_datetime(T.ts, unit="s").dt.year.astype(str), format="%B %d %Y", errors="coerce")
    T = T[T.city.isin(SERIES) & T.day.notna()]
    print(f"high-temperature buys in mapped cities: {len(T):,}", flush=True)
    evcache, rows = {}, []
    for r in T.itertuples():
        s = SERIES[r.city]; key = (s, r.day.date())
        if key not in evcache:
            evcache[key] = event_markets(s, r.day)
        m = [x for x in evcache[key] if x["lo"] == r.a and x["hi"] == r.b]
        if not m:
            rows.append(dict(city=r.city, matched=False)); continue
        m = m[0]
        if m["result"] not in ("yes", "no"):
            rows.append(dict(city=r.city, matched=True, resolved=False)); continue
        px = ask_at(s, m["ticker"], r.ts)
        pm_win = None
        rows.append(dict(city=r.city, matched=True, resolved=True, ticker=m["ticker"], pm_price=float(r.price),
                         R=1.0 if m["result"] == "yes" else 0.0, **{f"ask{L}": px.get(L, np.nan) for L in LAGS}))
    D = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    D.to_csv(f"{RD}/results/pm_weather_mirror.csv", index=False)
    M = D[D.get("resolved", False) == True].copy()
    print(f"::notice::mirror: {len(D)} buys, matched exactly {D.matched.mean() * 100:.0f}%, resolved on Kalshi {len(M)}")
    for L in LAGS:
        a = M[f"ask{L}"]
        ok = a.notna() & a.between(0.03, 0.97)
        x = M[ok]; p = a[ok]
        fee = np.ceil(0.07 * p * (1 - p) * 100 - 1e-9) / 100
        net = x.R - p - fee
        tick = x.ticker.str.rsplit("-", n=1).str[0]
        pe = net.groupby(tick).mean()
        print(f"::notice::mirror +{L}min on Kalshi (buy at ask + fee): {len(x)} buys, Kalshi ask avg {p.mean() * 100:.1f}c "
              f"vs their Polymarket price {x.pm_price.mean() * 100:.1f}c; win {x.R.mean() * 100:.0f}%; net {net.mean() * 100:+.2f}c "
              f"(city-day t {pe.mean() / pe.std() * np.sqrt(len(pe)):.1f}, {len(pe)} city-days)")
    print(M.groupby("city").agg(n=("R", "size"), win=("R", "mean")).round(2).to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        for line in traceback.format_exc().strip().splitlines()[-10:]:
            print(f"::error::{line}")
        raise
