"""
research/fetch_kalshi_fed.py -- Kalshi's public market-data API: Fed rate-decision markets, the odds at 13:00 ET on
decision day (before the 14:00 statement), for the condor event test. Nov 15, 2026. Data API only (no account, no
trading, no scraping).
Discovery: every series whose title mentions the Fed and a rate decision; settled markets; hourly candlesticks.
Per decision day: max_prob (the favourite's yes price at 13:00), entropy of the normalised outcome prices.
Output: research_data/results/kalshi_fed_events.csv
"""
import datetime as dt
import json
import math
import os
import time
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
BASE = "https://api.elections.kalshi.com/trade-api/v2"
ET = ZoneInfo("America/New_York")


def get(path, **q):
    url = f"{BASE}{path}" + ("?" + urllib.parse.urlencode(q) if q else "")
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "momentum-research"}), timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if "404" in str(e):
                return {}
            time.sleep(2 * (k + 1))
    return {}


def probe():
    for path, q in (("/markets", dict(series_ticker="KXFED", limit=1000)), ("/markets", dict(series_ticker="FED", limit=1000)),
                    ("/markets", dict(series_ticker="KXFED", status="closed", limit=1000)),
                    ("/events", dict(series_ticker="KXFED", limit=200)), ("/events", dict(series_ticker="FED", limit=200)),
                    ("/historical/markets", dict(series_ticker="KXFED", limit=1000)), ("/historical/markets", dict(series_ticker="FED", limit=1000)),
                    ("/historical/events", dict(series_ticker="FED", limit=200))):
        j = get(path, **q)
        items = j.get("markets") or j.get("events") or []
        dates = sorted(str(x.get("close_time") or x.get("strike_date") or x.get("event_ticker"))[:10] for x in items)
        print(f"probe {path} {q}: keys {list(j)[:5]} n {len(items)} cursor {bool(j.get('cursor'))} range {dates[:1]}..{dates[-1:]}", flush=True)


def main():
    if os.environ.get("MODE") == "probe" or os.environ.get("LIMIT") == "probe":
        return probe()
    S = get("/series").get("series", [])
    fed = [s for s in S if "fed" in (s.get("title", "") + s.get("ticker", "")).lower()
           and any(w in s.get("title", "").lower() for w in ("decision", "rate", "fomc", "hike", "cut"))]
    print(f"series total {len(S)}; Fed-rate candidates: {[(s['ticker'], s['title']) for s in fed][:20]}", flush=True)
    rows = []
    for s in fed:
        cur, mk = None, []
        while True:
            q = dict(series_ticker=s["ticker"], status="settled", limit=1000)
            if cur:
                q["cursor"] = cur
            j = get("/markets", **q)
            mk += j.get("markets", [])
            cur = j.get("cursor")
            if not cur:
                break
        print(f"  {s['ticker']}: {len(mk)} settled markets", flush=True)
        by_ev = {}
        for m in mk:
            by_ev.setdefault(m["event_ticker"], []).append(m)
        for ev, ms in by_ev.items():
            close = pd.Timestamp(max(m["close_time"] for m in ms)).tz_convert(ET)
            day = close.date()
            t13 = dt.datetime.combine(day, dt.time(13, 0), ET)
            px = []
            for m in ms:
                c = get(f"/series/{s['ticker']}/markets/{m['ticker']}/candlesticks",
                        start_ts=int((t13 - dt.timedelta(hours=6)).timestamp()), end_ts=int(t13.timestamp()), period_interval=60)
                cs = c.get("candlesticks", [])
                p = None
                for x in cs:
                    pr = x.get("price", {}) or {}
                    v = pr.get("close") or pr.get("previous") or (x.get("yes_bid", {}) or {}).get("close")
                    if v is not None:
                        p = float(v) / 100
                px.append((m["ticker"], p, m.get("result")))
            ps = [p for _, p, _ in px if p is not None and p > 0]
            if len(ps) < 2:
                continue
            tot = sum(ps)
            w = [p / tot for p in ps]
            rows.append(dict(series=s["ticker"], event=ev, date=day, n_markets=len(ms), priced=len(ps),
                             max_prob=max(ps), sum_prob=tot, entropy=-sum(x * math.log(x) for x in w if x > 0)))
    D = pd.DataFrame(rows).sort_values("date")
    os.makedirs(f"{RD}/results", exist_ok=True)
    D.to_csv(f"{RD}/results/kalshi_fed_events.csv", index=False)
    print(f"::notice::Kalshi Fed events: {len(D)} ({D.date.min() if len(D) else '-'}..{D.date.max() if len(D) else '-'}); "
          f"series {D.series.value_counts().to_dict() if len(D) else {}}")


if __name__ == "__main__":
    main()
