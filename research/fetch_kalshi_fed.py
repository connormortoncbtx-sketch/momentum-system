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
    # KXFED: "Fed funds upper bound above X% after this meeting" ladders, one event per meeting (live + archived)
    mk = []
    for path, q in (("/historical/markets", dict(series_ticker="KXFED", limit=1000)),
                    ("/markets", dict(series_ticker="KXFED", status="settled", limit=1000))):
        cur = None
        while True:
            j = get(path, **(dict(q, cursor=cur) if cur else q))
            mk += [dict(m, _hist=path.startswith("/historical")) for m in j.get("markets", [])]
            cur = j.get("cursor")
            if not cur:
                break
    mk = list({m["ticker"]: m for m in mk}.values())
    print(f"KXFED markets: {len(mk)}", flush=True)
    by_ev = {}
    for m in mk:
        by_ev.setdefault(m["event_ticker"], []).append(m)
    rows = []
    for ev, ms in sorted(by_ev.items()):
        close = pd.Timestamp(max(m["close_time"] for m in ms)).tz_convert(ET)
        day = close.date()
        t13 = dt.datetime.combine(day, dt.time(13, 0), ET)
        lad = []
        for m in ms:
            k = m.get("floor_strike")
            if k is None:
                try:
                    k = float(m["ticker"].rsplit("-T", 1)[1])
                except Exception:
                    continue
            q = dict(start_ts=int((t13 - dt.timedelta(hours=8)).timestamp()), end_ts=int(t13.timestamp()), period_interval=60)
            c = get(f"/historical/markets/{m['ticker']}/candlesticks", **q) if m["_hist"] else {}
            if not c.get("candlesticks"):
                c = get(f"/series/KXFED/markets/{m['ticker']}/candlesticks", **q)
            p = None
            for x in c.get("candlesticks", []):
                pr = x.get("price", {}) or {}
                v = pr.get("close") if pr.get("close") is not None else pr.get("previous")
                if v is None:
                    yb, ya = (x.get("yes_bid", {}) or {}).get("close"), (x.get("yes_ask", {}) or {}).get("close")
                    v = (yb + ya) / 2 if yb is not None and ya is not None else None
                if v is not None:
                    p = float(v) / (100 if float(v) > 1 else 1)
            if p is not None:
                lad.append((float(k), p))
        if len(lad) < 2:
            continue
        lad.sort()
        ks, ps = [k for k, _ in lad], [min(max(p, 0), 1) for _, p in lad]
        ps = list(pd.Series(ps)[::-1].cummax()[::-1])               # P(above k) must fall with k
        bins = [1 - ps[0]] + [ps[i] - ps[i + 1] for i in range(len(ps) - 1)] + [ps[-1]]
        bins = [max(b, 0) for b in bins]
        tot = sum(bins) or 1
        w = [b / tot for b in bins]
        rows.append(dict(event=ev, date=day, strikes=len(lad), max_prob=max(w),
                         entropy=-sum(x * math.log(x) for x in w if x > 0)))
        print(f"  {day}: {len(lad)} strikes, favourite {max(w):.2f}", flush=True)
    D = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame(columns=["event", "date", "max_prob", "entropy"])
    os.makedirs(f"{RD}/results", exist_ok=True)
    D.to_csv(f"{RD}/results/kalshi_fed_events.csv", index=False)
    print(f"::notice::Kalshi Fed meetings: {len(D)}" + (f" ({D.date.min()}..{D.date.max()}); median favourite {D.max_prob.median():.2f}" if len(D) else ""))


if __name__ == "__main__":
    main()
