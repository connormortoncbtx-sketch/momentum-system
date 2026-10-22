"""
kalshi_mm/sniper.py -- PAPER shadow of the decided-bucket snipe (Oct 22, 2026). Places NO orders.

Research (research/kalshi/obs_value.py): once a station's readings show today's high already past a bucket, that
bucket is decided, yet it often still trades at ~75-90c; buying the certain side made +16 to +21c/contract in
backtests even with the free NWS feed's ~20-minute delay and whole-degree-C readings. The open question is whether
those prices are really there to buy, and in what size. This shadow answers it with live data:
  every POLL_S seconds: read api.weather.gov observations for each station since local standard-time midnight
  -> high so far = round(max(C x 1.8 + 32)); a bucket is certain NO when high so far >= its top + 1 + MARGIN, and a
  "greater than" bucket is certain YES when high so far >= its threshold + MARGIN
  -> for each certain market, read Kalshi's live order book: the price to buy the certain side and how many contracts
  are available at <= 97c; logged to LOG_DIR/sniper.jsonl (first sighting, then every 10 min while still cheap)
'kmm sniper' scores the log once markets settle: wrong calls, P&L per contract and $ available.
Throttled to 2 Kalshi requests/s so it never competes with the market maker. Public data only.
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot as B  # noqa: E402

B.READ_LIMIT, B.WRITE_LIMIT = B.RateLimit(2), B.RateLimit(1)
POLL_S, MARGIN, MAX_PX, RECHECK_S = 30, 1, 0.97, 600
LOG = os.path.join(B.LOG_DIR, "sniper.jsonl")
# series -> (settlement station, local time zone)
STATIONS = {
    "KXHIGHNY": ("KNYC", "America/New_York"), "KXHIGHCHI": ("KMDW", "America/Chicago"),
    "KXHIGHLAX": ("KLAX", "America/Los_Angeles"), "KXHIGHMIA": ("KMIA", "America/New_York"),
    "KXHIGHPHIL": ("KPHL", "America/New_York"), "KXHIGHDEN": ("KDEN", "America/Denver"),
    "KXHIGHAUS": ("KAUS", "America/Chicago"), "KXHIGHTDAL": ("KDFW", "America/Chicago"),
    "KXHIGHTATL": ("KATL", "America/New_York"), "KXHIGHTBOS": ("KBOS", "America/New_York"),
    "KXHIGHTLV": ("KLAS", "America/Los_Angeles"), "KXHIGHTHOU": ("KHOU", "America/Chicago"),
    "KXHIGHTDC": ("KDCA", "America/New_York"), "KXHIGHTPHX": ("KPHX", "America/Phoenix"),
    "KXHIGHTSATX": ("KSAT", "America/Chicago"), "KXHIGHTSEA": ("KSEA", "America/Los_Angeles"),
    "KXHIGHTMIN": ("KMSP", "America/Chicago"), "KXHIGHTOKC": ("KOKC", "America/Chicago"),
    "KXHIGHTNOLA": ("KMSY", "America/Chicago"), "KXHIGHTSFO": ("KSFO", "America/Los_Angeles"),
    "KXHIGHTEWR": ("KEWR", "America/New_York"),
}
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def log(**rec):
    with open(LOG, "a") as f:
        f.write(json.dumps({"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **rec}) + "\n")


def std_day(tz, at):
    """Local STANDARD-time date and its midnight in UTC (the official daily high runs midnight-midnight standard time)."""
    z = ZoneInfo(tz)
    jan = dt.datetime(at.year, 1, 15, tzinfo=z).utcoffset()            # standard offset (no DST)
    local_std = at.astimezone(dt.timezone.utc) + jan
    d = local_std.date()
    return d, dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc) - jan


def nws_high(station, since):
    url = f"https://api.weather.gov/stations/{station}/observations?start={since.strftime('%Y-%m-%dT%H:%M:%SZ')}&limit=500"
    req = urllib.request.Request(url, headers={"User-Agent": "kmm-sniper-shadow (research)", "Accept": "application/geo+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        feats = json.loads(r.read()).get("features", [])
    temps, newest = [], None
    for f in feats:
        p = f.get("properties", {})
        v = (p.get("temperature") or {}).get("value")
        ts = p.get("timestamp")
        if v is None or ts is None:
            continue
        t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        temps.append(v * 1.8 + 32)
        newest = t if newest is None or t > newest else newest
    return (round(max(temps)) if temps else None), newest, len(temps)


def interval(m):
    st, fl, cp = m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike")
    if st == "between" and fl is not None and cp is not None:
        return float(fl), float(cp)
    if st == "greater" and fl is not None:
        return float(fl) + 1, float("inf")
    if st == "less" and cp is not None:
        return float("-inf"), float(cp) - 1
    return None


def market_date(ticker):
    try:
        d = ticker.split("-")[1]
        return dt.date(2000 + int(d[:2]), MON.index(d[2:5]) + 1, int(d[5:7]))
    except Exception:
        return None


def main():
    k = B.Kalshi("dry")                                           # public endpoints only
    mk_cache, mk_t, seen = {}, 0.0, {}
    while True:
        t0 = time.time()
        nowu = dt.datetime.now(dt.timezone.utc)
        if time.time() - mk_t > 300:
            for s in STATIONS:
                try:
                    mk_cache[s] = k.open_markets(s)
                except Exception as e:
                    B.log.warning(f"sniper markets {s}: {e}")
            mk_t = time.time()
        for s, (station, tz) in STATIONS.items():
            day, since = std_day(tz, nowu)
            todays = [m for m in mk_cache.get(s, []) if market_date(m["ticker"]) == day]
            if not todays:
                continue
            try:
                high, newest, n = nws_high(station, since)
            except Exception as e:
                B.log.warning(f"sniper nws {station}: {e}"); continue
            if high is None:
                continue
            for m in todays:
                iv = interval(m)
                if iv is None:
                    continue
                lo, hi = iv
                side = "no" if hi != float("inf") and high >= hi + 1 + MARGIN else (
                       "yes" if hi == float("inf") and high >= lo + MARGIN else None)
                if side is None:
                    continue
                tk = m["ticker"]
                if tk in seen and time.time() - seen[tk] < RECHECK_S:
                    continue
                try:
                    y, n_ = k.book(tk)
                except Exception as e:
                    B.log.warning(f"sniper book {tk}: {e}"); continue
                # buying NO = hitting YES bids (NO price 1 - yes bid); buying YES = hitting NO bids
                opp = y if side == "no" else n_
                levels = [(round(1 - p, 4), q) for p, q in opp]          # (price of the certain side, contracts)
                cheap = [(p, q) for p, q in levels if p <= MAX_PX]
                best = levels[0][0] if levels else None
                log(event="signal" if tk not in seen else "recheck", ticker=tk, station=station, side=side,
                    high_so_far=high, bucket=[lo, hi], newest_obs=newest.isoformat() if newest else None,
                    obs_age_min=round((nowu - newest).total_seconds() / 60, 1) if newest else None, n_obs=n,
                    best_px=best, contracts_le97=round(sum(q for _, q in cheap), 2),
                    usd_edge_le97=round(sum((1 - p - 0.07 * p * (1 - p)) * q for p, q in cheap), 2), levels=levels[:6])
                seen[tk] = time.time()
        time.sleep(max(1.0, POLL_S - (time.time() - t0)))


def report():
    rows = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    first = {}
    for r in rows:
        if r.get("event") == "signal":
            first.setdefault(r["ticker"], r)
    if not first:
        print("no signals yet"); return
    k = B.Kalshi("dry")
    n = wrong = cheap_n = 0
    pnl1, cap, ages = [], 0.0, []
    for tk, r in first.items():
        try:
            res = str(k.req("GET", f"/markets/{tk}").get("market", {}).get("result") or "").lower()
        except Exception:
            res = ""
        ages.append(r.get("obs_age_min") or 0)
        if res not in ("yes", "no"):
            continue
        n += 1
        ok = res == r["side"]
        wrong += not ok
        if r.get("best_px") is not None and r["best_px"] <= MAX_PX:
            cheap_n += 1
            p = r["best_px"]
            pnl1.append((1 if ok else 0) - p - 0.07 * p * (1 - p))
            cap += r["usd_edge_le97"] if ok else -sum(p_ * q for p_, q in r.get("levels", []) if p_ <= MAX_PX)
    ages.sort()
    print(f"signals {len(first)} markets (settled {n}); wrong calls {wrong}; buyable at <= 97c when first flagged: {cheap_n}")
    if pnl1:
        print(f"1 contract at the best price: {sum(pnl1) / len(pnl1) * 100:+.1f}c per contract over {len(pnl1)} markets")
        print(f"everything offered at <= 97c when first flagged: ${cap:+,.2f} total")
    print(f"reading age when flagged: median {ages[len(ages) // 2]:.0f} min")


if __name__ == "__main__":
    report() if len(sys.argv) > 1 and sys.argv[1] == "report" else main()
