"""
kalshi_mm/study.py -- where does the bot's edge come from, and where is it given away?   kmm study [YYYY-MM-DD]

Every settled fill (all of history, or market days >= the date given), P&L per contract to settlement, cut by what
the bot knew or controlled at the time of the fill (plus the 60-minute MARKOUT: how far the market's mid moved our way
in the hour after the fill -- several times less noisy than settlement, so a lever shows up in it much sooner):
  price paid, side, city, minute of the hour, hour of day (CT), hours to the market's close, how long the order had
  rested before it filled (order age), whether other fills hit the same city-day in the previous 10 minutes (burst),
  and how far behind the best bid the order was placed.
Each row: contracts, city-days, P&L per contract, ±2 standard errors CLUSTERED BY CITY-DAY (contracts in one city-day
move together, so per-contract errors would be far too small), and the same number in the first and second half of
the data -- a lever is only worth testing if both halves agree. Read-only; prints summaries only.
"""
import datetime as dt
import math
import sys
from zoneinfo import ZoneInfo

import report as R
import history as H

CT = ZoneInfo("America/Chicago")


def ts(s):
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def close_times(tickers):
    """{ticker: close datetime} from the diagnose cache, else the public market endpoint (~3/s)."""
    import diagnose as D
    import bot as B
    c = D.load_cache()
    k = B.Kalshi("dry")
    out, n = {}, 0
    for tk in tickers:
        m = c["meta"].get(tk)
        if not m or not m.get("close"):
            try:
                D.pace()
                mm = k.req("GET", f"/markets/{tk}").get("market", {})
                m = c["meta"][tk] = {"close": mm.get("close_time"), "result": mm.get("result")}
                n += 1
            except Exception:
                continue
        out[tk] = ts(m.get("close"))
    if n:
        D.save_cache(c)
    return out


def stats(rows, field="pnl"):
    """(contracts, city-days, mean c/contract, 2 s.e. clustered by city-day). field 'pnl' (to settlement, total cents)
    or 'm60' (markout, cents per contract; fills without one are skipped)."""
    by = {}
    for r in rows:
        if field == "pnl":
            v = r["pnl"]
        elif r.get(field) is None:
            continue
        else:
            v = r[field] * r["c"]
        a = by.setdefault(r["ev"], [0.0, 0.0]); a[0] += v; a[1] += r["c"]
    N = sum(a[1] for a in by.values())
    if N <= 0:
        return 0, 0, float("nan"), float("nan")
    m = sum(a[0] for a in by.values()) / N
    G = len(by)
    if G < 3:
        return N, G, m, float("nan")
    v = sum((a[0] - m * a[1]) ** 2 for a in by.values()) / N ** 2 * G / (G - 1)
    return N, G, m, 2 * math.sqrt(v)


def main():
    since = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
    print("loading fills, results, orders...", flush=True)
    res = R.market_results()
    orders = {o.get("order_id"): o for o in R.rows("orders") if o.get("order_id")}
    fills = []
    for f in R.rows("fills"):
        x = f.get("fill") or f
        tk = x.get("ticker"); t = ts(x.get("created_time"))
        if tk not in res or t is None:
            continue
        if since and (H.market_day(tk) or dt.date.min) < since:
            continue
        d, yp, c = R.fill_yes(x)
        if c <= 0:
            continue
        fills.append(dict(x=x, tk=tk, t=t, d=d, yp=yp, c=c, ev=tk.rsplit("-", 1)[0],
                          pnl=d * (res[tk] - yp) * c * 100, held=yp if d > 0 else 1 - yp))
    fills.sort(key=lambda r: r["t"])
    print(f"  {len(fills)} settled fills; looking up close times...", flush=True)
    ct = close_times(sorted({r["tk"] for r in fills}))
    last = {}
    for r in fills:
        cl = ct.get(r["tk"])
        r["hours"] = (cl - r["t"]).total_seconds() / 3600 if cl else None
        o = orders.get(r["x"].get("order_id"))
        ot = ts(o.get("ts")) if o else None
        r["age_min"] = (r["t"] - ot).total_seconds() / 60 if ot else None
        r["behind_c"] = None
        if o and o.get("price") is not None:
            best = o.get("best_yes_bid") if o.get("side") == "yes" else o.get("best_no_bid")
            if best is not None:
                r["behind_c"] = round((float(best) - float(o["price"])) * 100)
        prev = [u for u in last.get(r["ev"], []) if (r["t"] - u).total_seconds() <= 600]
        r["burst"] = len(prev)
        last.setdefault(r["ev"], []).append(r["t"])
        r["minute"] = r["t"].minute
        r["hour_ct"] = r["t"].astimezone(CT).hour
    import diagnose as D
    import bot as B
    k, c = B.Kalshi("dry"), D.load_cache()
    print("  markouts (1-minute candles, cached after the first run)...", flush=True)
    for i, r in enumerate(fills):
        cm = D.candles(k, c, r["tk"], r["t"] - dt.timedelta(minutes=2), r["t"] + dt.timedelta(minutes=70))
        m60 = D.mid_after(cm, r["t"], 60)
        r["m60"] = None if m60 is None else r["d"] * (m60 - r["yp"]) * 100
        if i % 300 == 299:
            D.save_cache(c); print(f"    {i + 1}/{len(fills)}", flush=True)
    D.save_cache(c)
    half = fills[len(fills) // 2]["t"] if fills else None
    cur = [r for r in fills if r["held"] >= 0.5 and not 45 <= r["minute"] <= 59]   # what the bot trades today
    print(f"\nP&L per contract to settlement{' from ' + str(since) if since else ''}; ±2se clustered by city-day; "
          f"1st/2nd half split at {half:%b %d %H:%M} UTC" if half else "")
    print("ALL = every settled fill; CURRENT = favourites (>= 50c) outside :45-:59, i.e. what the bot does now.\n")

    def table(title, key, rows):
        print(title)
        print(f"  {'group':22} {'contracts':>9} {'city-days':>9} {'c/contract':>10} {'±2se':>7} {'1st half':>9} {'2nd half':>9}"
              f" {'markout60':>9} {'±2se':>7} {'1st':>7} {'2nd':>7}")
        g = {}
        for r in rows:
            kk = key(r)
            if kk is not None:
                g.setdefault(kk, []).append(r)
        for kk in sorted(g, key=lambda z: (str(type(z)), z)):
            N, G, m, se = stats(g[kk])
            a = stats([r for r in g[kk] if r["t"] < half]); b = stats([r for r in g[kk] if r["t"] >= half])
            _, _, mk, mse = stats(g[kk], "m60")
            ma = stats([r for r in g[kk] if r["t"] < half], "m60")[2]; mb = stats([r for r in g[kk] if r["t"] >= half], "m60")[2]
            print(f"  {str(kk):22} {N:9.0f} {G:9d} {m:+9.2f}c {se:6.2f}c {a[2]:+8.2f}c {b[2]:+8.2f}c"
                  f" {mk:+8.2f}c {mse:6.2f}c {ma:+6.1f}c {mb:+6.1f}c")
        print()

    for label, rows in (("ALL", fills), ("CURRENT", cur)):
        N, G, m, se = stats(rows)
        _, _, mk, mse = stats(rows, "m60")
        print(f"{label}: {m:+.2f}c ±{se:.2f}c per contract over {N:.0f} contracts, {G} city-days; 60-min markout {mk:+.2f}c ±{mse:.2f}c")
    print()
    band = lambda p: "<20c" if p < .2 else "20-50c" if p < .5 else "50-60c" if p < .6 else "60-70c" if p < .7 else "70-80c" if p < .8 else "80-90c" if p < .9 else "90c+"
    table("PRICE PAID for the side held (ALL)", lambda r: band(r["held"]), fills)
    table("SIDE HELD (CURRENT)", lambda r: "YES" if r["d"] > 0 else "NO", cur)
    table("CITY (CURRENT)", lambda r: r["tk"].split("-")[0].replace("KXHIGHT", "").replace("KXHIGH", ""), cur)
    table("HOURS TO CLOSE (CURRENT)", lambda r: None if r["hours"] is None else
          "<6h" if r["hours"] < 6 else "6-12h" if r["hours"] < 12 else "12-18h" if r["hours"] < 18 else "18-24h" if r["hours"] < 24
          else "24-36h" if r["hours"] < 36 else "36h+", cur)
    table("HOUR OF DAY, Central (CURRENT)", lambda r: f"{r['hour_ct']:02d}h", cur)
    table("MINUTE OF THE HOUR (ALL; :45-:59 now quiet)", lambda r: ":00-:14" if r["minute"] < 15 else ":15-:29" if r["minute"] < 30
          else ":30-:44" if r["minute"] < 45 else ":45-:50" if r["minute"] < 51 else ":51-:59", fills)
    table("ORDER AGE when filled (CURRENT) -- a just-placed quote filled at once = someone was waiting for it",
          lambda r: None if r["age_min"] is None else "<1 min" if r["age_min"] < 1 else "1-5 min" if r["age_min"] < 5
          else "5-30 min" if r["age_min"] < 30 else "30-120 min" if r["age_min"] < 120 else "2h+", cur)
    table("BURST: earlier fills in the same city-day within 10 min (CURRENT)",
          lambda r: "0" if r["burst"] == 0 else "1" if r["burst"] == 1 else "2-3" if r["burst"] <= 3 else "4+", cur)
    table("CENTS BEHIND THE BEST BID when placed (CURRENT)", lambda r: None if r["behind_c"] is None else
          ("<=0 (at/above best)" if r["behind_c"] <= 0 else f"{min(r['behind_c'], 5)}{'+' if r['behind_c'] >= 5 else ''}c behind"), cur)


if __name__ == "__main__":
    main()
