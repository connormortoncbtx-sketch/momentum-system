"""
kalshi_mm/diagnose.py -- where is the live bot losing?   sudo kmm diagnose

For every live fill (fills.jsonl): what kind of fill it was and how the price moved afterwards.
  kind      SWEEP if, in the public trade tape, the same second has >= 2 prints spanning >= 2c (an impatient taker
            walking the book -- the fills the backtest counted); otherwise DRIFT (the price came to our resting order
            gradually / one print -- typically a stale quote picked off when news moves the market; the backtest did NOT
            count these)
  markout   our side's mid move 5 and 60 minutes after the fill, in cents (+ = moved our way), from 1-minute candles
  to_settle P&L per contract to the market's result (finished markets only)
  hours     hours to close at the fill
Tables: by kind, by hours-to-close, by price, by side, by city; contract-weighted. Public endpoints only (plus your own
fill log); results cached in LOG_DIR/diag_cache.json so re-runs are quick. Read-only.
"""
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot as B      # noqa: E402
import report as R   # noqa: E402

CACHE = os.path.join(R.LOG, "diag_cache.json")


_last = [0.0]


def pace():
    """Stay well under Kalshi's public rate limit (the bot shares the account's IP)."""
    w = 0.12 - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()


def ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def load_cache():
    try:
        return json.load(open(CACHE))
    except Exception:
        return {"meta": {}, "tape": {}, "candles": {}}


def save_cache(c):
    json.dump(c, open(CACHE + ".tmp", "w")); os.replace(CACHE + ".tmp", CACHE)


def meta(k, c, tk):
    if tk not in c["meta"]:
        try:
            pace()
            m = k.req("GET", f"/markets/{tk}").get("market", {})
            c["meta"][tk] = {"close": m.get("close_time"), "result": m.get("result")}
        except Exception:
            c["meta"][tk] = {"close": None, "result": None}
    return c["meta"][tk]


def tape_at(k, c, tk, t):
    """Prints in the same second as our fill: (count, price span in cents)."""
    key = f"{tk}|{int(t.timestamp())}"
    if key not in c["tape"]:
        try:
            pace()
            d = k.req("GET", "/markets/trades", ticker=tk, min_ts=int(t.timestamp()) - 1, max_ts=int(t.timestamp()) + 1, limit=100)
            pr = [B.num(x.get("yes_price_dollars")) or B.num(x.get("yes_price")) / 100 for x in d.get("trades", [])]
            c["tape"][key] = [len(pr), round((max(pr) - min(pr)) * 100, 1) if pr else 0.0]
        except Exception:
            c["tape"][key] = [0, 0.0]
    return c["tape"][key]


def candles(k, c, tk, t0, t1):
    """{minute_ts: yes mid} for the ticker over [t0, t1] (1-minute candles, bid/ask close)."""
    key = f"{tk}|{int(t0.timestamp()) // 3600}|{int(t1.timestamp()) // 3600}"
    if key not in c["candles"]:
        out = {}
        try:
            s = tk.split("-")[0]
            pace()
            d = k.req("GET", f"/series/{s}/markets/{tk}/candlesticks", start_ts=int(t0.timestamp()), end_ts=int(t1.timestamp()),
                      period_interval=1)
            for x in d.get("candlesticks", []):
                b, a = x.get("yes_bid") or {}, x.get("yes_ask") or {}
                bb = B.num(b.get("close_dollars")) if b.get("close_dollars") is not None else B.num(b.get("close")) / 100
                aa = B.num(a.get("close_dollars")) if a.get("close_dollars") is not None else B.num(a.get("close")) / 100
                if 0 < bb < aa < 1 and aa - bb <= 0.10:
                    out[str(x.get("end_period_ts"))] = (bb + aa) / 2
        except Exception:
            pass
        c["candles"][key] = out
    return c["candles"][key]


def mid_after(cm, t, minutes):
    tgt = int(t.timestamp()) + minutes * 60
    ks = sorted(int(x) for x in cm)
    for x in ks:
        if x >= tgt:
            return cm[str(x)]
    return None


def table(rows, key, title):
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    print(f"\n{title}")
    print(f"  {'group':14} {'fills':>6} {'contracts':>9} {'markout 5m':>10} {'markout 60m':>11} {'to settle':>9} {'settled n':>9}")
    for name in sorted(g, key=str):
        rs = g[name]
        n = sum(r["c"] for r in rs)

        def w(f):
            xs = [(r[f], r["c"]) for r in rs if r.get(f) is not None]
            tot = sum(cc for _, cc in xs)
            return (sum(v * cc for v, cc in xs) / tot, tot) if tot else (float("nan"), 0)
        m5, _ = w("m5"); m60, _ = w("m60"); st, sn = w("settle")
        print(f"  {str(name):14} {len(rs):6d} {n:9.0f} {m5:+9.2f}c {m60:+10.2f}c {st:+8.2f}c {sn:9.0f}")


def main():
    k = B.Kalshi("dry")                                        # public endpoints only
    c = load_cache()
    fills = [f.get("fill") or f for f in R.rows("fills")]
    rows, t_start = [], time.time()
    for i, x in enumerate(fills):
        tk = x.get("ticker"); t = ts(x.get("created_time"))
        side, act = str(x.get("side", "")).lower(), str(x.get("action", "")).lower()
        # normalise to "we bought S at price p" where S is YES or NO
        p = R.dollars_field(x, "yes_price" if side == "yes" else "no_price") / 100
        held = side if act == "buy" else ("no" if side == "yes" else "yes")
        p_held = p if act == "buy" else 1 - p
        cnt = B.num(x.get("count_fp", x.get("count")))
        m = meta(k, c, tk)
        hours = (ts(m["close"]) - t).total_seconds() / 3600 if m.get("close") else None
        n_prints, span = tape_at(k, c, tk, t)
        cm = candles(k, c, tk, t - dt.timedelta(minutes=2), t + dt.timedelta(minutes=70))
        m5, m60 = mid_after(cm, t, 5), mid_after(cm, t, 60)
        sign = 1 if held == "yes" else -1                     # YES mid up is good for a YES holder
        res = str(m.get("result") or "").lower()
        settle = None
        if res in ("yes", "no"):
            win = (1.0 if res == "yes" else 0.0) if held == "yes" else (1.0 if res == "no" else 0.0)
            settle = (win - p_held) * 100
        yes_fill = p if side == "yes" else 1 - p
        rows.append(dict(c=cnt, kind="SWEEP" if n_prints >= 2 and span >= 2 else "DRIFT", hours=hours,
                         price=p_held, held=held.upper(), city=tk.split("-")[0].replace("KXHIGHT", "").replace("KXHIGH", ""),
                         m5=None if m5 is None else sign * (m5 - yes_fill) * 100,
                         m60=None if m60 is None else sign * (m60 - yes_fill) * 100, settle=settle))
        if i % 100 == 0:
            save_cache(c)
            print(f"  ...{i}/{len(fills)} fills ({time.time() - t_start:.0f}s)", flush=True)
    save_cache(c)
    print(f"\n{len(rows)} fills, {sum(r['c'] for r in rows):.0f} contracts. Markout = mid move our way after the fill "
          f"(a market maker wants these >= 0; consistently negative = being picked off). To settle = P&L per contract.")
    table(rows, lambda r: r["kind"], "BY KIND (SWEEP = what the backtest counted; DRIFT = gradual / single print)")
    table(rows, lambda r: ("?" if r["hours"] is None else "<6h" if r["hours"] < 6 else "6-12h" if r["hours"] < 12
                           else "12-24h" if r["hours"] < 24 else "24-48h" if r["hours"] < 48 else "48h+"), "BY HOURS TO CLOSE")
    table(rows, lambda r: "<20c" if r["price"] < .2 else "20-50c" if r["price"] < .5 else "50-80c" if r["price"] < .8 else "80c+",
          "BY PRICE PAID (for the side held)")
    table(rows, lambda r: r["held"], "BY SIDE HELD")
    table(rows, lambda r: r["city"], "BY CITY")


if __name__ == "__main__":
    main()
