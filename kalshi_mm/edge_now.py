"""
kalshi_mm/edge_now.py -- the dashboard's "is it working?" numbers NOW, counting markets that are decided but not yet
settled (Kalshi pays out the next morning):   kmm edge-now

A market counts as decided when its YES bid >= 97c or ask <= 3c (or last trade beyond those): valued at 1 / 0, flagged
"provisional". Prints, for all history / current-logic trades / the clean test (EVAL_FROM), the edge per contract
weighted by city-day and by contract, ±2 standard errors, and how many city-days are provisional. Read-only.
"""
import datetime as dt
import math
import os

import report as R
import history as H


def provisional(res):
    """Add 1/0 for decided-but-unsettled markets the bot filled in; returns the set of provisional tickers."""
    import bot as B
    B.READ_LIMIT = B.RateLimit(3)                       # yield to the live bot on the shared rate limit
    k = B.Kalshi("dry")
    need = sorted({(f.get("fill") or f).get("ticker") for f in R.rows("fills")} - set(res) - {None})
    out = set()
    for tk in need:
        try:
            m = k.req("GET", f"/markets/{tk}").get("market", {})
        except Exception:
            continue
        yb = B.num(m.get("yes_bid_dollars")) or B.num(m.get("yes_bid")) / 100
        ya = B.num(m.get("yes_ask_dollars")) or B.num(m.get("yes_ask")) / 100
        lp = B.num(m.get("last_price_dollars")) or B.num(m.get("last_price")) / 100
        if yb >= 0.97 or (lp >= 0.97 and not ya):
            res[tk] = 1.0; out.add(tk)
        elif (0 < ya <= 0.03) or (0 < lp <= 0.03 and not yb):
            res[tk] = 0.0; out.add(tk)
    return out


def edge(fills, res, filt=None, since=None):
    ev = {}
    for f in fills:
        x = f.get("fill") or f
        tk = x.get("ticker")
        if tk not in res:
            continue
        if since and (H.market_day(tk) or dt.date.min) < since:
            continue
        d, yp, c = R.fill_yes(x)
        if filt and not filt(x, d, yp):
            continue
        e = ev.setdefault(tk.rsplit("-", 1)[0], [0.0, 0.0])
        e[0] += d * (res[tk] - yp) * c * 100 - R.dollars_field(x, "fee_cost"); e[1] += c
    xs = [(p / n, n) for p, n in ev.values() if n > 0]
    if len(xs) < 3:
        return len(xs), float("nan"), float("nan"), float("nan"), float("nan")
    G = len(xs)
    m1 = sum(a for a, _ in xs) / G
    se1 = math.sqrt(sum((a - m1) ** 2 for a, _ in xs) / (G - 1) / G)
    N = sum(n for _, n in xs)
    m2 = sum(a * n for a, n in xs) / N
    se2 = math.sqrt(sum(n * n * (a - m2) ** 2 for a, n in xs) / N ** 2 * G / (G - 1))
    return G, m1, 2 * se1, m2, 2 * se2


def main():
    res = R.market_results()
    print("checking decided-but-unsettled markets...", flush=True)
    prov = provisional(res)
    pdays = {tk.rsplit("-", 1)[0] for tk in prov}
    print(f"  {len(prov)} markets counted as decided (provisional), across {len(pdays)} city-days\n")
    fills = R.rows("fills")
    since = dt.date.fromisoformat(os.environ.get("EVAL_FROM", "2026-10-27"))
    print(f"  {'view':44} {'city-days':>9} {'per city-day':>15} {'per contract':>15}")
    for lab, kw in ((f"CLEAN TEST (market days from {since:%b %d})", dict(since=since)),
                    ("ALL HISTORY, current-logic trades only", dict(filt=H.current_logic)),
                    ("ALL market days since launch", {})):
        G, m1, b1, m2, b2 = edge(fills, res, **kw)
        print(f"  {lab:44} {G:9d} {m1:+7.1f}c ±{b1:4.1f} {m2:+7.1f}c ±{b2:4.1f}")
    print("\n(±2 standard errors; provisional markets are decided by price, not yet paid -- the dashboard catches up "
          "after Kalshi settles them)")


if __name__ == "__main__":
    main()
