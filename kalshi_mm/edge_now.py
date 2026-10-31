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
    if getattr(B.READ_LIMIT, "gap", 0) < 1 / 3:          # yield to the live bot (keep a slower limit, e.g. the dashboard's)
        B.READ_LIMIT = B.RateLimit(3)
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


# Oct 29, 2026 (Connor: "why 2 standard deviations?"). Three honest-ish refinements, all read-only:
#  1. clustered by MARKET DAY as well as by city-day: cities on one date share the national weather pattern and the same
#     market mood, so they are not independent pieces of evidence (Oct 29 was good almost everywhere at once);
#  2. the multiplier is Student-t 97.5% for the number of clusters (with few clusters 2.0 is too small);
#  3. a SHRUNK per-contract estimate: the research prior (+2.5c per contract, sd 2c) blended with the live number by
#     their precisions (live precision from the market-day error) -- a hot or cold week moves it only as far as it earns.
PRIOR_C, PRIOR_SD = float(os.environ.get("EDGE_PRIOR_C", "2.5")), float(os.environ.get("EDGE_PRIOR_SD", "2.0"))
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13,
        20: 2.09, 25: 2.06, 30: 2.04, 40: 2.02, 60: 2.00, 120: 1.98}


def tcrit(df):
    if df < 1:
        return float("nan")
    for k in sorted(T975):
        if df <= k:
            return T975[k]
    return 1.96


def clustered(xs, key):
    """xs: [(edge c/contract of a city-day, weight, cluster key)] -> (mean, half-width at t 97.5%, clusters)."""
    W = sum(w for _, w, _ in xs)
    m = sum(a * w for a, w, _ in xs) / W
    u = {}
    for a, w, g in xs:
        u[key(g)] = u.get(key(g), 0.0) + w * (a - m)
    G = len(u)
    if G < 2:
        return m, float("nan"), G
    se = math.sqrt(sum(v * v for v in u.values()) / W ** 2 * G / (G - 1))
    return m, tcrit(G - 1) * se, G


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
        e = ev.setdefault(tk.rsplit("-", 1)[0], [0.0, 0.0, H.market_day(tk)])
        e[0] += d * (res[tk] - yp) * c * 100 - R.dollars_field(x, "fee_cost"); e[1] += c
    xs = [(p / n, n, (k, day)) for k, (p, n, day) in ev.items() if n > 0]
    if len(xs) < 3:
        return None
    ones = [(a, 1.0, g) for a, _, g in xs]
    out = {"G": len(xs)}
    out["cd"] = clustered(ones, lambda g: g[0])          # per city-day, clustered by city-day
    out["cd_day"] = clustered(ones, lambda g: g[1])      # per city-day, clustered by market day
    out["ct"] = clustered(xs, lambda g: g[0])            # per contract, clustered by city-day
    out["ct_day"] = clustered(xs, lambda g: g[1])        # per contract, clustered by market day
    m, hw, D = out["ct_day"]
    se = hw / tcrit(D - 1) if D > 1 else float("nan")
    if se == se and se > 0:
        wp, wl = 1 / PRIOR_SD ** 2, 1 / se ** 2
        out["shrunk"] = ((PRIOR_C * wp + m * wl) / (wp + wl), 1.96 / math.sqrt(wp + wl), wl / (wp + wl))
    else:
        out["shrunk"] = (PRIOR_C, 1.96 * PRIOR_SD, 0.0)
    return out


def compute():
    """{'prov': n, 'pdays': n, 'since': date, 'rows': [(label, edge() dict or None)]} -- used by main() and the dashboard."""
    res = R.market_results()
    prov = provisional(res)
    fills = R.rows("fills")
    since = dt.date.fromisoformat(os.environ.get("EVAL_FROM", "2026-10-27"))
    rows = [(lab, edge(fills, res, **kw)) for lab, kw in (
        (f"CLEAN TEST (market days from {since:%b %d})", dict(since=since)),
        ("ALL HISTORY, current-logic trades only", dict(filt=H.current_logic)),
        ("ALL market days since launch", {}))]
    return dict(prov=len(prov), pdays=len({tk.rsplit("-", 1)[0] for tk in prov}), since=since, rows=rows)


def main():
    print("checking decided-but-unsettled markets...", flush=True)
    c = compute()
    print(f"  {c['prov']} markets counted as decided (provisional), across {c['pdays']} city-days\n")
    print(f"  {'view':40} {'city-days':>9} {'days':>4} {'per city-day':>14} {'by day':>7} {'per contract':>14} {'by day':>7}")
    for lab, o in c["rows"]:
        if not o:
            print(f"  {lab:40} too few city-days"); continue
        (m1, b1, _), (_, b1d, D), (m2, b2, _), (_, b2d, _) = o["cd"], o["cd_day"], o["ct"], o["ct_day"]
        print(f"  {lab:40} {o['G']:9d} {D:4d} {m1:+7.1f}c ±{b1:4.1f} ±{b1d:5.1f} {m2:+7.1f}c ±{b2:4.1f} ±{b2d:5.1f}")
        sm, sb, wt = o["shrunk"]
        print(f"  {'':40} {'':9} {'':4} shrunk per contract (research prior {PRIOR_C:+.1f}c ±{1.96 * PRIOR_SD:.1f}): "
              f"{sm:+.1f}c ±{sb:.1f}  (live data carries {wt:.0%} of it)")
    print("\n±: 95% band (Student-t for the number of clusters). First ± clusters by city-day; 'by day' clusters by market "
          "day -- the honest one, since cities on one date move together. Provisional markets are decided by price, not "
          "yet paid.")


if __name__ == "__main__":
    main()
