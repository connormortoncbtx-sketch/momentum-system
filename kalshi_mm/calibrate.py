"""
kalshi_mm/calibrate.py -- how far does the backtest overstate the bot?   kmm calibrate [YYYY-MM-DD]   (Oct 27, 2026)

Every research number on a series the bot doesn't trade (weekend rain, new cities...) comes from the same backtest:
replay the public tape, assume a quote OFFSET behind fills whenever a sweep trades OFFSET past its first price, up to
SIZE contracts, held to settlement (research/kalshi/weather_expand.py). The bot's real fills are worse than that
(queue position, cancels, latency, others quoting the same level). This puts a number on the gap: the SAME backtest
run on the SAME markets the bot actually traded, against what it actually got.

Markets: every settled market the bot filled in (market days >= the date given, default all). Both sides filtered to
what the bot does today: the favourite (held >= 0.5) and not in the quiet window, MIN_H-MAX_H before close.
Reports, per contract, clustered by city-day: backtest edge, real edge, the paired difference, the ratio
(real / backtest) -- the haircut to apply to backtest numbers on other series -- and backtest vs real contracts per
market (capacity). Public trades are fetched once per market (only the quoting window) and the simulated fills cached
in LOG_DIR/calib_cache.json; reruns resume. Read-only; prints summaries only.
"""
import datetime as dt
import json
import math
import os
import sys

import bot as B
B.READ_LIMIT = B.RateLimit(3)                     # yield to the live bot on the shared rate limit
import report as R
import history as H
import diagnose as D

CACHE = os.path.join(R.LOG, "calib_cache.json")
OFF, SIZE, MIN_H, MAX_H = B.OFFSET, B.SIZE, B.MIN_H, B.MAX_H


def ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def quiet(t):
    return bool(B.QUIET) and B.QUIET[0] <= t.minute <= B.QUIET[1]


def tape(k, tk, close):
    """Public trades in [close - MAX_H, close - MIN_H]: [(t, yes price, contracts, taker_side)] oldest first."""
    out, cur = [], None
    for _ in range(40):
        q = dict(ticker=tk, limit=1000, min_ts=int((close - dt.timedelta(hours=MAX_H)).timestamp()),
                 max_ts=int((close - dt.timedelta(hours=MIN_H)).timestamp()))
        if cur:
            q["cursor"] = cur
        D.pace()
        d = k.req("GET", "/markets/trades", **q)
        for x in d.get("trades", []):
            p = B.num(x.get("yes_price_dollars")) or B.num(x.get("yes_price")) / 100
            if x.get("is_block_trade") or not 0 < p < 1:
                continue
            out.append((ts(x["created_time"]), p, B.num(x.get("count_fp", x.get("count"))), x.get("taker_side")))
        cur = d.get("cursor")
        if not cur or not d.get("trades"):
            break
    out.sort(key=lambda r: r[0])                   # stable: keeps the API's within-second order
    return out


def simulate(tr):
    """weather_expand.deep_fills on one market: [(t iso, direction, yes price, contracts)] for the bot's filters."""
    sweeps = {}
    for t, p, c, side in tr:
        sweeps.setdefault((t, side), []).append((p, c))
    fills = []
    for (t, side), legs in sweeps.items():
        # a sweep's first print is the one nearest the touch (prints within one second aren't reliably ordered)
        first = max(p for p, c in legs) if side == "no" else min(p for p, c in legs)
        if side == "no":       # taker sold YES -> the sweep walks the YES bid down; our bid sits OFF below the first print
            q, d, px = sum(c for p, c in legs if p <= first - OFF + 1e-9), 1, first - OFF
        else:                  # taker bought YES -> walks the ask up; our NO bid = a YES ask OFF above
            q, d, px = sum(c for p, c in legs if p >= first + OFF - 1e-9), -1, first + OFF
        if q <= 0 or not 0.03 <= px <= 0.97:
            continue
        held = px if d > 0 else 1 - px
        if held < 0.5 or quiet(t):
            continue
        fills.append((t.isoformat(), d, px, min(q, SIZE)))
    return fills


def stats(rows):
    """rows: (city-day, pnl cents total, contracts). -> contracts, city-days, c/contract, 2 s.e. clustered."""
    by = {}
    for ev, v, c in rows:
        a = by.setdefault(ev, [0.0, 0.0]); a[0] += v; a[1] += c
    N = sum(a[1] for a in by.values())
    if N <= 0:
        return 0, 0, float("nan"), float("nan")
    m = sum(a[0] for a in by.values()) / N
    G = len(by)
    se = math.sqrt(sum((a[0] - m * a[1]) ** 2 for a in by.values()) / N ** 2 * G / (G - 1)) if G > 2 else float("nan")
    return N, G, m, 2 * se


def main():
    since = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
    res = R.market_results()
    real = []
    for f in R.rows("fills"):
        x = f.get("fill") or f
        tk, t = x.get("ticker"), x.get("created_time")
        if tk not in res or not t:
            continue
        if since and (H.market_day(tk) or dt.date.min) < since:
            continue
        d, yp, c = R.fill_yes(x)
        t = ts(t)
        if c <= 0 or (yp if d > 0 else 1 - yp) < 0.5 or quiet(t):
            continue
        real.append((tk, t, d, yp, c))
    tickers = sorted({r[0] for r in real})
    print(f"{len(real)} real fills (favourite, outside the quiet window) in {len(tickers)} settled markets; "
          f"backtest: {OFF * 100:.0f}c behind, {SIZE}-lot, {MIN_H:g}-{MAX_H:g}h before close", flush=True)
    try:
        cache = json.load(open(CACHE))
    except Exception:
        cache = {}
    k, dc = B.Kalshi("dry"), D.load_cache()
    todo = [tk for tk in tickers if tk not in cache]
    for i, tk in enumerate(todo):
        m = D.meta(k, dc, tk)
        if not m.get("close"):
            continue
        try:
            cache[tk] = simulate(tape(k, tk, ts(m["close"])))
        except Exception as e:
            print(f"  {tk}: {type(e).__name__} -- skipped this run", flush=True)
            continue
        if i % 25 == 24 or i == len(todo) - 1:
            json.dump(cache, open(CACHE + ".tmp", "w")); os.replace(CACHE + ".tmp", CACHE); D.save_cache(dc)
            print(f"  tape {i + 1}/{len(todo)}", flush=True)
    D.save_cache(dc)
    tickers = [tk for tk in tickers if tk in cache]
    ev = lambda tk: tk.rsplit("-", 1)[0]
    R_rows = [(ev(tk), d * (res[tk] - yp) * c * 100, c) for tk, t, d, yp, c in real if tk in cache]
    S_rows = [(ev(tk), d * (res[tk] - px) * c * 100, c) for tk in tickers for _, d, px, c in cache[tk]]
    rN, rG, rm, rse = stats(R_rows)
    sN, sG, sm, sse = stats(S_rows)
    # paired: per city-day, real minus backtest c/contract, averaged over city-days with both (equal weight)
    agg = {}
    for key, rows in (("r", R_rows), ("s", S_rows)):
        for e, v, c in rows:
            a = agg.setdefault(e, {}).setdefault(key, [0.0, 0.0]); a[0] += v; a[1] += c
    diffs = [a["r"][0] / a["r"][1] - a["s"][0] / a["s"][1] for a in agg.values() if "r" in a and "s" in a and a["s"][1] > 0]
    md = sum(diffs) / len(diffs) if diffs else float("nan")
    dse = 2 * math.sqrt(sum((x - md) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs)) if len(diffs) > 2 else float("nan")
    print(f"\nSAME {len(tickers)} MARKETS, per contract, ±2se clustered by city-day:")
    print(f"  backtest   {sm:+6.2f}c ±{sse:.2f}c   {sN:7.0f} contracts ({sN / max(len(tickers), 1):.1f}/market), {sG} city-days")
    print(f"  real       {rm:+6.2f}c ±{rse:.2f}c   {rN:7.0f} contracts ({rN / max(len(tickers), 1):.1f}/market), {rG} city-days")
    print(f"  real - backtest, paired by city-day: {md:+.2f}c ±{dse:.2f}c over {len(diffs)} city-days")
    if sm > 0:
        print(f"  ratio real/backtest: {rm / sm:.2f}   (capacity: real contracts = {rN / sN:.2f} x backtest)")
    print("\nby city (backtest vs real, c/contract):")
    cities = sorted({e.split("-")[0] for e in agg})
    for cty in cities:
        s = stats([r for r in S_rows if r[0].startswith(cty + "-")]); r = stats([x for x in R_rows if x[0].startswith(cty + "-")])
        print(f"  {cty.replace('KXHIGHT', '').replace('KXHIGH', ''):6} backtest {s[2]:+6.2f}c ({s[0]:5.0f})   real {r[2]:+6.2f}c ({r[0]:5.0f})")
    print(f"\nBACKTEST_HAIRCUT real {rm:+.2f} backtest {sm:+.2f} diff {md:+.2f}±{dse:.2f} markets {len(tickers)}")


if __name__ == "__main__":
    main()
