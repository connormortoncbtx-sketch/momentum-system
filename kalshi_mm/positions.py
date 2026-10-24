"""
kalshi_mm/positions.py -- dissect the open book right now:  sudo kmm positions

Per open position: side held, contracts, what it cost, the market's current YES bid / ask, the implied chance that the
held side wins (YES mid, or 1 - mid for NO), the sell-now value (at the bid of the held side) and the expected value
at settlement (contracts x implied chance). Grouped by market date (today = mostly decided after the afternoon high)
and city, with totals. Read-only: places and cancels nothing.
"""
import datetime as dt
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot as B  # noqa: E402
B.READ_LIMIT, B.WRITE_LIMIT = B.RateLimit(3), B.RateLimit(1)   # yield to the live bot on the shared rate limit


def collect(k=None):
    """(rows, cash) for the open book -- shared by 'kmm positions' and the dashboard."""
    k = k or B.Kalshi(B.MODE)
    pos = k.positions()
    cost = getattr(k, "cost", {}) or {}
    bal = k.balance()
    rows = []
    lot = B.LF.tickers(B.LOG_DIR)                          # the weekly roll's markets (shown apart from the bot's)
    for tk, q in pos.items():
        if not q:
            continue
        try:
            y, n = k.book(tk)
        except Exception as e:
            print(f"book {tk}: {e}"); continue
        ybid = y[0][0] if y else 0.0
        yask = 1 - n[0][0] if n else 1.0
        mid = (ybid + yask) / 2
        src, settles = "mid", None
        if yask - ybid > 0.10:                                 # dead/thin book: the mid is meaningless, use last trade
            try:
                mk = k.req("GET", f"/markets/{tk}").get("market", {})
                res = str(mk.get("result") or "").lower()
                lp = B.num(mk.get("last_price_dollars")) or B.num(mk.get("last_price")) / 100
                if res in ("yes", "no"):                       # decided, waiting for payout: worth exactly 1 or 0
                    mid, src, settles = (1.0 if res == "yes" else 0.0), "result", True
                elif str(mk.get("status", "")).lower() in ("closed", "determined", "settled", "finalized") and 0 < lp < 1:
                    mid, src, settles = lp, "last", True       # closed, result not out yet: no book to sell into
                elif 0 < lp < 1:
                    mid, src = lp, "last"
            except Exception:
                pass
        side = "YES" if q > 0 else "NO"
        c = abs(q)
        p_win = mid if q > 0 else 1 - mid                      # implied chance the held side pays $1
        # what selling now would bring; a closed market can't be sold -- it pays at settlement, so value it there
        sell = c * p_win if settles else c * (ybid if q > 0 else (n[0][0] if n else 0.0))
        m = re.match(r"(KXHIGHT?|KXLOWT?)([A-Z]+)-(\d{2}[A-Z]{3}\d{2})-(.*)", tk)
        city, day, bucket = (m.group(2), m.group(3), m.group(4)) if m else (tk, "?", "")
        rows.append(dict(city=city, day=day, bucket=bucket, side=side, n=c, cost=cost.get(tk, 0.0), sell=sell,
                         ev=c * p_win, p=p_win, spread=yask - ybid, src=src, lottery=tk in lot, ticker=tk))
    rows.sort(key=lambda r: (r["day"], r["city"], r["bucket"]))
    collect.netted = getattr(k, "netted", 0.0) or 0.0     # sure payouts Kalshi already moved into cash (see bot.positions)
    return rows, bal


collect.netted = 0.0


def kalshi_truth(k=None):
    """Kalshi's own figures, to reconcile against ours: the raw /portfolio/balance fields (in $) and, over every market
    the account has touched (settled and unsettled), Kalshi's realized P&L, fees and amount traded."""
    k = k or B.Kalshi(B.MODE)
    b = k.req("GET", "/portfolio/balance", auth=True)
    bal = {key: (B.num(v) if key.endswith("_dollars") or (isinstance(v, str) and "." in v) else B.num(v) / 100)
           for key, v in b.items() if isinstance(v, (int, float, str)) and "ts" not in key and B.num(v, None) is not None}
    tot = {"markets": 0, "realized_pnl": 0.0, "fees_paid": 0.0, "total_traded": 0.0}
    cur = None
    for _ in range(20):
        q = dict(limit=1000, settlement_status="all")
        if cur:
            q["cursor"] = cur
        d = k.req("GET", "/portfolio/positions", auth=True, **q)
        for p in d.get("market_positions", []):
            tot["markets"] += 1
            for f in ("realized_pnl", "fees_paid", "total_traded"):
                tot[f] += B.num(p[f + "_dollars"]) if p.get(f + "_dollars") is not None else B.num(p.get(f)) / 100
        cur = d.get("cursor")
        if not cur or not d.get("market_positions"):
            break
    return bal, tot


def main():
    rows, bal = collect()
    print(f"{'day':8} {'city':5} {'bucket':7} {'side':4} {'n':>6} {'cost':>7} {'chance':>7} {'sell now':>9} {'exp. value':>10}")
    for r in rows:
        flag = "  decided-loser" if r["p"] < 0.03 else ("  decided-winner" if r["p"] > 0.97 else "")
        print(f"{r['day']:8} {r['city']:5} {r['bucket']:7} {r['side']:4} {r['n']:6.1f} ${r['cost']:6.2f} {r['p'] * 100:5.0f}%{'*' if r['src'] == 'last' else ' '} "
              f"${r['sell']:8.2f} ${r['ev']:9.2f}{flag}")
    print()
    for day in sorted({r["day"] for r in rows}):
        g = [r for r in rows if r["day"] == day]
        tc, ts, te = sum(r["cost"] for r in g), sum(r["sell"] for r in g), sum(r["ev"] for r in g)
        print(f"{day}: {len(g)} positions | cost ${tc:.2f} | sell now ${ts:.2f} | expected at settlement ${te:.2f} "
              f"({te - tc:+.2f} vs cost) | decided winners {sum(r['p'] > .97 for r in g)}, losers {sum(r['p'] < .03 for r in g)}")
    net = collect.netted
    tc, ts, te = sum(r["cost"] for r in rows) - net, sum(r["sell"] for r in rows) - net, sum(r["ev"] for r in rows) - net
    print(f"(less ${net:.2f} of sure payouts on mutually exclusive buckets that Kalshi already paid into cash)")
    dep = B.num(os.environ.get("DEPOSITS_USD"))
    print(f"\nALL: cost ${tc:.2f} | sell now ${ts:.2f} | expected at settlement ${te:.2f} ({te - tc:+.2f} vs cost)")
    print(f"cash ${bal:.2f} -> account: sell-now ${bal + ts:.2f}, expected ${bal + te:.2f}"
          + (f" | vs ${dep:.2f} deposited: sell-now {bal + ts - dep:+.2f}, expected {bal + te - dep:+.2f}" if dep else ""))
    try:
        kb, kt = kalshi_truth()
        print("\nKALSHI'S OWN NUMBERS: balance endpoint " + ", ".join(f"{k} ${v:,.2f}" for k, v in kb.items())
              + f" | across {kt['markets']} markets: realized P&L ${kt['realized_pnl']:+,.2f}, fees ${kt['fees_paid']:,.2f}, "
              f"traded ${kt['total_traded']:,.2f}")
    except Exception as e:
        print(f"(Kalshi's own numbers unavailable: {e})")
    print("('expected' uses current market odds -- for today's decided buckets that is ~the settlement value;"
          " * = book too thin, chance taken from the last trade)")


if __name__ == "__main__":
    main()
