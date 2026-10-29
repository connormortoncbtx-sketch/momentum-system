"""
research/kalshi/salvage.py -- can the bot sell its dead positions for 2-3c instead of letting them go to zero?
(Oct 29, 2026; Connor: "it's the 29th and we are well past the highs ... why not just take those?")

For every settled market of the bot's 11 daily-high series in the last DAYS days (public data only):
  1. at EVAL local standard time on the market day (default 17:00, after the typical high), the CHEAP side = the side
     whose last traded price is <= 5c (if any). How often does the cheap side win anyway? (the safety question; split by
     its price 1c / 2c / 3c / 4-5c)
  2. after EVAL until the close: how many contracts did takers BUY on the cheap side at >= 2c / >= 3c / >= 4c? Each is a
     fill a resting ask at that price would have got (if first in the queue). Reported per market, capped at CAP
     contracts (a typical dead position), and x 0.28 (the bot's real/backtest fill ratio from kmm calibrate) as the
     realistic version; dollars per day across the 11 cities.
Maker sells pay no fee here (weather series: no maker fee). Output: results/kalshi_salvage.csv.
"""
import concurrent.futures as cf
import datetime as dt
import os, sys
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mm_scan as M

RD = os.environ.get("RESEARCH_DATA", "research_data")
DAYS = int(os.environ.get("DAYS") or 90)
EVAL_H = float(os.environ.get("EVAL_H") or 17)
CAP, FILL_RATIO = 20.0, 0.28
TZ = {"KXHIGHNY": "America/New_York", "KXHIGHCHI": "America/Chicago", "KXHIGHLAX": "America/Los_Angeles",
      "KXHIGHMIA": "America/New_York", "KXHIGHPHIL": "America/New_York", "KXHIGHDEN": "America/Denver",
      "KXHIGHAUS": "America/Chicago", "KXHIGHTDAL": "America/Chicago", "KXHIGHTATL": "America/New_York",
      "KXHIGHTBOS": "America/New_York", "KXHIGHTLV": "America/Los_Angeles"}
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
NOW = pd.Timestamp.now(tz="UTC")
pd.set_option("display.width", 200)


def eval_time(series, ticker):
    d = ticker.split("-")[1]
    day = dt.date(2000 + int(d[:2]), MON.index(d[2:5]) + 1, int(d[5:7]))
    z = ZoneInfo(TZ[series])
    std = dt.datetime(day.year, 1, 15, tzinfo=z).utcoffset()        # local STANDARD time, as the official day
    return pd.Timestamp(dt.datetime(day.year, day.month, day.day) - std + dt.timedelta(hours=EVAL_H), tz="UTC")


def markets(s):
    rows = M.paged("/markets", "markets", max_pages=10, series_ticker=s, status="settled") + \
           M.paged("/historical/markets", "markets", max_pages=10, series_ticker=s)
    out = []
    for m in rows:
        if m.get("result") not in ("yes", "no"):
            continue
        cl = pd.Timestamp(m.get("close_time"))
        if cl < NOW - pd.Timedelta(days=DAYS):
            continue
        out.append(dict(series=s, ticker=m["ticker"], close=cl, R=1.0 if m["result"] == "yes" else 0.0))
    return out


def one(m):
    try:
        ev = eval_time(m["series"], m["ticker"])
    except Exception:
        return None
    if ev >= m["close"]:
        return None
    tr = M.paged("/markets/trades", "trades", max_pages=10, ticker=m["ticker"],
                 min_ts=int((ev - pd.Timedelta(hours=6)).timestamp()), max_ts=int(m["close"].timestamp()))
    if not tr:
        return None
    t = pd.DataFrame([dict(ts=pd.Timestamp(x["created_time"]), p=M.f(x.get("yes_price_dollars")),
                           c=M.f(x.get("count_fp", x.get("count"))), side=x.get("taker_side")) for x in tr]).sort_values("ts")
    before = t[t.ts <= ev]
    if before.empty:
        return None
    p = float(before.p.iloc[-1])
    if p <= 0.05:
        cheap, cp = "yes", p
    elif p >= 0.95:
        cheap, cp = "no", 1 - p
    else:
        return dict(ticker=m["ticker"], series=m["series"], ev=ev, cheap=None)
    after = t[t.ts > ev]
    buys = after[after.side == cheap].copy()                       # takers buying the cheap side
    buys["cp"] = buys.p if cheap == "yes" else 1 - buys.p
    won = (m["R"] == 1.0) == (cheap == "yes")
    out = dict(ticker=m["ticker"], series=m["series"], ev=ev, cheap=cheap, cheap_px=round(cp, 2), won=won)
    for k in (2, 3, 4):
        out[f"buy_ge{k}"] = float(buys.loc[buys.cp >= k / 100 - 1e-9, "c"].sum())
    return out


def main():
    ms = []
    with cf.ThreadPoolExecutor(4) as ex:
        for r in ex.map(markets, TZ):
            ms += r
    print(f"{len(ms)} settled markets in the last {DAYS} days across {len(TZ)} series; looking at the tape after "
          f"{EVAL_H:g}:00 local", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(one, ms)):
            if r:
                rows.append(r)
            if i % 500 == 499:
                print(f"  {i + 1}/{len(ms)}", flush=True)
    D = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    D.to_csv(f"{RD}/results/kalshi_salvage.csv", index=False)
    c = D[D.cheap.notna()].copy()
    c["day"] = c.ev.dt.date
    ndays = c.day.nunique()
    print(f"\nmarkets with a side at <= 5c at {EVAL_H:g}:00 local: {len(c)} of {len(D)} ({ndays} market days)")
    print("\nSAFETY -- how often the 'dead' side won anyway, by its price at that time:")
    for lab, x in (("1c", c[c.cheap_px <= 0.01]), ("2c", c[(c.cheap_px > 0.01) & (c.cheap_px <= 0.02)]),
                   ("3c", c[(c.cheap_px > 0.02) & (c.cheap_px <= 0.03)]), ("4-5c", c[c.cheap_px > 0.03])):
        print(f"  priced {lab:5}: {len(x):5d} markets, won {int(x.won.sum()):3d} ({x.won.mean() * 100 if len(x) else float('nan'):.2f}%)")
    print("\nDEMAND -- contracts takers bought on the dead side after that time (a resting ask at that price would sell):")
    for k in (2, 3, 4):
        col = c[f"buy_ge{k}"]
        capped = np.minimum(col, CAP)
        lost = c.loc[c.won, f"buy_ge{k}"].clip(upper=CAP).sum()          # sold something that then won: pay $1 each
        usd_day = (capped.sum() * k / 100 - lost) / ndays
        print(f"  ask {k}c: markets with any {(col > 0).mean() * 100:5.1f}% | contracts/market median {col.median():.0f}, "
              f"mean {col.mean():.0f} | capped at {CAP:.0f}: ${usd_day:.2f}/day before queue, "
              f"${usd_day * FILL_RATIO:.2f}/day at the bot's 0.28 fill ratio")
    s = c.groupby(c.cheap_px.round(2)).agg(n=("won", "size"), won=("won", "mean"))
    print(f"::notice::salvage: {len(c)} dead-side markets at {EVAL_H:g}:00 local over {ndays} days; dead side won "
          f"{c.won.mean() * 100:.2f}%; takers bought >= 2c in {(c.buy_ge2 > 0).mean() * 100:.0f}% of them "
          f"(median {c.buy_ge2.median():.0f} contracts), >= 3c in {(c.buy_ge3 > 0).mean() * 100:.0f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
