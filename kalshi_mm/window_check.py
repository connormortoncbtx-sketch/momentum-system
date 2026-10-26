"""
kalshi_mm/window_check.py -- settled P&L per contract by minute-of-hour window and favourite / long shot:  kmm windows

Uses every settled market (report.market_results), not diagnose's cache. Prints progress so a slow result lookup is
visible. Optional argument: only market days >= YYYY-MM-DD.
"""
import datetime as dt
import sys
import time

import report as R
import history as H

t0 = time.time()
print("looking up results of settled markets (new ones are fetched, ~3/s)...", flush=True)
res = R.market_results()
print(f"  {len(res)} markets with a result ({time.time() - t0:.0f}s)", flush=True)
since = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
g = {}
for f in R.rows("fills"):
    x = f.get("fill") or f
    tk = x.get("ticker")
    if tk not in res:
        continue
    if since and (H.market_day(tk) or dt.date.min) < since:
        continue
    d, yp, c = R.fill_yes(x)
    held = yp if d > 0 else 1 - yp
    try:
        m = int(str(x.get("created_time", ""))[14:16])
    except ValueError:
        continue
    for k in ((("fav " if held >= .5 else "long") + (" :45-:59" if m >= 45 else " :00-:44")),
              ("fav " if held >= .5 else "long") + " " + (":00-:14" if m < 15 else ":15-:29" if m < 30 else ":30-:44" if m < 45 else ":45-:50" if m < 51 else ":51-:59")):
        a = g.setdefault(k, [0.0, 0.0, set()])
        a[0] += d * (res[tk] - yp) * c * 100; a[1] += c; a[2].add(tk.rsplit("-", 1)[0])
print(f"\nsettled P&L per contract{' from ' + str(since) if since else ''} (city-days = distinct city-days in the group)")
for k in sorted(g):
    p, n, ev = g[k]
    print(f"  {k:18} {p / n:+7.2f}c over {n:6.0f} contracts, {len(ev):3d} city-days")
