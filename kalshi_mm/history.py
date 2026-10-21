"""
kalshi_mm/history.py -- trend series for the dashboard (all from the bot's own logs; no Kalshi calls).

  account_series()  hourly: account value (at the bid when logged, else mid) minus money deposited so far
                    deposits as a dated schedule DEPOSIT_HISTORY="2026-10-19 12:00=100;2026-10-20 07:00=245"
                    (falls back to the constant DEPOSITS_USD, which makes pre-deposit history look low)
  settled_days()    per market date: P&L in $ and c/contract of every fill in settled markets, city-days
  convergence()     running edge per contract over city-days, with a 2-standard-error band (the "is it working" chart)
  fills_per_day()   contracts filled per (Central) day
"""
import datetime as dt
import os
from zoneinfo import ZoneInfo

import report as R

CT = ZoneInfo("America/Chicago")
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def deposits_at():
    """function(ts) -> cumulative net deposits at that time."""
    sched = []
    for part in (os.environ.get("DEPOSIT_HISTORY") or "").split(";"):
        if "=" in part:
            t, a = part.split("=", 1)
            try:
                sched.append((dt.datetime.fromisoformat(t.strip()).replace(tzinfo=CT), float(a)))
            except ValueError:
                pass
    sched.sort()
    const = R.num(os.environ.get("DEPOSITS_USD"))
    if not sched:
        return lambda ts: const
    return lambda ts: sum(a for t, a in sched if t <= ts)


def account_series():
    dep = deposits_at()
    out, last_hour = [], None
    for s in R.rows("snapshots"):
        try:
            ts = dt.datetime.fromisoformat(s["ts"]).astimezone(CT)
        except Exception:
            continue
        hr = ts.replace(minute=0, second=0, microsecond=0)
        v = s.get("value_at_bid", s.get("value"))
        if v is None:
            continue
        if hr != last_hour:
            out.append([hr, v - dep(ts)])
            last_hour = hr
        else:
            out[-1][1] = v - dep(ts)                       # keep the hour's last reading
    return out


def market_day(ticker):
    """'KXHIGHNY-26OCT20-B67.5' -> date(2026, 10, 20)."""
    try:
        d = ticker.split("-")[1]
        return dt.date(2000 + int(d[:2]), MON[d[2:5]], int(d[5:7]))
    except Exception:
        return None


def settled_fills():
    result = R.market_results()
    out = []
    for f in R.rows("fills"):
        x = f.get("fill") or f
        tk = x.get("ticker")
        if tk not in result:
            continue
        c = R.num(x.get("count_fp", x.get("count")))
        side, act = str(x.get("side", "")).lower(), str(x.get("action", "")).lower()
        p = R.dollars_field(x, "yes_price" if side == "yes" else "no_price") / 100
        win = result[tk] if side == "yes" else 1 - result[tk]
        pnl = (win - p if act == "buy" else p - win) * c * 100 - R.dollars_field(x, "fee_cost")
        out.append(dict(day=market_day(tk), event=tk.rsplit("-", 1)[0], pnl_c=pnl, n=c))
    return out


def settled_days():
    by = {}
    for r in settled_fills():
        if r["day"] is None:
            continue
        d = by.setdefault(r["day"], {"pnl_c": 0.0, "n": 0.0, "events": set()})
        d["pnl_c"] += r["pnl_c"]; d["n"] += r["n"]; d["events"].add(r["event"])
    return [(day, v["pnl_c"] / 100, v["pnl_c"] / v["n"] if v["n"] else 0.0, len(v["events"])) for day, v in sorted(by.items())]


def convergence():
    """[(city_days, running mean c/contract, 2 s.e.)] in settlement order, per-city-day edges equally weighted."""
    ev = {}
    for r in settled_fills():
        e = ev.setdefault(r["event"], [r["day"], 0.0, 0.0])
        e[1] += r["pnl_c"]; e[2] += r["n"]
    vals = [v[1] / v[2] for v in sorted(ev.values(), key=lambda v: (v[0] or dt.date.min)) if v[2] > 0]
    out = []
    for i in range(1, len(vals) + 1):
        xs = vals[:i]; m = sum(xs) / i
        se = (sum((x - m) ** 2 for x in xs) / (i - 1)) ** 0.5 / i ** 0.5 if i > 2 else float("nan")
        out.append((i, m, 2 * se))
    return out


def fills_per_day():
    by = {}
    for f in R.rows("fills"):
        x = f.get("fill") or f
        try:
            d = dt.datetime.fromisoformat(str(x.get("created_time")).replace("Z", "+00:00")).astimezone(CT).date()
        except Exception:
            continue
        by[d] = by.get(d, 0.0) + R.num(x.get("count_fp", x.get("count")))
    return sorted(by.items())
