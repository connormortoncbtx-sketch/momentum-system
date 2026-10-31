"""
kalshi_mm/history.py -- trend series for the dashboard (all from the bot's own logs; no Kalshi calls).

  account_series()  hourly: account value (at the bid when logged, else mid) minus money deposited so far
                    deposits as a dated schedule DEPOSIT_HISTORY="2026-10-19 12:00=100;2026-10-20 07:00=245"
                    (falls back to the constant DEPOSITS_USD, which makes pre-deposit history look low)
  settled_days()    per market date: P&L in $ and c/contract of every fill in settled markets, city-days
  convergence()     running edge per contract over city-days, with a 95% band clustered by market day (the "is it working" chart)
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
        v += R.num(s.get("lottery_value"))                # whole account vs deposits: the bot's value + the roll's
        if hr != last_hour:
            out.append([hr, v - dep(ts)])
            last_hour = hr
        else:
            out[-1][1] = v - dep(ts)                       # keep the hour's last reading
    return out


def recent_settled(n_days=3):
    """Settled positions of the last n market days, per market: [(day, [(ticker, net contracts held at settlement
    (+YES / -NO), contracts filled, P&L $, result 1/0)], day P&L $)] newest first. P&L counts every fill, including
    round trips, so it matches the settled-P&L table."""
    res = R.market_results()
    by = {}
    for f in R.rows("fills"):
        x = f.get("fill") or f
        tk = x.get("ticker")
        if tk not in res:
            continue
        d, yp, c = R.fill_yes(x)
        m = by.setdefault(market_day(tk), {}).setdefault(tk, [0.0, 0.0, 0.0])
        m[0] += d * c; m[1] += c; m[2] += (d * (res[tk] - yp) * c * 100 - R.dollars_field(x, "fee_cost")) / 100
    out = []
    for day in sorted((k for k in by if k), reverse=True)[:n_days]:
        rows = sorted(((tk, v[0], v[1], v[2], res[tk]) for tk, v in by[day].items()), key=lambda r: r[0])
        out.append((day, rows, sum(r[3] for r in rows)))
    return out


def deployment_series():
    """Hourly [(hour, % of the bot's account value tied up in positions at cost, % budget)] from the snapshots.
    % of value (not $) so it stays comparable as the account grows or shrinks; the budget is BUDGET_PCT of value."""
    out, last = [], None
    for s in R.rows("snapshots"):
        v, u = s.get("value"), s.get("used_cost")
        if not v or u is None:
            continue
        try:
            hr = dt.datetime.fromisoformat(s["ts"]).astimezone(CT).replace(minute=0, second=0, microsecond=0)
        except Exception:
            continue
        row = [hr, 100 * R.num(u) / v, 100 * R.num(s.get("budget")) / v if s.get("budget") else None]
        if hr != last:
            out.append(row); last = hr
        else:
            out[-1] = row
    return out


def market_day(ticker):
    """'KXHIGHNY-26OCT20-B67.5' -> date(2026, 10, 20)."""
    try:
        d = ticker.split("-")[1]
        return dt.date(2000 + int(d[:2]), MON[d[2:5]], int(d[5:7]))
    except Exception:
        return None


def current_logic(x, d, yp):
    """Would today's bot have made this fill? Favourites only (held side >= 50c, FAV_MIN) and not in the :45-:59
    quiet window (QUIET_MINUTES, on from Oct 26). Lets the old-logic history count for the trades that still happen."""
    held = yp if d > 0 else 1 - yp
    try:
        minute = int(str(x.get("created_time", ""))[14:16])
    except ValueError:
        return False
    return held >= 0.5 and not 45 <= minute <= 59


def settled_fills(filt=None):
    result = R.market_results()
    out = []
    for f in R.rows("fills"):
        x = f.get("fill") or f
        tk = x.get("ticker")
        if tk not in result:
            continue
        d, yp, c = R.fill_yes(x)
        if filt and not filt(x, d, yp):
            continue
        pnl = d * (result[tk] - yp) * c * 100 - R.dollars_field(x, "fee_cost")
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


T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13,
        20: 2.09, 25: 2.06, 30: 2.04, 40: 2.02, 60: 2.00, 120: 1.98}


def tcrit(df):
    for k in sorted(T975):
        if df <= k:
            return T975[k]
    return 1.96


def convergence(since=None, weight="cityday", filt=None):
    """[(city_days, running mean c/contract, 95% half-width, market days)] in settlement order. weight 'cityday': each
    city-day's edge per contract counts once; 'contract': city-days weighted by contracts (what the account feels).
    since: market days >= it. Oct 31: the band is clustered by MARKET DAY (cities on one date share the weather pattern
    and the market's mood, so they are not independent evidence) with a Student-t multiplier -- as kmm edge-now."""
    ev = {}
    for r in settled_fills(filt):
        if since and (r["day"] is None or r["day"] < since):
            continue
        e = ev.setdefault(r["event"], [r["day"], 0.0, 0.0])
        e[1] += r["pnl_c"]; e[2] += r["n"]
    evs = [v for v in sorted(ev.values(), key=lambda v: (v[0] or dt.date.min)) if v[2] > 0]
    out = []
    for i in range(1, len(evs) + 1):
        xs = [(v[1] / v[2], v[2] if weight == "contract" else 1.0, v[0]) for v in evs[:i]]
        W = sum(w for _, w, _ in xs); m = sum(a * w for a, w, _ in xs) / W
        u = {}
        for a, w, d in xs:
            u[d] = u.get(d, 0.0) + w * (a - m)
        G = len(u)
        hw = tcrit(G - 1) * (sum(x * x for x in u.values()) / W ** 2 * G / (G - 1)) ** 0.5 if G > 1 else float("nan")
        out.append((i, m, hw, G))
    return out


def caps(event_cap, inv_cap, size):
    """How close the bot runs to its limits, from the snapshots (positions logged every cycle).
    'blocked' = held > cap - size: one more lot would breach, so the bot only quotes sides that reduce.
    Returns dict(now=..., budget_days=[(date, avg %, max %)], market_days=[(date, city-days, blocked, peak held,
    markets blocked)], open_events=[(event, held, {ticker: held})])."""
    snaps = [s for s in R.rows("snapshots") if s.get("budget")]
    if not snaps:
        return None
    lot = R.LF.tickers(R.LOG)                       # the weekly roll is not the bot's (older snapshots included it)
    bud, ev_peak, mk_peak = {}, {}, {}
    for s in snaps:
        try:
            d = dt.datetime.fromisoformat(s["ts"]).astimezone(CT).date()
        except Exception:
            continue
        bud.setdefault(d, []).append(100 * R.num(s.get("used_cost")) / s["budget"])
        e = {}
        for t, p in (s.get("positions") or {}).items():
            if t in lot:
                continue
            mk_peak[t] = max(mk_peak.get(t, 0.0), abs(p))
            e[t.rsplit("-", 1)[0]] = e.get(t.rsplit("-", 1)[0], 0.0) + abs(p)
        for k, v in e.items():
            ev_peak[k] = max(ev_peak.get(k, 0.0), v)
    by_day = {}
    for ev, v in ev_peak.items():
        d = market_day(ev + "-x")
        r = by_day.setdefault(d, [0, 0, 0.0, 0])
        r[0] += 1; r[1] += v > event_cap - size; r[2] = max(r[2], v)
    for t, v in mk_peak.items():
        if v > inv_cap - size and market_day(t) in by_day:
            by_day[market_day(t)][3] += 1
    last = snaps[-1]
    cur = {}
    for t, p in (last.get("positions") or {}).items():
        if p and t not in lot:
            cur.setdefault(t.rsplit("-", 1)[0], {})[t] = abs(p)
    return dict(
        now=dict(used=R.num(last.get("used_cost")), budget=last["budget"], ts=last.get("ts")),
        budget_days=[(d, sum(v) / len(v), max(v)) for d, v in sorted(bud.items())],
        market_days=[(d, *v) for d, v in sorted(by_day.items(), key=lambda kv: kv[0] or dt.date.min)],
        open_events=sorted(((ev, sum(m.values()), m) for ev, m in cur.items()), key=lambda x: -x[1]))


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
