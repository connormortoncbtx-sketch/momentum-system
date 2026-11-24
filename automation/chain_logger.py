"""
automation/chain_logger.py
==========================
Same-day option chain logger (Connor, Nov 24, 2026: "I'd rather just start collecting this data via Alpaca"). Builds
our own history of what same-day premium actually pays, so the thin-credit rule, $1 vs wider wings, other entry times
and real trading costs can be tested on our own data instead of a ThetaData month.

Runs as one long job started by the exact-time dispatcher's 09:15 CT live-stack trigger (GitHub cron runs hours late:
shadow #11 never caught a single slot from Nov 15-24 because of it). Every 15 minutes from 10:00 to 15:45 ET, for SPY,
QQQ and XSP same-day expiries, it takes one chain snapshot and records DERIVED values only (public repo; raw vendor
quotes are not redistributed):
  spot (XSP by put-call parity), ATM straddle mid / natural, and for each k in K_GRID (shorts k implied moves out) and
  each wing in WINGS ($): iron condor, put spread and call spread credits at natural and mid, the strikes, and the
  bid-ask width of the four legs as a share of the condor's mid credit.
Settlement (the 4pm close vs the strikes) is computed later in research from daily closes, so nothing is kept open.
It also starts shadow #11 (automation/shadow_premium.py) inside its noon / 2pm / 3pm windows, which its own cron misses.
Output: data/chain_log/YYYY-MM-DD.csv.gz (one row per underlying x time x k x wing).
Keys: ALPACA_API_KEY / ALPACA_SECRET_KEY (data; paper key is enough). `--once` takes a single snapshot now (testing).
"""
import csv
import datetime as dt
import gzip
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")
OUT = ROOT / "data" / "chain_log"
DATA, TRADE = "https://data.alpaca.markets", "https://paper-api.alpaca.markets"
UNDS = ("SPY", "QQQ", "XSP")
K_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)
WINGS = (1, 2, 5)
FIRST, LAST = dt.time(10, 0), dt.time(15, 45)
SHADOW_AT = (dt.time(12, 5), dt.time(14, 5), dt.time(15, 5))
FIELDS = ["date", "time", "und", "feed", "spot", "straddle_mid", "straddle_nat", "k", "wing", "ps", "cs",
          "condor_nat", "condor_mid", "put_nat", "put_mid", "call_nat", "call_mid", "leg_spread_share", "n_quotes"]


def get(base, path, **q):
    h = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
         "Accept": "application/json"}
    url = base + path + ("?" + urllib.parse.urlencode(q) if q else "")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GET {path} -> {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")


def chain(und, day):
    """{(type 'C'/'P', strike): (bid, ask)} for today's expiry, near the money; first feed that answers."""
    last = None
    for feed in ("opra", "indicative"):
        out, tok = {}, None
        try:
            while True:
                q = dict(feed=feed, limit=1000, expiration_date=day.isoformat())
                if tok:
                    q["page_token"] = tok
                js = get(DATA, f"/v1beta1/options/snapshots/{und}", **q)
                for sym, s in (js.get("snapshots") or {}).items():
                    lq = s.get("latestQuote") or {}
                    bp, ap = float(lq.get("bp") or 0), float(lq.get("ap") or 0)
                    if ap > 0:
                        tail = sym[-15:]                       # yymmdd + C/P + 8-digit strike x1000
                        out[(tail[6], int(tail[7:]) / 1000)] = (bp, ap)
                tok = js.get("next_page_token")
                if not tok:
                    break
            if out:
                return out, feed
            last = f"{feed}: empty"
        except Exception as e:
            last = f"{feed}: {str(e)[:120]}"
    raise RuntimeError(f"no chain for {und} ({last})")


def spot_of(und, Q):
    """Put-call parity at the strike where call and put mids are closest (works for XSP, which has no stock quote)."""
    ks = [k for (t, k) in Q if t == "C" and ("P", k) in Q]
    mid = lambda x: (x[0] + x[1]) / 2
    par = {k: mid(Q[("C", k)]) - mid(Q[("P", k)]) for k in ks}
    k0 = min(par, key=lambda k: abs(par[k]))
    return k0 + par[k0]


def rows_for(und, now, Q, feed):
    S = spot_of(und, Q)
    ks = sorted({k for (t, k) in Q if ("C", k) in Q and ("P", k) in Q})
    atm = min(ks, key=lambda k: abs(k - S))
    c, p = Q[("C", atm)], Q[("P", atm)]
    M = (c[0] + c[1]) / 2 + (p[0] + p[1]) / 2
    nat_str = c[0] + p[0]
    out = []
    for k in K_GRID:
        lo, hi = S - k * M, S + k * M
        puts = [x for x in ks if x <= lo and ("P", x) in Q]
        calls = [x for x in ks if x >= hi and ("C", x) in Q]
        if not puts or not calls:
            continue
        ps, cs = max(puts), min(calls)
        for w in WINGS:
            lp, lc = ("P", ps - w), ("C", cs + w)
            if lp not in Q or lc not in Q:
                continue
            sp, sc_, bp_, bc = Q[("P", ps)], Q[("C", cs)], Q[lp], Q[lc]
            put_nat, put_mid = sp[0] - bp_[1], (sp[0] + sp[1]) / 2 - (bp_[0] + bp_[1]) / 2
            call_nat, call_mid = sc_[0] - bc[1], (sc_[0] + sc_[1]) / 2 - (bc[0] + bc[1]) / 2
            cmid = put_mid + call_mid
            spread = sum(a - b for b, a in (sp, sc_, bp_, bc))
            out.append(dict(date=now.date().isoformat(), time=f"{now:%H:%M}", und=und, feed=feed, spot=round(S, 3),
                            straddle_mid=round(M, 4), straddle_nat=round(nat_str, 4), k=k, wing=w, ps=ps, cs=cs,
                            condor_nat=round(put_nat + call_nat, 4), condor_mid=round(cmid, 4),
                            put_nat=round(put_nat, 4), put_mid=round(put_mid, 4), call_nat=round(call_nat, 4),
                            call_mid=round(call_mid, 4),
                            leg_spread_share=round(spread / cmid, 4) if cmid > 0 else None, n_quotes=len(Q)))
    return out


def snapshot(now, day=None):
    rows, notes = [], []
    for und in UNDS:
        try:
            Q, feed = chain(und, day or now.date())
            r = rows_for(und, now, Q, feed)
            rows += r
            notes.append(f"{und} {len(Q)} quotes ({feed}), {len(r)} rows")
        except Exception as e:
            notes.append(f"{und} error: {str(e)[:120]}")
    return rows, notes


def save(rows, day):
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{day.isoformat()}.csv.gz"
    old = gzip.open(f, "rt").read() if f.exists() else ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    if not old:
        w.writeheader()
    w.writerows(rows)
    with gzip.open(f, "wt") as fh:
        fh.write(old + buf.getvalue())


def market_open():
    try:
        return bool(get(TRADE, "/v2/clock").get("is_open"))
    except Exception:
        return True


def main():
    if not os.environ.get("ALPACA_API_KEY"):
        print("::notice::chain logger: no ALPACA_API_KEY -- skipped"); return
    if "--once" in sys.argv:
        now = dt.datetime.now(ET)
        day = now.date()
        if now.time() >= dt.time(16, 0) or day.weekday() >= 5:      # after the close: test on the next expiry
            day += dt.timedelta(days=1)
            while day.weekday() >= 5:
                day += dt.timedelta(days=1)
        rows, notes = snapshot(now, day)
        print("::notice::chain logger test: " + "; ".join(notes))
        if rows:
            r = [x for x in rows if x["k"] == 1.0 and x["wing"] == 1]
            print("::notice::k=1, $1 wings: " + "; ".join(f"{x['und']} condor nat {x['condor_nat']} mid {x['condor_mid']} "
                                                         f"spread/credit {x['leg_spread_share']}" for x in r))
        return
    now = dt.datetime.now(ET)
    if now.time() > LAST or not market_open():
        print(f"::notice::chain logger: nothing to do ({now:%H:%M} ET, market open: {market_open()})"); return
    marks = [dt.datetime.combine(now.date(), dt.time(h, m), ET) for h in range(10, 16) for m in (0, 15, 30, 45)]
    marks = [m for m in marks if FIRST <= m.time() <= LAST]
    shadow = [dt.datetime.combine(now.date(), t, ET) for t in SHADOW_AT]
    events = sorted([(m, "snap") for m in marks] + [(s, "shadow") for s in shadow])
    done, total = 0, 0
    for when, kind in events:
        if dt.datetime.now(ET) > when + dt.timedelta(minutes=10):
            continue                                            # started late: skip what has passed
        time.sleep(max(0.0, (when - dt.datetime.now(ET)).total_seconds()))
        if kind == "shadow":
            r = subprocess.run([sys.executable, str(ROOT / "automation" / "shadow_premium.py")], capture_output=True, text=True)
            print(f"shadow #11 at {when:%H:%M}: exit {r.returncode}; " + " | ".join(
                l for l in (r.stdout + r.stderr).splitlines() if "::" in l or "ERROR" in l)[-400:])
            continue
        rows, notes = snapshot(dt.datetime.now(ET))
        save(rows, now.date())
        done += 1; total += len(rows)
        print(f"{when:%H:%M} " + "; ".join(notes), flush=True)
    print(f"::notice::chain logger {now.date()}: {done} snapshots, {total} rows")


if __name__ == "__main__":
    main()
