"""
automation/live_condor.py
=========================
REAL-MONEY same-day XSP iron condor in Connor's live Alpaca account (Connor, Nov 24, 2026: "I want to try this in the
live account"; 1 contract; dry run Wed Nov 25, live from Mon Nov 30).

Trade (the 2pm afternoon condor of docs/2026-10_strategy_research.md, small-account form):
  underlying  XSP (mini-SPX): European, CASH-settled -> no early exercise, no assignment, no overnight shares, so the
              loss is capped at width - credit (the "account goes negative" path of SPY options does not exist)
  entry       14:00-14:30 ET, same-day expiry (XSP has daily expiries), held to expiry (no stop: hold ~= stop in the
              backtest, and a cash-settled hold avoids paying the spread twice)
  strikes     shorts 1 implied move out (ATM straddle mid at entry), long wings $1 further out
  skip        when SPY's high-low range so far today >= 1.4x its 20-day mean daily range (pre-registered Nov 15 rule,
              top ~10% of days by 2pm: kept 100-118% of the P&L, removed ~1/3 of max-loss trades)
              and when the natural credit is < $0.05 (nothing to sell)
  size        CONDOR_STAKE_PCT (e.g. 30): contracts = floor(pct x equity / today's max loss per contract), at least 1,
              at most CONDOR_MAX (default 5; a ceiling while the live record is short); skipped if the max loss per
              contract is outside 40-98% of the width (bad quotes). Without CONDOR_STAKE_PCT: CONDOR_CONTRACTS (default 1).
              Skipped if options buying power can't cover it.
  order       one 4-leg limit at the mid credit, stepping $0.01 toward the live natural every 45 s, never past natural

Modes (GitHub variable CONDOR_LIVE): "yes" places the order; "probe" places an UNFILLABLE order (credit $0.95 on a $1
wide condor) and cancels it, to check Alpaca accepts the order type in this account; anything else = dry run (plan
only). Every mode sends the plan to the phone (ntfy).
Public repo: the Actions log and git get status words and RATIOS only (credit/width, return on risk) -- never dollar
amounts, strikes, quantities or positions. Ledger: data/live_condor_ledger.jsonl (ratios only).
Keys: ALPACA_LIVE_KEY / ALPACA_LIVE_SECRET (trading + quotes), ALPACA_API_KEY / ALPACA_SECRET_KEY (data fallback).
"""
import datetime as dt
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("live_condor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
ET = ZoneInfo("America/New_York")
LEDGER = ROOT / "data" / "live_condor_ledger.jsonl"
TRADE, DATA_API = "https://api.alpaca.markets", "https://data.alpaca.markets"
UND, PROXY = "XSP", "SPY"
K_MOVE, WING = 1.0, 1.0
ENTRY_FROM, ENTRY_TO = dt.time(13, 58), dt.time(14, 30)
RANGE_CUT, MIN_NAT = 1.4, 0.05
# Floor rule (Connor, Nov 24): below this account equity one contract is far above the ~30% stake -> pause and ask.
FLOOR = float(os.environ.get("CONDOR_FLOOR") or 180)
STEP, WAIT_S, MAX_STEPS = 0.01, 45, 12


# ------------------------------------------------------------------ http
def _keys(live=True):
    return (os.environ["ALPACA_LIVE_KEY"], os.environ["ALPACA_LIVE_SECRET"]) if live else \
        (os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_SECRET_KEY", ""))


def api(method, path, body=None, base=TRADE, live=True, **q):
    k, s = _keys(live)
    url = base + path + ("?" + urllib.parse.urlencode(q) if q else "")
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s,
                                          "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            t = r.read()
            return json.loads(t) if t else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")


def option_quotes(syms):
    """{symbol: (bid, ask)} from the first key/feed that returns every symbol with an ask > 0."""
    last = None
    for live in (True, False):
        if not _keys(live)[0]:
            continue
        for feed in ("opra", "indicative"):
            try:
                q = api("GET", "/v1beta1/options/quotes/latest", base=DATA_API, live=live,
                        symbols=",".join(syms), feed=feed).get("quotes", {})
                out = {s: (float(v["bp"]), float(v["ap"])) for s, v in q.items()}
                if all(s in out and out[s][1] > 0 for s in syms):
                    return out, f"{'live' if live else 'paper'}-key/{feed}"
                last = f"{feed}: missing {len([s for s in syms if s not in out])}"
            except Exception as e:
                last = f"{feed}: {str(e)[:120]}"
    raise RuntimeError(f"no option quotes ({last})")


def stock_bars(sym, tf, start, end=None):
    q = dict(symbols=sym, timeframe=tf, start=start, limit=10000, adjustment="raw", feed="sip")
    if end:
        q["end"] = end
    for live in (True, False):
        for feed in ("sip", "iex"):
            try:
                q["feed"] = feed
                b = api("GET", "/v2/stocks/bars", base=DATA_API, live=live, **q).get("bars", {}).get(sym, [])
                if b:
                    return b
            except Exception:
                continue
    return []


# ------------------------------------------------------------------ signal pieces
def range_ratio(today):
    """SPY high-low so far today / mean daily high-low of the prior 20 sessions."""
    d = stock_bars(PROXY, "1Day", (today - dt.timedelta(days=45)).isoformat())
    d = [b for b in d if b["t"][:10] < today.isoformat()][-20:]
    m = stock_bars(PROXY, "1Min", dt.datetime.combine(today, dt.time(9, 30), ET).isoformat())
    m = [b for b in m if b["t"][:10] == today.isoformat()]
    if len(d) < 15 or len(m) < 30:
        return None
    mean_rng = sum(b["h"] - b["l"] for b in d) / len(d)
    return (max(b["h"] for b in m) - min(b["l"] for b in m)) / mean_rng


def contracts_today(today):
    out, tok = {}, None
    while True:
        q = dict(underlying_symbols=UND, expiration_date=today.isoformat(), limit=1000)
        if tok:
            q["page_token"] = tok
        js = api("GET", "/v2/options/contracts", **q)
        for c in js.get("option_contracts") or []:
            if c.get("tradable"):
                out[(c["type"][0].upper(), float(c["strike_price"]))] = c["symbol"]
        tok = js.get("next_page_token")
        if not tok:
            return out


def price_condor(today):
    """XSP spot by put-call parity, implied move from the ATM straddle, then the four legs and their credits."""
    C = contracts_today(today)
    if not C:
        raise RuntimeError("no tradable XSP contracts expiring today")
    spy = stock_bars(PROXY, "1Min", (dt.datetime.now(ET) - dt.timedelta(minutes=10)).isoformat())
    ref = spy[-1]["c"] if spy else None
    strikes = sorted({k for (_, k) in C if ("C", k) in C and ("P", k) in C})
    if ref:                                                   # XSP ~ SPX/10 trades within ~1% of SPY
        strikes = [k for k in strikes if abs(k / ref - 1) < 0.015] or strikes
    mid_k = strikes[len(strikes) // 2]
    near = sorted(strikes, key=lambda k: abs(k - mid_k))[:15]
    q, feed = option_quotes([C[("C", k)] for k in near] + [C[("P", k)] for k in near])
    mid = lambda s: (q[s][0] + q[s][1]) / 2
    par = {k: mid(C[("C", k)]) - mid(C[("P", k)]) for k in near}
    k0 = min(par, key=lambda k: abs(par[k]))
    spot = k0 + par[k0]
    atm = min(near, key=lambda k: abs(k - spot))
    move = mid(C[("C", atm)]) + mid(C[("P", atm)])
    ps = max(k for k in strikes if k <= spot - K_MOVE * move) if any(k <= spot - K_MOVE * move for k in strikes) else None
    cs = min(k for k in strikes if k >= spot + K_MOVE * move) if any(k >= spot + K_MOVE * move for k in strikes) else None
    allk = sorted({k for (_, k) in C})
    if ps is None or cs is None or ("P", ps - WING) not in C or ("C", cs + WING) not in C:
        # wings must be listed exactly $1 out; widen the search to the full strike list
        ps = max([k for k in allk if k <= spot - K_MOVE * move and ("P", k) in C and ("P", k - WING) in C], default=None)
        cs = min([k for k in allk if k >= spot + K_MOVE * move and ("C", k) in C and ("C", k + WING) in C], default=None)
        if ps is None or cs is None:
            raise RuntimeError("no $1-wide wings listed around 1 implied move")
    legs = {"short_put": C[("P", ps)], "long_put": C[("P", ps - WING)],
            "short_call": C[("C", cs)], "long_call": C[("C", cs + WING)]}
    lq, feed2 = option_quotes(list(legs.values()))
    b = lambda n: lq[legs[n]][0]; a = lambda n: lq[legs[n]][1]
    natural = b("short_put") + b("short_call") - a("long_put") - a("long_call")
    midc = sum((1 if n.startswith("short") else -1) * (lq[legs[n]][0] + lq[legs[n]][1]) / 2 for n in legs)
    return dict(spot=round(spot, 2), move=round(move, 3), ps=ps, cs=cs, legs=legs, natural=round(natural, 2),
                mid=round(midc, 2), feed=feed2)


# ------------------------------------------------------------------ orders
def leg_body(legs):
    return [{"symbol": legs["long_put"], "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
            {"symbol": legs["short_put"], "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
            {"symbol": legs["short_call"], "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
            {"symbol": legs["long_call"], "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"}]


def place(legs, qty, credit):                     # negative limit = credit (the paper account's verified convention)
    return api("POST", "/v2/orders", {"order_class": "mleg", "qty": str(qty), "type": "limit", "time_in_force": "day",
                                      "limit_price": f"{-credit:.2f}", "legs": leg_body(legs)})


def cancel(oid):
    try:
        api("DELETE", f"/v2/orders/{oid}")
    except RuntimeError:
        pass
    time.sleep(2)
    return api("GET", f"/v2/orders/{oid}", nested="true")


def credit_of(st):
    lg = st.get("legs") or []
    return round(sum((1 if l.get("side") == "sell" else -1) * float(l.get("filled_avg_price") or 0) for l in lg), 4)


def work_order(e, qty):
    """Mid -> natural in $0.01 steps; returns (status, credit, steps)."""
    price, steps = max(e["mid"], e["natural"]), 0
    while True:
        o = place(e["legs"], qty, price)
        t0 = time.time()
        while time.time() - t0 < WAIT_S:
            time.sleep(5)
            st = api("GET", f"/v2/orders/{o['id']}", nested="true")
            if st.get("status") == "filled":
                return "filled", credit_of(st), steps
            if st.get("status") in ("rejected", "canceled", "expired"):
                return f"order {st.get('status')}", None, steps
        st = cancel(o["id"])
        if st.get("status") == "filled":
            return "filled", credit_of(st), steps
        steps += 1
        try:
            q, _ = option_quotes(list(e["legs"].values()))
            L = e["legs"]
            e["natural"] = round(q[L["short_put"]][0] + q[L["short_call"]][0] - q[L["long_put"]][1] - q[L["long_call"]][1], 2)
        except Exception:
            pass
        nxt = round(max(e["natural"], price - STEP), 2)
        if steps > MAX_STEPS or e["natural"] < MIN_NAT or nxt == price:
            return "no fill at natural", None, steps
        price = nxt


# ------------------------------------------------------------------ reconcile (ratios only)
def reconcile(today, notify):
    done = set()
    if LEDGER.exists():
        done = {json.loads(l)["date"] for l in LEDGER.read_text().splitlines() if l.strip() and "ror" in json.loads(l)}
    pend = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()] if LEDGER.exists() else []
    pend = [r for r in pend if r.get("status") == "filled" and r["date"] < today.isoformat() and r["date"] not in done]
    for r in pend:
        acts = api("GET", "/v2/account/activities", after=f"{r['date']}T00:00:00Z", until=f"{r['date']}T23:59:59Z",
                   page_size=100)
        acts = [a for a in acts if UND in str(a.get("symbol", ""))]
        fills = [a for a in acts if a.get("activity_type") == "FILL"]
        other = [a for a in acts if a.get("activity_type") != "FILL"]
        credit = sum((1 if a.get("side", "").startswith("sell") else -1) * float(a["price"]) * float(a["qty"]) * 100 for a in fills)
        settle = sum(float(a.get("net_amount") or 0) for a in other)
        q = max(float(a.get("qty") or 1) for a in fills)
        risk = WING * 100 * q - credit
        if not fills or risk <= 0:
            continue
        ror = round((credit + settle) / risk, 4)
        row = {"date": r["date"], "ror": ror, "events": sorted({a.get("activity_type") for a in other})}
        with LEDGER.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        notify(f"XSP condor {r['date']}: {'+' if ror >= 0 else ''}{ror * 100:.1f}% of the amount at risk "
               f"(credit ${credit:,.2f}, settlement ${settle:,.2f}, events {row['events']})")
        print(f"::notice::live condor {r['date']} settled: return on risk {ror * 100:+.1f}%")


def main():
    from automation.notifier import notify_success, notify_alert
    if not os.environ.get("ALPACA_LIVE_KEY"):
        print("::notice::live condor: no ALPACA_LIVE_KEY -- skipped"); return
    mode = (os.environ.get("CONDOR_LIVE") or "dry").strip().lower()
    mode = mode if mode in ("yes", "probe") else "dry"
    qty = max(1, int(os.environ.get("CONDOR_CONTRACTS") or 1))
    now = dt.datetime.now(ET)
    say = lambda m: notify_success("Live XSP condor", m)
    try:
        reconcile(now.date(), say)
    except Exception as ex:
        print(f"::warning::live condor reconcile failed: {str(ex)[:200]}")
    if not (ENTRY_FROM <= now.time() <= ENTRY_TO):
        print(f"::notice::live condor: outside the entry window ({now:%H:%M} ET) -- reconcile only"); return
    if not api("GET", "/v2/clock").get("is_open"):
        print("::notice::live condor: market closed"); return
    if LEDGER.exists() and any(json.loads(l).get("date") == now.date().isoformat() and json.loads(l).get("mode") == mode
                               for l in LEDGER.read_text().splitlines() if l.strip()):
        print("::notice::live condor: already handled today"); return
    row = {"date": now.date().isoformat(), "mode": mode}   # no quantity: public repo
    eq = float(api("GET", "/v2/account").get("equity") or 0)
    if eq < FLOOR:
        row["status"] = "paused: account below the floor"
        _finish(row)
        notify_alert("Live XSP condor", f"PAUSED {row['date']}: account ${eq:,.2f} is below the ${FLOOR:,.0f} floor -- "
                     "no condor until it is back above (add funds, or decide to stop). Nothing was traded.")
        print("::notice::live condor: paused -- account below the floor rule"); return
    rr = range_ratio(now.date())
    row["range_ratio"] = round(rr, 3) if rr is not None else None
    sizing_note = ""
    try:
        e = price_condor(now.date())
    except Exception as ex:
        row["status"] = f"error: {str(ex)[:160]}"
        _finish(row); notify_alert("Live XSP condor", f"{mode}: could not price today's condor: {ex}"); return
    row.update(credit_frac_natural=round(e["natural"] / WING, 3), credit_frac_mid=round(e["mid"] / WING, 3), feed=e["feed"])
    pct = float(os.environ.get("CONDOR_STAKE_PCT") or 0) / 100
    ml = (WING - e["natural"]) * 100                      # max loss per contract at the natural credit
    bad_ml = not (0.40 * WING * 100 <= ml <= 0.98 * WING * 100)
    if pct > 0 and not bad_ml:
        cap = max(1, int(os.environ.get("CONDOR_MAX") or 5))
        want = max(1, int(pct * eq // ml))
        qty = min(want, cap)
        sizing_note = (f"; stake {pct * 100:.0f}% of ${eq:,.0f} / ${ml:,.0f} max loss = {want} contract(s)"
                       + (f", CAPPED at {cap} (raise CONDOR_MAX to go higher)" if want > cap else ""))
    plan = (f"XSP {e['spot']:.2f}, implied move {e['move']:.2f}; sell {e['ps']:g}/{e['cs']:g}, buy {e['ps'] - WING:g}/"
            f"{e['cs'] + WING:g} x{qty}; credit natural {e['natural']:.2f} / mid {e['mid']:.2f}; max loss ~${(WING - e['natural']) * 100 * qty:,.0f}; "
            f"SPY range so far {('%.2f' % rr) if rr is not None else 'n/a'}x its 20-day mean (skip >= {RANGE_CUT})" + sizing_note)
    acct = api("GET", "/v2/account")
    obp = float(acct.get("options_buying_power") or 0)
    need = (WING - max(e["natural"], 0)) * 100 * qty
    if bad_ml:
        row["status"] = "skipped: max loss per contract looks wrong (bad quotes?)"
    elif rr is not None and rr >= RANGE_CUT:
        row["status"] = "skipped: high range day"
    elif e["natural"] < MIN_NAT:
        row["status"] = "skipped: credit too small"
    elif obp < need:
        row["status"] = "skipped: not enough options buying power"
    elif mode == "dry":
        row["status"] = "dry run"
    elif mode == "probe":
        try:
            o = place(e["legs"], 1, 0.95)
            st = cancel(o["id"])
            row["status"] = f"probe: order accepted ({st.get('status')})"
        except RuntimeError as ex:
            row["status"] = f"probe: order REJECTED: {str(ex)[:160]}"
    else:
        try:
            status, credit, steps = work_order(e, qty)
            row.update(status=status, steps=steps, credit_frac=round(credit / WING, 3) if credit else None)
            if credit is not None and credit <= 0:
                notify_alert("Live XSP condor", f"filled at a DEBIT ({credit}) -- check the account now")
            plan += f"\nRESULT: {status}" + (f" at credit {credit:.2f} after {steps} steps" if credit else "")
        except RuntimeError as ex:
            row["status"] = f"error: {str(ex)[:160]}"
    _finish(row)
    msg = f"{mode.upper()} {row['date']}: {row['status']}\n{plan}\noptions buying power covers it: {obp >= need}"
    (notify_alert if row["status"].startswith(("error", "probe: order REJECTED")) else notify_success)("Live XSP condor", msg)
    print(f"::notice::live condor ({mode}): {row['status']}; credit/width natural {row.get('credit_frac_natural')}, "
          f"range ratio {row.get('range_ratio')}")


def _finish(row):
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc(); print(tb)
        print("::error::" + tb[-1000:].replace("\n", " | "))
        raise
