"""
automation/paper_condor.py
==========================
Paper ORDERS for same-day iron condors, in a separate Alpaca paper account (Connor, Oct 22, 2026).

From Oct 26, 2026 (Connor, Oct 25): the AFTERNOON version from the ThetaData study (docs/2026-10_strategy_research.md,
"Full 0DTE options grid"): enter at 14:00 ET, shorts 1 implied move out, SPY $5 / QQQ 0.8% wings, stop when the cost to
close at mid reaches 3x the credit (= a loss of 2x the credit), checked on the 15-minute marks like the backtest, else
hold to the close. Its own workflow (paper_condor.yml, dispatched 13:00 CT); legs priced live here with shadow #9's
helpers -- shadow #9 itself stays the 10:00 reference. Before Oct 26 the account copied shadow #9's 10:00 legs.

The shadow (automation/shadow_condor.py) prices each day's condor from quotes but places nothing. Paper fills are
optimistic, so this does NOT settle the cost question -- it tests the mechanics a real account would face:
  - does a 4-leg order fill, at what credit vs the shadow's mid / natural, after how many price steps
  - what Alpaca does at expiry with physically settled SPY/QQQ options: expired, exercised, assigned (a short leg
    in the money with its insurance out of it leaves SHARES overnight -- the backtest assumed cash settlement)
  - buying-power / approval rejections
Each run (same schedule as the shadow, after it): (1) reconcile earlier days -- account equity, option activities
(expiry / exercise / assignment), leftover share positions closed at market; (2) enter today: for each underlying
the shadow entered, one multi-leg limit order for the same strikes, sized to 2% of paper equity at risk, starting at
the mid credit and stepping 5c toward the natural credit every 60s until filled (never past natural).
Keys: ALPACA_PAPER2_KEY / ALPACA_PAPER2_SECRET (GitHub secrets) -- a paper account used for nothing else.
Without them it exits quietly. State: data/paper_condor_state.json; ledger: data/paper_condor_ledger.jsonl.
"""
import datetime as dt
import json
import logging
import math
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
from automation.notifier import notify_success, notify_alert  # noqa: E402

log = logging.getLogger("paper_condor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
ET = ZoneInfo("America/New_York")
DATA = ROOT / "data"
# Variants (CONDOR_VARIANT env), each in its own paper account (Connor, Oct 25):
#   p14  14:00 ET, shorts 1 implied move, SPY $5 / QQQ 0.8% wings, stop at 3x credit  (ALPACA_PAPER2_*)
#   p15  15:00 ET, shorts 0.5 implied move, narrowest wings SPY $2 / QQQ 0.4%, hold to the close -- the dev-chosen
#        variant of research/mix_options_v2.py and the realistic small-account form (ALPACA_PAPER3_*); stop not
#        acted on, but the 15-minute and 1-minute marks are recorded so a stop can be evaluated later
VARIANTS = {"p14": dict(key="ALPACA_PAPER2", entry=dt.time(14, 0), k=1.0, spy_w=5, qqq_f=0.008, stop=3.0, files="paper_condor"),
            "p15": dict(key="ALPACA_PAPER3", entry=dt.time(15, 0), k=0.5, spy_w=2, qqq_f=0.004, stop=0.0, files="paper_condor_1500")}
VARIANT = os.environ.get("CONDOR_VARIANT", "p14")
V = VARIANTS[VARIANT]
KEY_ENV, SECRET_ENV = V["key"] + "_KEY", V["key"] + "_SECRET"
STATE, LEDGER, SHADOW = DATA / f"{V['files']}_state.json", DATA / f"{V['files']}_ledger.jsonl", DATA / "shadow_condor_state.json"
BASE = "https://paper-api.alpaca.markets"
RISK_FRAC, STEP, WAIT_S, MAX_STEPS = 0.10, 0.05, 60, 8   # paper: 10%/day (5% per ETF) to stress buying power & assignment (Connor, Oct 23)
_e = dt.datetime.combine(dt.date.today(), V["entry"])
ENTRY_FROM, ENTRY_TO = (_e - dt.timedelta(minutes=2)).time(), (_e + dt.timedelta(minutes=30)).time()
STOP_MULT, LAST_CHECK = V["stop"], dt.time(15, 45)        # stop: cost to close >= STOP_MULT x credit (0 = never); last mark


def api(method, path, body=None, **q):
    url = BASE + path + ("?" + urllib.parse.urlencode(q) if q else "")
    h = {"APCA-API-KEY-ID": os.environ[KEY_ENV], "APCA-API-SECRET-KEY": os.environ[SECRET_ENV],
         "Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url, method=method, headers=h, data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            t = r.read()
            return json.loads(t) if t else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")


def equity():
    return float(api("GET", "/v2/account")["equity"])


def reconcile(state, today):
    rows = []
    for key in sorted(k for k in state["days"] if k < today.isoformat() and not state["days"][k].get("reconciled")):
        d = state["days"][key]
        acts = api("GET", "/v2/account/activities", after=f"{key}T00:00:00Z", page_size=100)
        opt = [a for a in acts if str(a.get("activity_type", "")).startswith("OP")]
        pos = api("GET", "/v2/positions")
        closed = []
        for p in pos:                                                  # shares left by an assignment / exercise
            if p.get("asset_class") == "us_equity":
                try:
                    api("DELETE", f"/v2/positions/{p['symbol']}")
                    closed.append({"symbol": p["symbol"], "qty": p["qty"], "unrealized_pl": p.get("unrealized_pl")})
                except Exception as e:
                    closed.append({"symbol": p["symbol"], "qty": p["qty"], "error": str(e)[:150]})
        if closed:
            time.sleep(5)
        eq = equity()
        row = {"date": key, "equity_before": d["equity_before"], "equity_after": round(eq, 2),
               "pnl": round(eq - d["equity_before"], 2), "pnl_pct": round((eq / d["equity_before"] - 1) * 100, 3),
               "orders": d["orders"], "option_events": [{k: a.get(k) for k in ("activity_type", "symbol", "qty", "price", "net_amount", "date")} for a in opt],
               "shares_closed": closed}
        rows.append(row)
        d["reconciled"] = True
        ev = sorted({a.get("activity_type") for a in opt})
        msg = (f"paper condor {key}: P&L {row['pnl']:+,.2f} ({row['pnl_pct']:+.2f}%), fills: "
               + "; ".join(f"{o['und']} {o.get('status')} credit {o.get('filled_credit')} (mid {o['mid']}, natural {o['natural']}, "
                           f"{o.get('steps', 0)} steps)" + (f" STOPPED {o['stopped_at'][11:16]} for {o.get('stop_debit')}" if o.get("stopped_at") else "")
                           for o in d["orders"])
               + (f"; option events {ev}" if ev else "") + (f"; ASSIGNED SHARES closed: {closed}" if closed else ""))
        log.info(msg)
        (notify_alert if closed else notify_success)(f"Paper condor {VARIANT}", msg)
    return rows


def leg_list(legs):
    return [{"symbol": legs["long_put"], "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
            {"symbol": legs["short_put"], "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
            {"symbol": legs["short_call"], "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
            {"symbol": legs["long_call"], "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"}]


def place(legs, qty, credit, sign):
    body = {"order_class": "mleg", "qty": str(qty), "type": "limit", "time_in_force": "day",
            "limit_price": f"{sign * credit:.2f}", "legs": leg_list(legs)}
    return api("POST", "/v2/orders", body)


def filled(o_rec, st, steps, price):
    legs = st.get("legs") or []
    net = sum((1 if l.get("side") == "sell" else -1) * float(l.get("filled_avg_price") or 0) for l in legs)
    o_rec.update(status="filled", steps=steps, limit=price, filled_avg_price=st.get("filled_avg_price"),
                 filled_credit=round(net, 4) if legs else abs(float(st.get("filled_avg_price") or 0)),
                 legs={l["symbol"]: [l.get("side"), l.get("filled_avg_price")] for l in legs}, filled_at=st.get("filled_at"))
    if legs and net <= 0:
        notify_alert(f"Paper condor {VARIANT}", f"{o_rec['und']} filled at a DEBIT ({net:.2f}) -- limit price sign is wrong; check the code")
    return o_rec


def live_natural(legs):
    """Natural credit (sell at bid, buy at ask) from live quotes now; None if unavailable."""
    syms = ",".join(legs.values())
    for feed in ("opra", "indicative"):
        try:
            url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols={syms}&feed={feed}"
            h = {"APCA-API-KEY-ID": os.environ[KEY_ENV], "APCA-API-SECRET-KEY": os.environ[SECRET_ENV]}
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=20) as r:
                q = json.loads(r.read()).get("quotes", {})
            b = lambda k: float(q[legs[k]]["bp"]); a = lambda k: float(q[legs[k]]["ap"])
            return round(b("short_put") + b("short_call") - a("long_put") - a("long_call"), 2)
        except Exception:
            continue
    return None


def enter_one(e, eq, risk_frac):
    W, mid, nat = e["width"], e["credit"]["mid"], e["credit"]["natural"]
    o_rec = {"und": e["und"], "mid": mid, "natural": nat, "width": W}
    if nat <= 0 or mid <= 0:
        o_rec["status"] = "skipped: no credit"; return o_rec
    qty = max(1, math.floor(risk_frac * eq / ((W - nat) * 100)))
    o_rec["qty"] = qty
    price, sign, order, steps = round(mid, 2), -1, None, 0          # negative = credit (assumed; the leg fills verify it)
    while True:
        try:
            order = place(e["legs"], qty, price, sign)
        except RuntimeError as ex:
            if sign == -1 and ("limit_price" in str(ex) or "422" in str(ex)) and steps == 0 and "sign_tried" not in o_rec:
                o_rec["sign_tried"] = str(ex)[:200]; sign = 1; continue      # convention is positive: retry once
            o_rec["status"] = f"rejected: {str(ex)[:200]}"; return o_rec
        o_rec["limit_sign"] = sign
        t0 = time.time()
        while time.time() - t0 < WAIT_S:
            time.sleep(5)
            st = api("GET", f"/v2/orders/{order['id']}", nested="true")
            if st.get("status") == "filled":
                return filled(o_rec, st, steps, price)
            if st.get("status") in ("rejected", "canceled", "expired"):
                o_rec.update(status=f"order {st.get('status')}"); return o_rec
        try:
            api("DELETE", f"/v2/orders/{order['id']}")
        except RuntimeError:
            pass
        time.sleep(2)
        st = api("GET", f"/v2/orders/{order['id']}", nested="true")
        if st.get("status") == "filled":                                          # filled while cancelling
            return filled(o_rec, st, steps, price)
        steps += 1
        ln = live_natural(e["legs"])                   # the market moves after 10:00: chase the LIVE natural, not the
        if ln is not None:                             # shadow's snapshot (Oct 23: QQQ never filled at a stale natural)
            nat = ln
            o_rec.setdefault("live_naturals", []).append(ln)
        nxt = round(max(nat, price - STEP), 2)
        if steps > MAX_STEPS or nat <= 0 or (nxt == price and price <= nat):
            o_rec.update(status="no fill at natural", steps=steps, limit=price); return o_rec
        price = nxt


def main():
    if not os.environ.get(KEY_ENV):
        print(f"::notice::paper condor {VARIANT}: no {KEY_ENV} secret -- skipped"); return
    now = dt.datetime.now(ET)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"days": {}}
    rows = reconcile(state, now.date())
    if rows:
        with LEDGER.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    acct = api("GET", "/v2/account")
    print(f"::notice::paper condor account: equity {acct.get('equity')}, options level {acct.get('options_trading_level')}, "
          f"approved {acct.get('options_approved_level')}, status {acct.get('status')}")
    log.info(f"paper account: equity {acct.get('equity')}, options level {acct.get('options_trading_level')} "
             f"(approved {acct.get('options_approved_level')}), status {acct.get('status')}")
    if not (ENTRY_FROM <= now.time() <= ENTRY_TO):
        STATE.write_text(json.dumps(state, indent=1)); return
    import automation.shadow_condor as SC
    oc, sc, tc = SC.clients()
    if not tc.get_clock().is_open:
        log.info("market closed today -- no entry"); STATE.write_text(json.dumps(state, indent=1)); return
    key = now.date().isoformat()
    day = state["days"].setdefault(key, {"orders": [], "equity_before": None, "version": f"{VARIANT}: {V['entry']:%H:%M} k{V['k']:g} "
                                                f"SPY ${V['spy_w']} / QQQ {V['qqq_f'] * 100:g}% {'stop ' + format(STOP_MULT, 'g') + 'x' if STOP_MULT else 'hold'}"})
    done = {o["und"] for o in day["orders"]}
    todo = [u for u in SC.UNDS if u not in done]
    if todo:
        if day["equity_before"] is None:
            day["equity_before"] = round(equity(), 2)
        for u in todo:
            try:
                e = price_condor(oc, sc, SC, u, now.date(), dt.datetime.now(ET))     # live legs at the entry time
                rec = enter_one(e, day["equity_before"], RISK_FRAC / len(SC.UNDS))
                rec.update(legs_sym=e["legs"], spot=e["spot"], implied_move=e["implied_move"],
                           entered_at=dt.datetime.now(ET).isoformat(timespec="seconds"))
            except Exception as ex:
                rec = {"und": u, "status": f"error: {str(ex)[:200]}", "mid": None, "natural": None}
                notify_alert(f"Paper condor {VARIANT}", f"{u} order error: {ex}")
            day["orders"].append(rec)
            log.info(f"paper {u}: {rec}")
            STATE.write_text(json.dumps(state, indent=1))
    if not day["orders"]:
        state["days"].pop(key, None)
    STATE.write_text(json.dumps(state, indent=1))
    monitor(state, day, oc, SC)


def price_condor(oc, sc, SC, und, day, now):
    """Strikes and credits from live quotes for this variant (k, wing widths); same method as shadow #9's enter_one."""
    s0, sfeed = SC.spot_now(sc, und)
    atm = round(s0)
    W = V["spy_w"] if und == "SPY" else max(1, round(V["qqq_f"] * s0))
    q1, _ = SC.quotes(oc, [SC.occ(und, day, "C", atm), SC.occ(und, day, "P", atm)])
    mid = lambda q: (q["bid"] + q["ask"]) / 2
    M = mid(q1[SC.occ(und, day, "C", atm)]) + mid(q1[SC.occ(und, day, "P", atm)])
    ps, cs = round(s0 - V["k"] * M), round(s0 + V["k"] * M)
    if cs <= ps:
        ps, cs = atm - 1, atm + 1
    legs = {"short_put": SC.occ(und, day, "P", ps), "long_put": SC.occ(und, day, "P", ps - W),
            "short_call": SC.occ(und, day, "C", cs), "long_call": SC.occ(und, day, "C", cs + W)}
    q, feed = SC.quotes(oc, list(legs.values()))
    Q = {k_: q[v] for k_, v in legs.items()}
    natural = Q["short_put"]["bid"] + Q["short_call"]["bid"] - Q["long_put"]["ask"] - Q["long_call"]["ask"]
    midc = sum((1 if k_.startswith("short") else -1) * mid(Q[k_]) for k_ in Q)
    log.info(f"{VARIANT} {und}: {s0:.2f}, move {M:.2f}, puts {ps}/{ps - W}, calls {cs}/{cs + W}, natural {natural:.2f} / mid {midc:.2f}")
    return {"und": und, "spot": s0, "implied_move": round(M, 4), "width": W, "legs": legs,
            "credit": {"natural": round(natural, 4), "mid": round(midc, 4)}}


def leg_mid_cost(oc, SC, legs):
    """(cost to close at mid, at natural) for an open condor, from live quotes."""
    q, _ = SC.quotes(oc, list(legs.values()))
    Q = {k: q[v] for k, v in legs.items()}
    mid = lambda x: (x["bid"] + x["ask"]) / 2
    cmid = sum((1 if k.startswith("short") else -1) * mid(Q[k]) for k in Q)
    cnat = Q["short_put"]["ask"] + Q["short_call"]["ask"] - Q["long_put"]["bid"] - Q["long_call"]["bid"]
    return round(cmid, 4), round(cnat, 4)


def close_one(rec, oc, SC):
    """Buy the condor back: mleg limit at the mid cost, stepping 5c toward (and finally 10c past) the natural cost."""
    legs, qty, sign = rec["legs_sym"], rec["qty"], -rec.get("limit_sign", -1)       # debit = opposite sign of credit
    body_legs = [{"symbol": legs["long_put"], "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_close"},
                 {"symbol": legs["short_put"], "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_close"},
                 {"symbol": legs["short_call"], "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_close"},
                 {"symbol": legs["long_call"], "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_close"}]
    cmid, cnat = leg_mid_cost(oc, SC, legs)
    price = round(cmid, 2)
    for step in range(8):
        o = api("POST", "/v2/orders", {"order_class": "mleg", "qty": str(qty), "type": "limit", "time_in_force": "day",
                                       "limit_price": f"{sign * price:.2f}", "legs": body_legs})
        t0 = time.time()
        while time.time() - t0 < 30:
            time.sleep(5)
            st = api("GET", f"/v2/orders/{o['id']}", nested="true")
            if st.get("status") == "filled":
                lg = st.get("legs") or []
                debit = sum((1 if l.get("side") == "buy" else -1) * float(l.get("filled_avg_price") or 0) for l in lg)
                rec.update(stopped_at=dt.datetime.now(ET).isoformat(timespec="seconds"), stop_debit=round(debit, 4),
                           stop_steps=step, stop_mid=cmid, stop_natural=cnat)
                return True
        try:
            api("DELETE", f"/v2/orders/{o['id']}")
        except RuntimeError:
            pass
        time.sleep(2)
        if api("GET", f"/v2/orders/{o['id']}", nested="true").get("status") == "filled":
            rec.update(stopped_at=dt.datetime.now(ET).isoformat(timespec="seconds"), stop_steps=step, stop_mid=cmid,
                       stop_natural=cnat)
            return True
        _, cnat = leg_mid_cost(oc, SC, legs)
        price = round(min(price + STEP, cnat + 0.10), 2) if price < cnat + 0.10 else round(cnat + 0.10, 2)
    rec["stop_status"] = "stop triggered but not filled"
    return False


def monitor(state, day, oc, SC):
    """Every minute until 15:45 ET: record the cost to close (observation only -- a finer stop is untested), and act on
    the stop ONLY on the 15-minute marks, as the backtest did (3x credit). The minute marks (r["marks_1m"]) are the data
    for testing a faster or emergency stop later."""
    open_ = [r for r in day["orders"] if r.get("status") == "filled" and r.get("legs_sym") and not r.get("stopped_at")]
    while open_:
        now = dt.datetime.now(ET)
        nxt = (now + dt.timedelta(minutes=1)).replace(second=20, microsecond=0)
        if nxt.time() > dt.time(LAST_CHECK.hour, LAST_CHECK.minute, 59):
            break
        time.sleep(max(0, (nxt - now).total_seconds()))
        on_mark = nxt.minute % 15 == 0
        for r in list(open_):
            try:
                cmid, cnat = leg_mid_cost(oc, SC, r["legs_sym"])
            except Exception as ex:
                log.warning(f"{r['und']} marks: {ex}"); continue
            c0 = float(r.get("filled_credit") or 0)
            r.setdefault("marks_1m", []).append([f"{nxt:%H:%M}", cmid, cnat])
            if not on_mark:
                continue
            r.setdefault("marks", []).append([f"{nxt:%H:%M}", cmid, cnat])
            if STOP_MULT and c0 > 0 and cmid >= STOP_MULT * c0:
                log.info(f"{r['und']} STOP: cost to close {cmid:.2f} >= {STOP_MULT:.0f}x credit {c0:.2f}")
                ok = close_one(r, oc, SC)
                notify_alert(f"Paper condor {VARIANT}", f"{r['und']} stop at {nxt:%H:%M}: cost {cmid:.2f} vs credit {c0:.2f}; "
                             + (f"closed for {r.get('stop_debit')}" if ok else "NOT filled -- check the account"))
                open_.remove(r)
        if on_mark:
            STATE.write_text(json.dumps(state, indent=1))
    STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
