"""
automation/paper_condor.py
==========================
Paper ORDERS for shadow #9's iron condors, in a separate Alpaca paper account (Connor, Oct 22, 2026).

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
STATE, LEDGER, SHADOW = DATA / "paper_condor_state.json", DATA / "paper_condor_ledger.jsonl", DATA / "shadow_condor_state.json"
BASE = "https://paper-api.alpaca.markets"
RISK_FRAC, STEP, WAIT_S, MAX_STEPS = 0.02, 0.05, 60, 8


def api(method, path, body=None, **q):
    url = BASE + path + ("?" + urllib.parse.urlencode(q) if q else "")
    h = {"APCA-API-KEY-ID": os.environ["ALPACA_PAPER2_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_PAPER2_SECRET"],
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
                           f"{o.get('steps', 0)} steps)" for o in d["orders"])
               + (f"; option events {ev}" if ev else "") + (f"; ASSIGNED SHARES closed: {closed}" if closed else ""))
        log.info(msg)
        (notify_alert if closed else notify_success)("Paper condor", msg)
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
        notify_alert("Paper condor", f"{o_rec['und']} filled at a DEBIT ({net:.2f}) -- limit price sign is wrong; check the code")
    return o_rec


def live_natural(legs):
    """Natural credit (sell at bid, buy at ask) from live quotes now; None if unavailable."""
    syms = ",".join(legs.values())
    for feed in ("opra", "indicative"):
        try:
            url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols={syms}&feed={feed}"
            h = {"APCA-API-KEY-ID": os.environ["ALPACA_PAPER2_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_PAPER2_SECRET"]}
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
    if not os.environ.get("ALPACA_PAPER2_KEY"):
        print("::notice::paper condor: no ALPACA_PAPER2_KEY secret -- skipped"); return
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
    if not (dt.time(9, 58) <= now.time() <= dt.time(11, 30)):           # same entry window as the shadow
        STATE.write_text(json.dumps(state, indent=1)); return
    key = now.date().isoformat()
    shadow = json.loads(SHADOW.read_text()).get("open", {}).get(key, {}) if SHADOW.exists() else {}
    day = state["days"].setdefault(key, {"orders": [], "equity_before": None})
    done = {o["und"] for o in day["orders"]}
    todo = [u for u in shadow if u not in done]
    if todo:
        if day["equity_before"] is None:
            day["equity_before"] = round(equity(), 2)
        for u in todo:
            try:
                rec = enter_one(shadow[u], day["equity_before"], RISK_FRAC / len(shadow))   # 2% total, split like COMBO
            except Exception as ex:
                rec = {"und": u, "status": f"error: {str(ex)[:200]}", "mid": None, "natural": None}
                notify_alert("Paper condor", f"{u} order error: {ex}")
            day["orders"].append(rec)
            log.info(f"paper {u}: {rec}")
    if not day["orders"]:
        state["days"].pop(key, None)
    STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
