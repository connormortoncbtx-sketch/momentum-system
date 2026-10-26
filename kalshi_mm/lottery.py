"""
kalshi_mm/lottery.py -- the weekly weather roll: a $5 "lottery ticket" built as a rolling parlay (Oct 20, 2026).

Each week (from Monday 9am Central) it stakes LOT_STAKE dollars on one daily-high bucket priced ~50-62c, bought with a
post-only limit order at the bid (maker: no fee, no spread paid). If the bucket settles YES, the whole payout rolls
into the next leg; a NO ends the week. LOT_LEGS straight wins (~1% for 7 legs at ~52c) pays roughly 100x and stops.
Why built this way: a rolling parlay of separate, liquid, fairly priced legs entered as a maker loses far less to
margin than a combo/parlay ticket (each leg's margin compounds). It is still a ~99%-to-lose ticket.

Separation from the market maker (bot.py):
  - legs come from weather series the bot does NOT quote (LOT_SERIES minus SERIES), so they never share a market
  - orders carry client_order_id 'lot-...'; the bot's order handling (trims, restarts, cancel_all) skips them
  - honours the bot's KILL file: cancels its own order and does nothing while it exists
Runs every 15 minutes from systemd (kmm-lottery.timer); state in LOG_DIR/lottery.json; phone updates via ntfy.
"""
import datetime as dt
import json
import math
import os
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot as B  # noqa: E402

E = os.environ.get
CT = ZoneInfo("America/Chicago")
ENABLED = E("LOT_ENABLED", "yes").lower() == "yes"
STAKE = float(E("LOT_STAKE", "5"))
LEGS = int(E("LOT_LEGS", "7"))
LO, HI, AIM = float(E("LOT_MIN_PRICE", "0.48")), float(E("LOT_MAX_PRICE", "0.62")), 0.55
MIN_H, MAX_H = 6.0, 40.0                     # leg must close 6-40h out (the window where weather quotes are fair)
REPRICE_H = 2.0                              # an unfilled order older than this is moved to the current bid
LOT_SERIES = [s for s in E("LOT_SERIES", "KXHIGHTHOU,KXHIGHTDC,KXHIGHTPHX,KXHIGHTSATX,KXHIGHTSEA,KXHIGHTMIN,"
                                         "KXHIGHTOKC,KXHIGHTNOLA,KXHIGHTSFO,KXHIGHTEWR").split(",")
              if s and s not in B.SERIES]
STATE = os.path.join(B.LOG_DIR, "lottery.json")
B.READ_LIMIT, B.WRITE_LIMIT = B.RateLimit(3), B.RateLimit(2)   # yield to the market maker on the shared rate limit


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"status": "idle", "week": None, "leg": 0, "cash": 0.0, "history": []}


def save(st):
    os.makedirs(B.LOG_DIR, exist_ok=True)
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def say(msg, prio="default"):
    B.log.info(msg)
    B.jlog("lottery", msg=msg)
    B.notify(msg, title="Kalshi weekly roll", prio=prio)


def hours_to_close(m, now):
    t = m.get("close_time")
    if not t:
        return -1
    return (dt.datetime.fromisoformat(t.replace("Z", "+00:00")) - now).total_seconds() / 3600


def pick_leg(k, now, exclude=()):
    """Best candidate: an open bucket closing 6-40h out whose mid is 48-62c, closest to 55c, tightest spread."""
    best = None
    for s in LOT_SERIES:
        try:
            mk = k.open_markets(s)
        except Exception as e:
            B.log.warning(f"lottery markets {s}: {e}"); continue
        for m in mk:
            h = hours_to_close(m, now)
            if not (MIN_H <= h <= MAX_H) or m["ticker"] in exclude:
                continue
            yb, ya = B.num(m.get("yes_bid_dollars")), B.num(m.get("yes_ask_dollars"))
            if not (0 < yb < ya):
                continue
            mid, spr = (yb + ya) / 2, ya - yb
            if not (LO <= mid <= HI) or spr > 0.06:
                continue
            score = abs(mid - AIM) + spr
            if best is None or score < best[0]:
                best = (score, m["ticker"], h)
    return (best[1], best[2]) if best else (None, None)


def entry_price(k, ticker):
    """Join the best YES bid (or improve it by 1c when the spread is >= 3c) -- always below the ask, so post-only."""
    y, n = k.book(ticker)
    if not y or not n:
        return None
    bid, ask = y[0][0], 1 - n[0][0]
    px = bid + 0.01 if ask - bid >= 0.03 else bid
    return round(px, 2) if LO - 0.02 <= px <= HI + 0.02 and px < ask else None


def order(k, oid):
    d = k.req("GET", f"/portfolio/orders/{oid}", auth=True)
    return d.get("order", d)


def market(k, ticker):
    d = k.req("GET", f"/markets/{ticker}")
    return d.get("market", d)


def sync(k, st):
    """Book any new fills of the current order into held / cash; returns the order."""
    o = order(k, st["order_id"])
    filled = B.num(o.get("fill_count_fp", o.get("fill_count")))
    px = st.get("price", 0)
    new = filled - st.get("order_filled", 0.0)
    if new > 0:
        st["held"] += new; st["cash"] = round(st["cash"] - new * px, 4); st["order_filled"] = filled
    return o


def step(k, st, now):
    """One pass of the state machine. Returns the (mutated) state."""
    local = now.astimezone(CT)
    week = f"{local.isocalendar()[0]}-W{local.isocalendar()[1]:02d}"
    if os.path.exists(B.KILL_FILE):
        if st.get("order_id"):
            try:
                k.cancel(st["order_id"], st.get("ticker"))
            except Exception:
                pass
            st["order_id"] = None
        return st
    # a new week starts Monday 9am CT, unless a roll is still alive
    if st["status"] in ("idle", "done") and st.get("week") != week and (local.weekday() > 0 or local.hour >= 9):
        st.update(status="buying", week=week, leg=1, cash=STAKE, ticker=None, order_id=None, held=0.0,
                  placed_at=None, history=st.get("history", [])[-30:])
        say(f"Weekly roll {week} started: ${STAKE:.2f}, {LEGS} legs to the jackpot")
    if st["status"] == "buying":
        # 1) follow the resting order
        if st.get("order_id"):
            o = sync(k, st)
            status = str(o.get("status", "")).lower()
            h = hours_to_close(market(k, st["ticker"]), now)
            stale = (now - dt.datetime.fromisoformat(st["placed_at"])).total_seconds() / 3600 > REPRICE_H
            if status in ("executed", "filled") or B.num(o.get("remaining_count_fp", o.get("remaining_count"))) <= 0:
                st["order_id"] = None
            elif h < MIN_H - 2 or stale:
                try:
                    k.cancel(st["order_id"], st["ticker"])
                except Exception as e:
                    B.log.warning(f"lottery cancel: {e}")
                sync(k, st)                             # count any fill that landed before the cancel
                st["order_id"] = None
                if h < MIN_H - 2:                       # too close to the close: stop buying this leg
                    if st["held"] > 0:
                        st["status"] = "holding"
                    else:
                        st["ticker"] = None
            if st["order_id"] is None and st["held"] > 0 and st["cash"] < st.get("price", 1):
                st["status"] = "holding"                # fully invested (leftover cash < one contract)
        # 2) place / re-place
        if st["status"] == "buying" and not st.get("order_id"):
            tk, h = (st["ticker"], hours_to_close(market(k, st["ticker"]), now)) if st.get("ticker") else pick_leg(k, now)
            if tk and h >= MIN_H - 2:
                px = entry_price(k, tk)
                n = math.floor(st["cash"] / px) if px else 0
                if px and n >= 1:
                    o = k.place(tk, "yes", px, n, coid_prefix=B.LOTTERY_PREFIX)
                    o = o.get("order", o)
                    st.update(ticker=tk, order_id=o.get("order_id"), price=px, order_filled=0.0,
                              placed_at=now.isoformat())
                    B.jlog("lottery", event="order", ticker=tk, price=px, count=n, leg=st["leg"])
                    if st.get("announced") != [st["week"], st["leg"], tk]:   # once per leg/market, not every reprice
                        st["announced"] = [st["week"], st["leg"], tk]
                        say(f"Leg {st['leg']}/{LEGS}: bidding {px * 100:.0f}c for {n} x {tk} (${st['cash']:.2f} riding; "
                            f"closes in {h:.0f}h)")
                    if B.num(o.get("fill_count_fp", o.get("fill_count"))) > 0:
                        pass                            # picked up on the next pass via the order status
                elif st["held"] > 0:
                    st["status"] = "holding"
            elif st.get("ticker") and st["held"] > 0:
                st["status"] = "holding"
            elif st.get("ticker"):
                st["ticker"] = None                     # leg expired unfilled: pick another next pass
    if st["status"] == "holding":
        m = market(k, st["ticker"])
        res = str(m.get("result", "")).lower()
        if res in ("yes", "no"):
            payout = st["held"] * (1.0 if res == "yes" else 0.0)
            st["history"].append(dict(week=st["week"], leg=st["leg"], ticker=st["ticker"], price=st.get("price"),
                                      held=st["held"], result=res))
            if res == "yes":
                st["cash"] = round(st["cash"] + payout, 2)
                if st["leg"] >= LEGS:
                    say(f"JACKPOT: all {LEGS} legs hit -- ${STAKE:.2f} became ${st['cash']:.2f}. Roll stops here.", prio="high")
                    st["status"] = "done"
                else:
                    say(f"Leg {st['leg']}/{LEGS} hit ({st['ticker']}): now ${st['cash']:.2f}. Rolling into leg {st['leg'] + 1}.")
                    st.update(status="buying", leg=st["leg"] + 1, ticker=None, held=0.0, order_id=None)
            else:
                say(f"Leg {st['leg']}/{LEGS} missed ({st['ticker']}). Week over; next roll starts Monday 9am.")
                st.update(status="done", held=0.0, ticker=None)
    return st


def main():
    if not ENABLED or B.MODE != "prod" or B.E("I_UNDERSTAND_REAL_MONEY") != "yes":
        B.log.info("lottery disabled (LOT_ENABLED / MODE / I_UNDERSTAND_REAL_MONEY)"); return
    k = B.Kalshi(B.MODE)
    st = load()
    try:
        st = step(k, st, dt.datetime.now(dt.timezone.utc))
    finally:
        save(st)


if __name__ == "__main__":
    main()
