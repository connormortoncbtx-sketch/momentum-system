"""
kalshi_mm/bot.py -- passive "deep quote" market maker for Kalshi daily-high temperature markets.

Strategy (from research/kalshi/market_making.py, Oct 16 2026): makers in these markets earned
~0.4c/contract at the touch but ~1.3-1.9c at deeper levels, where impatient takers sweep the
book. So we rest small post-only orders OFFSET cents behind the best bid on BOTH sides:
  buy YES at (best YES bid - OFFSET)      and      buy NO at (best NO bid - OFFSET)
and hold fills to settlement. Makers pay no fee on these series (fee_type 'quadratic').

Modes (env MODE):
  dry   public market data only, no keys, no orders -- logs the quotes it WOULD place
  demo  Kalshi demo exchange (fake money), needs KALSHI_KEY_ID + KALSHI_PRIVATE_KEY
  prod  real money -- same keys for the real account. Never the default.

Safety: post_only orders only; per-market inventory cap; total capital budget; daily loss
limit; a KILL file or KILL=1 env cancels everything and exits; all orders cancelled on exit.
Every quote/fill/snapshot is appended to LOG_DIR/*.jsonl for the edge measurement.
"""
import base64, datetime as dt, json, logging, os, signal, sys, time, uuid
import urllib.error, urllib.parse, urllib.request

HOSTS = {"dry": "https://api.elections.kalshi.com", "prod": "https://external-api.kalshi.com",
         "demo": "https://external-api.demo.kalshi.co"}
PREFIX = "/trade-api/v2"

# --- parameters (env-overridable) -------------------------------------------------------
E = os.environ.get
MODE = E("MODE", "dry")
SERIES = E("SERIES", "KXHIGHNY,KXHIGHCHI,KXHIGHLAX,KXHIGHMIA,KXHIGHPHIL,KXHIGHDEN,KXHIGHAUS,KXHIGHTDAL").split(",")
OFFSET = float(E("OFFSET_C", "2")) / 100          # cents behind the best bid
SIZE = int(E("SIZE", "5"))                        # contracts per quote
INV_CAP = int(E("INV_CAP", "25"))                 # max |net position| per market, contracts
EVENT_CAP = float(E("EVENT_CAP", "30"))           # max total |contracts| across one event (city-day: all its buckets),
                                                  # held + resting orders that would add to it. The buckets of one
                                                  # city-day are one temperature, so they're one bet.


def event_of(ticker):
    return ticker.rsplit("-", 1)[0]


def adds_exposure(side, pos):
    """True if a fill on this side would increase |position| in its market."""
    return (side == "yes" and pos >= 0) or (side == "no" and pos <= 0)
BUDGET = float(E("BUDGET_USD", "200"))            # max collateral in positions + resting orders
DAILY_LOSS = float(E("DAILY_LOSS_USD", "40"))     # stop for the day beyond this mark-to-market loss
# Relative sizing (overrides the fixed $ numbers when > 0): budget = BUDGET_PCT x account value (cash + positions
# at mid), capped at BUDGET_MAX_USD if set; daily loss limit = DAILY_LOSS_PCT x the day's starting value.
BUDGET_PCT = float(E("BUDGET_PCT", "0"))
BUDGET_MAX = float(E("BUDGET_MAX_USD", "0"))
DAILY_LOSS_PCT = float(E("DAILY_LOSS_PCT", "0"))


def budget_for(value):
    if BUDGET_PCT > 0 and value > 0:
        b = BUDGET_PCT * value
        return min(b, BUDGET_MAX) if BUDGET_MAX > 0 else b
    return BUDGET


def sizing_desc():
    b = f"{BUDGET_PCT:.0%} of value" + (f" (max ${BUDGET_MAX:.0f})" if BUDGET_MAX > 0 else "") if BUDGET_PCT > 0 else f"${BUDGET:.0f}"
    d = f"{DAILY_LOSS_PCT:.0%} of day-start value" if DAILY_LOSS_PCT > 0 else f"${DAILY_LOSS:.0f}"
    return f"{b} daily_loss={d}"


def loss_limit(day_start_value):
    return DAILY_LOSS_PCT * day_start_value if DAILY_LOSS_PCT > 0 and day_start_value else DAILY_LOSS
MIN_H, MAX_H = float(E("MIN_HOURS", "1")), float(E("MAX_HOURS", "48"))
P_LO, P_HI = 0.03, 0.97                           # don't quote markets priced outside this
LOOP_S = float(E("LOOP_SECONDS", "20"))
RUN_MIN = float(E("RUN_MINUTES", "0"))            # 0 = run until stopped
LOG_DIR = E("LOG_DIR", "kalshi_mm_logs")
KILL_FILE = E("KILL_FILE", "KILL")
NTFY = E("NTFY_CHANNEL") or ""                    # phone alerts; also the remote stop: any message
STOP_TOPIC = f"{NTFY}-kmm-stop" if NTFY else ""   # containing "stop" on topic <NTFY_CHANNEL>-kmm-stop
KILLED_EXIT = 3                                   # systemd does not restart on this code

log = logging.getLogger("kmm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def now():
    return dt.datetime.now(dt.timezone.utc)


def jlog(name, **rec):
    os.makedirs(LOG_DIR, exist_ok=True)
    rec = {"ts": now().isoformat(timespec="seconds"), "mode": MODE, **rec}
    with open(f"{LOG_DIR}/{name}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def notify(msg, title="Kalshi bot", prio="default"):
    if not NTFY:
        return
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{NTFY}", data=msg.encode()[:3000],
                               headers={"Title": f"{title} [{MODE}]", "Priority": prio}), timeout=15)
    except Exception as e:
        log.warning(f"ntfy: {e}")


def remote_stop(since):
    """True if a message containing 'stop' was posted to STOP_TOPIC after unix time `since`."""
    if not STOP_TOPIC:
        return False
    try:
        with urllib.request.urlopen(f"https://ntfy.sh/{STOP_TOPIC}/json?poll=1&since={int(since)}", timeout=15) as r:
            return any("stop" in (json.loads(l).get("message") or "").lower() for l in r.read().decode().splitlines() if l.strip())
    except Exception as e:
        log.warning(f"stop check: {e}")
        return False


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# --- API client -----------------------------------------------------------------------------
class Kalshi:
    def __init__(self, mode):
        self.base = HOSTS[mode]
        self.key_id, self.key = None, None
        if mode != "dry":
            from cryptography.hazmat.primitives import serialization
            self.key_id = E("KALSHI_KEY_ID")
            pem = E("KALSHI_PRIVATE_KEY") or open(E("KALSHI_PRIVATE_KEY_FILE")).read()
            if not self.key_id or not pem:
                sys.exit("KALSHI_KEY_ID and KALSHI_PRIVATE_KEY (or _FILE) are required outside dry mode")
            self.key = serialization.load_pem_private_key(pem.encode(), password=None)

    def _sign(self, method, path):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
        ts = str(int(time.time() * 1000))
        msg = (ts + method + path.split("?")[0]).encode()
        if isinstance(self.key, ed25519.Ed25519PrivateKey):
            sig = self.key.sign(msg)
        elif isinstance(self.key, rsa.RSAPrivateKey):
            sig = self.key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                                hashes.SHA256())
        else:
            raise ValueError("unsupported key type")
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}

    def req(self, method, path, body=None, auth=False, **q):
        path = PREFIX + path + (("?" + urllib.parse.urlencode(q)) if q else "")
        for i in range(4):
            h = {"Accept": "application/json", "Content-Type": "application/json"}
            if auth:
                h.update(self._sign(method, path))
            r = urllib.request.Request(self.base + path, method=method, headers=h,
                                       data=json.dumps(body).encode() if body is not None else None)
            try:
                with urllib.request.urlopen(r, timeout=20) as resp:
                    txt = resp.read()
                    return json.loads(txt) if txt else {}
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and i < 3:
                    time.sleep(1.5 * (i + 1)); continue
                raise RuntimeError(f"{method} {path} -> {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError) as e:
                if i < 3:
                    time.sleep(1.5 * (i + 1)); continue
                raise RuntimeError(f"{method} {path} -> {e}")

    # public
    def open_markets(self, series):
        out, cursor = [], None
        while True:
            q = {"series_ticker": series, "status": "open", "limit": 200}
            if cursor:
                q["cursor"] = cursor
            d = self.req("GET", "/markets", **q)
            out += d.get("markets", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                return out

    def book(self, ticker):
        d = self.req("GET", f"/markets/{ticker}/orderbook", depth=10)
        ob = d.get("orderbook_fp") or {}
        if ob:
            yes = [(num(p), num(q)) for p, q in (ob.get("yes_dollars") or [])]
            no = [(num(p), num(q)) for p, q in (ob.get("no_dollars") or [])]
        else:                                           # older cent-integer shape
            ob = d.get("orderbook") or {}
            yes = [(num(p) / 100, num(q)) for p, q in (ob.get("yes") or [])]
            no = [(num(p) / 100, num(q)) for p, q in (ob.get("no") or [])]
        # best first (the API orders best->worst per the docs; sort to be safe)
        return sorted(yes, reverse=True), sorted(no, reverse=True)

    # private
    def balance(self):
        d = self.req("GET", "/portfolio/balance", auth=True)
        return num(d.get("balance")) / 100

    def positions(self):
        """{ticker: net position}; also sets self.cost = {ticker: $ cost basis} from Kalshi's market_exposure
        (what the position cost; used for the budget so paper gains don't crowd out new quotes)."""
        d = self.req("GET", "/portfolio/positions", auth=True, limit=1000, count_filter="position")
        mp = d.get("market_positions", [])
        self.cost = {}
        for p in mp:
            if p.get("market_exposure_dollars") is not None:
                self.cost[p["ticker"]] = num(p["market_exposure_dollars"])
            elif p.get("market_exposure") is not None:
                self.cost[p["ticker"]] = num(p["market_exposure"]) / 100
        return {p["ticker"]: num(p.get("position_fp", p.get("position"))) for p in mp}

    def resting(self):
        d = self.req("GET", "/portfolio/orders", auth=True, status="resting", limit=1000)
        return d.get("orders", [])

    def settlements(self, min_ts):
        d = self.req("GET", "/portfolio/settlements", auth=True, min_ts=min_ts, limit=1000)
        return d.get("settlements", [])

    def fills(self, min_ts):
        d = self.req("GET", "/portfolio/fills", auth=True, min_ts=min_ts, limit=1000)
        return d.get("fills", [])

    def place(self, ticker, side, price, count):
        """side 'yes' = buy YES at price (book bid); side 'no' = buy NO at price, i.e. sell YES
        at 1 - price (book ask). V2 orders are expressed on the YES leg only."""
        yes_px = price if side == "yes" else 1 - price
        body = {"ticker": ticker, "side": "bid" if side == "yes" else "ask", "count": f"{count:.2f}",
                "price": f"{yes_px:.4f}", "time_in_force": "good_till_canceled",
                "self_trade_prevention_type": "maker", "post_only": True,
                "client_order_id": str(uuid.uuid4()), "cancel_order_on_pause": True}
        return self.req("POST", "/portfolio/events/orders", body=body, auth=True)

    def cancel(self, order_id, ticker=None):
        q = {"market_ticker": ticker} if ticker else {}
        return self.req("DELETE", f"/portfolio/events/orders/{order_id}", auth=True, **q)


# --- quoting logic (pure, unit-tested) ---------------------------------------------------
def order_side(o):
    """Our side label for a resting order: 'yes' (bid on YES) or 'no' (ask on YES = buy NO)."""
    bs = o.get("book_side")
    if bs in ("bid", "ask"):
        return "yes" if bs == "bid" else "no"
    side, action = o.get("outcome_side") or o.get("side"), o.get("action", "buy")
    return side if action == "buy" else ("no" if side == "yes" else "yes")


def order_price(o, side):
    v = o.get(f"{side}_price_dollars")
    if v is not None:
        return round(num(v), 2)
    if o.get(f"{side}_price") is not None:
        return round(num(o.get(f"{side}_price")) / 100, 2)
    other = "no" if side == "yes" else "yes"
    return round(1 - num(o.get(f"{other}_price_dollars")), 2)


def remaining(o):
    return num(o.get("remaining_count_fp", o.get("remaining_count")))


def targets(yes_bids, no_bids, pos, hours_left):
    """Desired resting orders {side: price} for one market given its book and our net
    position (+ = long YES). Empty dict = quote nothing."""
    if not (MIN_H <= hours_left <= MAX_H) or not yes_bids or not no_bids:
        return {}
    by, bn = yes_bids[0][0], no_bids[0][0]
    mid = (by + (1 - bn)) / 2
    if not (P_LO <= mid <= P_HI) or by + bn >= 1:       # crossed/locked book: skip
        return {}
    out = {}
    py, pn = round(by - OFFSET, 2), round(bn - OFFSET, 2)
    if py >= 0.01 and pos + SIZE <= INV_CAP:
        out["yes"] = py
    if pn >= 0.01 and -pos + SIZE <= INV_CAP:
        out["no"] = pn
    return out


def collateral(pos, mark_yes):
    """Approximate money tied up in a position (cost basis unknown -> mark)."""
    return pos * mark_yes if pos > 0 else -pos * (1 - mark_yes)


# --- main loop --------------------------------------------------------------------------------
class Bot:
    def __init__(self):
        self.k = Kalshi(MODE)
        self.stop = False
        self.day, self.day_start_value = None, None
        self.last_fill_ts = int(time.time())
        self.t_start = time.time()
        self.last_stop_check, self.last_err_alert, self.err_count = 0.0, 0.0, 0
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))

    def killed(self):
        if os.path.exists(KILL_FILE) or E("KILL") == "1":
            return True
        if time.time() - self.last_stop_check > 60:
            self.last_stop_check = time.time()
            if remote_stop(self.t_start):
                open(KILL_FILE, "w").write("remote stop via ntfy\n")   # stays stopped across restarts
                return True
        return False

    def cancel_all(self, reason):
        """Cancel every resting order: per-order error handling, re-check, up to 5 passes."""
        if MODE == "dry":
            return
        left, errs = None, 0
        for attempt in range(5):
            try:
                rest = self.k.resting()
            except Exception as e:
                log.error(f"cancel_all list failed: {e}"); time.sleep(2); continue
            left = len(rest)
            if not rest:
                break
            for o in rest:
                try:
                    self.k.cancel(o["order_id"], o.get("ticker"))
                except Exception as e:
                    errs += 1; log.warning(f"cancel {o.get('order_id')}: {e}")
            time.sleep(1.5)
        else:
            try:
                left = len(self.k.resting())
            except Exception:
                pass
        jlog("events", event="cancel_all", reason=reason, resting_after=left, cancel_errors=errs)
        (log.info if not left else log.error)(f"cancel_all ({reason}): {left} resting left, {errs} errors")
        if left:
            notify(f"cancel_all ({reason}) left {left} orders resting -- check kalshi.com", prio="high")

    def markets(self):
        out = []
        for s in SERIES:
            try:
                for m in self.k.open_markets(s):
                    ct = dt.datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
                    out.append((m["ticker"], (ct - now()).total_seconds() / 3600))
            except Exception as e:
                log.warning(f"{s}: {e}")
                jlog("errors", where="markets", series=s, error=str(e)[:300])
        return out

    def cycle(self):
        mkts = self.markets()
        live = MODE != "dry"
        pos = self.k.positions() if live else {}
        rest = self.k.resting() if live else []
        mine = {}
        for o in rest:
            mine.setdefault((o["ticker"], order_side(o)), []).append(o)

        books, marks, bids = {}, {}, {}
        for tk, h in mkts:
            if MIN_H - 0.5 <= h <= MAX_H or tk in pos:
                try:
                    books[tk] = self.k.book(tk)
                except Exception as e:
                    log.warning(f"book {tk}: {e}")
                    continue
                y, n = books[tk]
                if y and n:
                    marks[tk] = (y[0][0] + 1 - n[0][0]) / 2
                bids[tk] = (y[0][0] if y else 0.0, n[0][0] if n else 0.0)

        # risk: budget and daily loss
        if live:
            bal = self.k.balance()
            pos_val = sum(p * marks.get(t, 0.5) if p > 0 else -p * (1 - marks.get(t, 0.5)) for t, p in pos.items())
            value = bal + pos_val
            used_mark = sum(collateral(p, marks.get(t, 0.5)) for t, p in pos.items())
            cost = getattr(self.k, "cost", {}) or {}
            # budget counts what positions COST (money actually tied up); mark value only if the cost field is missing
            used = sum(cost[t] if t in cost else collateral(p, marks.get(t, 0.5)) for t, p in pos.items() if p)
            today = now().date().isoformat()
            if self.day != today:
                self.day, self.day_start_value = today, value
            pnl_today = value - self.day_start_value
            # size off a CONSERVATIVE value: positions at what they'd sell for now (the bid), not the mid
            liq = bal + sum(p * bids.get(t, (0, 0))[0] if p > 0 else -p * bids.get(t, (0, 0))[1] for t, p in pos.items())
            budget, max_loss = budget_for(liq), loss_limit(self.day_start_value)
            jlog("snapshots", balance=round(bal, 2), position_value=round(pos_val, 2), value=round(value, 2),
                 pnl_today=round(pnl_today, 2), positions={t: p for t, p in pos.items() if p}, n_resting=len(rest),
                 budget=round(budget, 2), loss_limit=round(max_loss, 2), used_cost=round(used, 2), used_mark=round(used_mark, 2))
            if pnl_today < -max_loss:
                self.cancel_all(f"daily loss {pnl_today:.2f}")
                log.warning("daily loss limit hit; pausing until tomorrow (UTC)")
                notify(f"Daily loss limit hit ({pnl_today:+.2f}); orders cancelled, paused until tomorrow (UTC)", prio="high")
                return "paused"
            self.record_fills()
            if int(time.time()) % 3600 < LOOP_S * 2:
                self.record_settlements()
        else:
            used, budget, bal = 0.0, BUDGET, 0.0

        hours = dict(mkts)
        placed = cancelled = 0
        ev_held, ev_exp = {}, {}
        for t_, p_ in pos.items():
            ev_held[event_of(t_)] = ev_held.get(event_of(t_), 0.0) + abs(p_)
        ev_exp.update(ev_held)
        for o in rest:
            if adds_exposure(order_side(o), pos.get(o["ticker"], 0.0)):
                ev_exp[event_of(o["ticker"])] = ev_exp.get(event_of(o["ticker"]), 0.0) + remaining(o)
        resting_cost = sum(order_price(o, order_side(o)) * remaining(o) for o in rest)
        if live and used + resting_cost > budget:            # fills moved money into positions: trim
            for o in sorted(rest, key=lambda o: -order_price(o, order_side(o)) * remaining(o)):
                if used + resting_cost <= budget:
                    break
                try:
                    self.k.cancel(o["order_id"], o.get("ticker")); cancelled += 1
                    resting_cost -= order_price(o, order_side(o)) * remaining(o)
                    mine.get((o["ticker"], order_side(o)), []).remove(o) if o in mine.get((o["ticker"], order_side(o)), []) else None
                except Exception as e:
                    log.warning(f"trim cancel: {e}")
        for tk, (y, n) in books.items():
            want = targets(y, n, pos.get(tk, 0.0), hours.get(tk, -1))
            ev = event_of(tk)
            if live and ev_held.get(ev, 0.0) + SIZE > EVENT_CAP:     # event full: only quote sides that reduce it
                want = {sd: px for sd, px in want.items() if not adds_exposure(sd, pos.get(tk, 0.0))}
            if not live:
                if want:
                    jlog("dry_quotes", ticker=tk, hours_left=round(hours[tk], 2), best_yes_bid=y[0][0], best_no_bid=n[0][0],
                         depth_yes=y[:4], depth_no=n[:4], want=want)
                    placed += len(want)
                continue
            for side in ("yes", "no"):
                have = mine.get((tk, side), [])
                tgt = want.get(side)
                keep = None
                for o in have:
                    if tgt is not None and keep is None and abs(order_price(o, side) - tgt) < 0.005:
                        keep = o
                    else:
                        try:
                            self.k.cancel(o["order_id"], tk); cancelled += 1
                            resting_cost -= order_price(o, side) * remaining(o)
                        except Exception as e:
                            log.warning(f"cancel {o.get('order_id')}: {e}")
                if tgt is None or keep is not None:
                    continue
                if used + resting_cost + tgt * SIZE > budget:
                    continue
                if live and resting_cost + tgt * SIZE > 0.98 * bal:     # never rest more than the cash on hand
                    continue
                grows = adds_exposure(side, pos.get(tk, 0.0))
                if grows and ev_exp.get(ev, 0.0) + SIZE > EVENT_CAP:
                    continue
                try:
                    o = self.k.place(tk, side, tgt, SIZE)
                    resting_cost += tgt * SIZE; placed += 1
                    if grows:
                        ev_exp[ev] = ev_exp.get(ev, 0.0) + SIZE
                    jlog("orders", ticker=tk, side=side, price=tgt, count=SIZE, hours_left=round(hours[tk], 2),
                         best_yes_bid=y[0][0], best_no_bid=n[0][0], order_id=o.get("order_id"),
                         filled=o.get("fill_count"), remaining=o.get("remaining_count"))
                except Exception as e:
                    if "post only cross" in str(e):
                        # not an error: the book moved through our price before the order arrived, and post-only
                        # refused to let it trade as a taker -- the safety working. Logged as an event, not an error.
                        log.info(f"place {tk} {side} {tgt}: post-only refused (book moved)")
                        jlog("events", event="post_only_cross", ticker=tk, side=side, price=tgt)
                        continue
                    log.warning(f"place {tk} {side} {tgt}: {e}")
                    jlog("errors", where="place", ticker=tk, side=side, price=tgt, error=str(e)[:300])
        jlog("cycles", open_markets=len(mkts), books=len(books), placed=placed, cancelled=cancelled,
             resting_usd=round(resting_cost, 2), positions_usd=round(used, 2))
        log.info(f"{len(mkts)} open mkts, {len(books)} books, placed {placed}, cancelled {cancelled}, "
                 f"resting ${resting_cost:,.2f}, in positions ${used:,.2f}")
        return "ok"

    def record_fills(self):
        try:
            fs = self.k.fills(self.last_fill_ts - 5)
        except Exception as e:
            log.warning(f"fills: {e}"); return
        seen = set()
        path = f"{LOG_DIR}/fills.jsonl"
        if os.path.exists(path):
            seen = {json.loads(l).get("trade_id") or (json.loads(l).get("fill") or {}).get("trade_id")
                    for l in open(path) if l.strip()}
        for f in fs:
            if f.get("trade_id") in seen:
                continue
            jlog("fills", trade_id=f.get("trade_id"), fill=f)
            ts = f.get("created_time")
            if ts:
                self.last_fill_ts = max(self.last_fill_ts, int(dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()))

    def record_settlements(self):
        path = f"{LOG_DIR}/settlements.jsonl"
        seen = set()
        if os.path.exists(path):
            seen = {json.loads(l).get("key") for l in open(path) if l.strip()}
        try:
            ss = self.k.settlements(int(time.time()) - 14 * 86400)
        except Exception as e:
            log.warning(f"settlements: {e}"); return
        for x in ss:
            key = f"{x.get('ticker')}|{x.get('settled_time')}"
            if key not in seen:
                jlog("settlements", key=key, settlement=x)

    def run(self):
        log.info(f"MODE={MODE} series={','.join(SERIES)} offset={OFFSET * 100:.0f}c size={SIZE} cap={INV_CAP} event_cap={EVENT_CAP:.0f} "
                 f"budget={sizing_desc()}")
        jlog("events", event="start", params=dict(series=SERIES, offset=OFFSET, size=SIZE, inv_cap=INV_CAP, event_cap=EVENT_CAP,
                                                   budget=BUDGET, budget_pct=BUDGET_PCT, budget_max=BUDGET_MAX,
                                                   daily_loss=DAILY_LOSS, daily_loss_pct=DAILY_LOSS_PCT, min_h=MIN_H, max_h=MAX_H))
        t_end = time.time() + RUN_MIN * 60 if RUN_MIN > 0 else float("inf")
        paused_day = None
        killed = False
        if self.killed():
            log.warning("KILL set at start; cancelling and exiting")
            self.cancel_all("kill at start")
            sys.exit(KILLED_EXIT)
        notify(f"Started: budget {sizing_desc()}, offset {OFFSET * 100:.0f}c, size {SIZE}")
        try:
            while not self.stop and time.time() < t_end:
                if self.killed():
                    log.warning("KILL set"); killed = True; break
                if paused_day == now().date():
                    time.sleep(60); continue
                try:
                    if self.cycle() == "paused":
                        paused_day = now().date()
                    self.err_count = 0
                except Exception as e:
                    log.error(f"cycle error: {e}")
                    jlog("events", event="error", error=str(e)[:300])
                    self.err_count += 1
                    if self.err_count >= 5 and time.time() - self.last_err_alert > 3600:
                        self.last_err_alert = time.time()
                        notify(f"{self.err_count} cycle errors in a row; last: {str(e)[:300]}", prio="high")
                time.sleep(LOOP_S)
        finally:
            self.cancel_all("exit")
            jlog("events", event="stop", killed=killed)
            notify("Stopped by kill switch; all orders cancelled" if killed else "Stopped; all orders cancelled",
                   prio="high" if killed else "default")
        if killed:
            sys.exit(KILLED_EXIT)


if __name__ == "__main__":
    if MODE not in HOSTS:
        sys.exit(f"MODE must be one of {list(HOSTS)}")
    if MODE == "prod" and E("I_UNDERSTAND_REAL_MONEY") != "yes":
        sys.exit("prod mode needs I_UNDERSTAND_REAL_MONEY=yes")
    Bot().run()
