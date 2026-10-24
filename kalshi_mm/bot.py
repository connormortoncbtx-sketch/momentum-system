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
import base64, datetime as dt, http.client, json, logging, os, signal, sys, threading, time, uuid
import urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lotfiles as LF  # noqa: E402  (the weekly roll's tickers and cash, so the bot leaves them out)

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
# only take on inventory in the FAVOURITE side (priced >= FAV_MIN): live fills (Oct 19-21) and the 2021-26 backtest
# agree -- fills that leave the bot holding a < 50c side lose (holdout -1.2 / -5.0c per city-day), favourite-side fills
# win (+7.6 / +4.8c); favourite-only = +6.0c per city-day (t 11.8). Orders that REDUCE a position are always allowed.
# FAV_MIN=0 turns the filter off.
FAV_MIN = float(E("FAV_MIN", "0.5"))
EVENT_CAP = float(E("EVENT_CAP", "30"))           # max total |contracts| across one event (city-day: all its buckets),
                                                  # held + resting orders that would add to it. The buckets of one
                                                  # city-day are one temperature, so they're one bet.


def event_of(ticker):
    return ticker.rsplit("-", 1)[0]


def favourite_only(want, pos):
    """Quote only sides priced >= FAV_MIN (price = what we'd pay for that side). This also drops a long-shot quote that
    would REDUCE a favourite position: netted, buying the long shot is the same bet as any long-shot fill (-1 to -5c
    per contract in the backtest), and it sells a favourite that is +EV to hold. Live check Oct 22: 46% of contracts
    filled after the switch were such reducing long-shot fills. Exits from old long-shot positions are favourite-side
    buys, so they are still quoted."""
    return {sd: px for sd, px in want.items() if px >= FAV_MIN}


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
LOOP_S = float(E("LOOP_SECONDS", "5"))             # pause between cycles (a cycle itself takes a few seconds)
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


LOTTERY_PREFIX = "lot-"                            # client_order_id prefix of kalshi_mm/lottery.py orders


class RateLimit:
    """Spaces requests to at most `per_s` per second across threads (Kalshi's basic tier allows ~20 reads, ~10 writes/s)."""
    def __init__(self, per_s):
        self.gap, self.next, self.lock = 1.0 / per_s, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            t = time.monotonic()
            slot = max(t, self.next)
            self.next = slot + self.gap
        if slot > t:
            time.sleep(slot - t)


READ_LIMIT = RateLimit(float(E("READ_PER_S", "10")))   # leaves headroom for the dashboard and weekly roll
ACCT_S = float(E("ACCOUNT_REFRESH_S", "4"))        # positions / orders / balance re-read at most this often (sooner
                                                   # right after we placed or cancelled); keeps the account under
                                                   # Kalshi's rate limit now that the stream makes cycles ~1s
WRITE_LIMIT = RateLimit(float(E("WRITE_PER_S", "8")))
BOOK_WORKERS = int(E("BOOK_WORKERS", "6"))         # order books fetched in parallel each cycle
MARKETS_TTL = 300                                  # the list of open markets changes slowly: refresh every 5 min
# QUIET_MINUTES="45-59": during those minutes past each hour, only quote sides that REDUCE a position (default off).
# Candidate fix for fills picked off around the hourly station readings -- see docs (Oct 22); turn on only if the
# favourite-only live data still shows it.
# USE_WS=yes: keep live order books from Kalshi's WebSocket and re-quote the moment a book changes (kalshi_mm/ws.py);
# any market without a trustworthy stream copy falls back to a normal request. MIN_GAP_S = shortest time between cycles.
USE_WS = E("USE_WS", "no").lower() == "yes"
LOG_EVERY_S = 60                                   # snapshots / quiet-cycle lines at most once a minute
MIN_GAP_S = float(E("MIN_GAP_S", "1"))
WS_CHECK_S = 60                                    # cross-check a few stream books against normal requests this often
QUIET = tuple(int(x) for x in E("QUIET_MINUTES", "").split("-")) if E("QUIET_MINUTES", "").strip() else None


def watchdog_ping():
    """Tell systemd the bot is alive (WatchdogSec in kmm.service restarts it if these stop). No-op outside systemd."""
    path = os.environ.get("NOTIFY_SOCKET")
    if not path:
        return
    import socket
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s_:
            s_.connect("\0" + path[1:] if path.startswith("@") else path)
            s_.sendall(b"WATCHDOG=1")
    except OSError:
        pass


def quiet_now(minute):
    return QUIET is not None and QUIET[0] <= minute <= QUIET[1]


# --- API client -----------------------------------------------------------------------------
class Kalshi:
    def __init__(self, mode):
        self.base = HOSTS[mode]
        self._tl = threading.local()
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

    def _conn(self, fresh=False):
        """One kept-alive HTTPS connection per thread: skips a TLS handshake (~3 round trips) on every request."""
        c = getattr(self._tl, "c", None)
        if fresh or c is None:
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
            scheme, host = self.base.split("://", 1)
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection   # http: local tests
            c = self._tl.c = cls(host, timeout=20)
        return c

    def req(self, method, path, body=None, auth=False, **q):
        path = PREFIX + path + (("?" + urllib.parse.urlencode(q)) if q else "")
        for i in range(4):
            (READ_LIMIT if method == "GET" else WRITE_LIMIT).wait()
            h = {"Accept": "application/json", "Content-Type": "application/json", "Connection": "keep-alive"}
            if auth:
                h.update(self._sign(method, path))
            try:
                c = self._conn()
                c.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=h)
                resp = c.getresponse()
                txt = resp.read()
            except (http.client.HTTPException, OSError) as e:      # dropped keep-alive, timeout, reset: reconnect
                self._conn(fresh=True)
                if i < 3:
                    time.sleep(0.5 * (i + 1)); continue
                raise RuntimeError(f"{method} {path} -> {e}")
            if resp.status >= 400:
                msg = txt.decode("utf-8", "replace")[:300]
                if resp.status in (429, 500, 502, 503, 504) and i < 3:
                    time.sleep(1.5 * (i + 1)); continue
                raise RuntimeError(f"{method} {path} -> {resp.status}: {msg}")
            return json.loads(txt) if txt else {}

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
        pv = d.get("portfolio_value_dollars", None)
        self.portfolio_value = num(pv) if pv is not None else (num(d.get("portfolio_value")) / 100 if d.get("portfolio_value") is not None else None)
        return num(d.get("balance")) / 100

    def positions(self):
        """{ticker: net position}; also sets self.cost = {ticker: $ cost basis} from Kalshi's market_exposure
        (what the position cost; used for the budget so paper gains don't crowd out new quotes)."""
        d = self.req("GET", "/portfolio/positions", auth=True, limit=1000, count_filter="position")
        mp = d.get("market_positions", [])
        # Kalshi nets mutually exclusive buckets: NO on several buckets of one city-day is partly a sure thing (at most
        # one bucket wins), and Kalshi pays that sure part back into cash early. Cash already includes it, so position
        # values must exclude it or it's counted twice (~$40 overstatement seen Oct 22). netted = sum of market
        # exposures - sum of event exposures.
        ev = d.get("event_positions", [])
        mx = sum(num(p.get("market_exposure_dollars")) for p in mp)
        ex = sum(num(e.get("event_exposure_dollars")) for e in ev)
        self.netted = max(0.0, mx - ex) if ev else 0.0
        self.cost = {}
        for p in mp:
            if p.get("market_exposure_dollars") is not None:
                self.cost[p["ticker"]] = num(p["market_exposure_dollars"])
            elif p.get("market_exposure") is not None:
                self.cost[p["ticker"]] = num(p["market_exposure"]) / 100
        return {p["ticker"]: num(p.get("position_fp", p.get("position"))) for p in mp}

    def resting(self, include_lottery=False):
        """Resting orders; the lottery's own orders (client_order_id 'lot-...') are left out unless asked for."""
        d = self.req("GET", "/portfolio/orders", auth=True, status="resting", limit=1000)
        return [o for o in d.get("orders", [])
                if include_lottery or not str(o.get("client_order_id") or "").startswith(LOTTERY_PREFIX)]

    def settlements(self, min_ts):
        d = self.req("GET", "/portfolio/settlements", auth=True, min_ts=min_ts, limit=1000)
        return d.get("settlements", [])

    def fills(self, min_ts):
        d = self.req("GET", "/portfolio/fills", auth=True, min_ts=min_ts, limit=1000)
        return d.get("fills", [])

    def place(self, ticker, side, price, count, coid_prefix=""):
        """side 'yes' = buy YES at price (book bid); side 'no' = buy NO at price, i.e. sell YES
        at 1 - price (book ask). V2 orders are expressed on the YES leg only. coid_prefix tags orders that belong to
        another component (the lottery uses 'lot-'), which the market maker then leaves alone."""
        yes_px = price if side == "yes" else 1 - price
        body = {"ticker": ticker, "side": "bid" if side == "yes" else "ask", "count": f"{count:.2f}",
                "price": f"{yes_px:.4f}", "time_in_force": "good_till_canceled",
                "self_trade_prevention_type": "maker", "post_only": True,
                "client_order_id": coid_prefix + str(uuid.uuid4()), "cancel_order_on_pause": True}
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


def trading_day(t=None):
    """The loss stop's day: the Central-time calendar date (was UTC, which rolled over at 7pm Central)."""
    from zoneinfo import ZoneInfo
    return (t or now()).astimezone(ZoneInfo("America/Chicago")).date().isoformat()


BASELINE_BASIS = "netted-lot-v1"   # bump when the value definition changes, so a stale baseline isn't compared to a new value


def day_baseline(day, value, path=None):
    """Start-of-day account value for the daily loss stop, saved to disk so a restart (kmm update, crash) on the same
    day keeps the original baseline instead of quietly resetting the stop to the value at the restart."""
    path = path or os.path.join(LOG_DIR, "day_start.json")
    try:
        d = json.load(open(path))
        if d.get("day") == day and d.get("basis") == BASELINE_BASIS and isinstance(d.get("value"), (int, float)):
            return float(d["value"])
    except Exception:
        pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump({"day": day, "value": round(value, 2), "basis": BASELINE_BASIS}, f)
    os.replace(path + ".tmp", path)
    return value


def collateral(pos, mark_yes):
    """Approximate money tied up in a position (cost basis unknown -> mark)."""
    return pos * mark_yes if pos > 0 else -pos * (1 - mark_yes)


# --- main loop --------------------------------------------------------------------------------
class Bot:
    def __init__(self):
        self.k = Kalshi(MODE)
        self.stop = False
        self.day, self.day_start_value = None, None
        self.mark_cache = {}
        self.last_fill_ts = int(time.time())
        self.t_start = time.time()
        self.last_stop_check, self.last_err_alert, self.err_count = 0.0, 0.0, 0
        self.stream, self.last_ws_check = None, 0.0
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

    def fallback_mark(self, ticker):
        """YES value for a position without a live mid: 1/0 once the result is in, else the last trade (cached 5 min)."""
        c = self.mark_cache.get(ticker)
        if c and (c[1] == "final" or time.time() - c[2] < 300):
            return c[0]
        try:
            m = self.k.req("GET", f"/markets/{ticker}").get("market", {})
        except Exception as e:
            log.warning(f"mark {ticker}: {e}"); return c[0] if c else None
        res = str(m.get("result", "")).lower()
        if res in ("yes", "no"):
            v, kind = (1.0 if res == "yes" else 0.0), "final"
        else:
            lp = num(m.get("last_price_dollars")) or num(m.get("last_price")) / 100
            v, kind = (lp if 0 < lp < 1 else None), "last"
        if v is not None:
            self.mark_cache[ticker] = (v, kind, time.time())
        return v

    def markets(self):
        """[(ticker, hours to close)]; the market list itself is cached MARKETS_TTL seconds (hours are recomputed)."""
        if time.time() - getattr(self, "_mk_t", 0) > MARKETS_TTL:
            got = []
            for s in SERIES:
                try:
                    for m in self.k.open_markets(s):
                        got.append((m["ticker"], dt.datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))))
                except Exception as e:
                    log.warning(f"{s}: {e}")
                    jlog("errors", where="markets", series=s, error=str(e)[:300])
            if got:
                self._mk, self._mk_t = got, time.time()
        n = now()
        return [(tk, (ct - n).total_seconds() / 3600) for tk, ct in getattr(self, "_mk", [])
                if ct > n]

    def cycle(self):
        mkts = self.markets()
        live = MODE != "dry"
        acct = getattr(self, "_acct", None)
        if live and (acct is None or getattr(self, "_acct_dirty", True) or time.time() - acct[3] > ACCT_S):
            acct = self._acct = (self.k.positions(), self.k.resting(), self.k.balance(), time.time())
            self._acct_dirty = False
        pos = acct[0] if live else {}
        rest = acct[1] if live else []
        # the weekly roll (lottery.py) shares the account but is not the bot's: its positions and its cash are left out
        # of everything below -- caps, budget, the loss stop, quoting (see lotfiles.py)
        lot_t = LF.tickers(LOG_DIR)
        lot_pos = {t: p for t, p in pos.items() if t in lot_t}
        if lot_pos:
            pos = {t: p for t, p in pos.items() if t not in lot_t}
        mine = {}
        for o in rest:
            mine.setdefault((o["ticker"], order_side(o)), []).append(o)

        t0 = time.time()
        books, marks, bids = {}, {}, {}
        need = [tk for tk, h in mkts if MIN_H - 0.5 <= h <= MAX_H or tk in pos]
        if USE_WS and not getattr(self, "ws_disabled", False):
            if self.stream is None:
                import ws as WS
                if WS.websocket is None:
                    log.warning("USE_WS=yes but websocket-client is not installed; using normal requests")
                else:
                    self.stream = WS.BookStream(self.k, MODE, LOG_DIR, log)
                    self.stream.start(need)
            else:
                self.stream.set_tickers(need)
        from_ws = 0

        def fetch(tk):
            b = self.stream.book(tk) if self.stream is not None else None
            if b is not None:
                return tk, ("ws", b)
            try:
                return tk, self.k.book(tk)
            except Exception as e:
                log.warning(f"book {tk}: {e}")
                return tk, None
        if getattr(self, "_pool", None) is None:             # long-lived workers keep their connections alive
            self._pool = ThreadPoolExecutor(BOOK_WORKERS)
        got = dict(self._pool.map(fetch, need))
        ws_tk = []
        for tk in list(got):
            if got[tk] is not None and got[tk][0] == "ws":
                got[tk] = got[tk][1]; from_ws += 1; ws_tk.append(tk)
        if ws_tk and time.time() - self.last_ws_check > WS_CHECK_S:
            self.last_ws_check = time.time()
            import random                              # a different random 3 each time, so live markets get checked too
            self.ws_check(random.sample(ws_tk, min(3, len(ws_tk))), got)
        for tk in need:
            if got.get(tk) is not None:
                books[tk] = got[tk]
                y, n = books[tk]
                if y and n and (1 - n[0][0]) - y[0][0] <= 0.10:
                    marks[tk] = (y[0][0] + 1 - n[0][0]) / 2
                bids[tk] = (y[0][0] if y else 0.0, n[0][0] if n else 0.0)
        # positions with no usable mid (market closed awaiting settlement, or a dead book with a 10c+ spread): value
        # them from the market's result once decided, else its last trade -- never a flat 50c (which made the daily
        # loss stop fire on a phantom -$62 when yesterday's closed markets settled overnight)
        for t in pos:
            if pos[t] and t not in marks:
                m = self.fallback_mark(t) if live else None
                if m is not None:
                    marks[t] = m
                    if t not in bids:                       # closed market: the decided value is what it pays
                        bids[t] = (m, 1 - m)

        # risk: budget and daily loss
        if live:
            bal_all = acct[2]
            lst = LF.state(LOG_DIR)
            ltk = lst.get("ticker")
            lres = self.fallback_mark(ltk) if ltk and LF.num(lst.get("held")) > 0 and not lot_pos.get(ltk) else None
            lot_cash = LF.cash_in_bal(lst, lot_pos, lres)
            bal = bal_all - lot_cash                               # the bot's share of the cash
            lot_value = lot_cash + sum(p * (self.fallback_mark(t) or LF.num(lst.get("price")) or 0.5)
                                       for t, p in lot_pos.items() if p > 0)   # logged only, for the whole-account line
            pos_val = sum(p * marks.get(t, 0.5) if p > 0 else -p * (1 - marks.get(t, 0.5)) for t, p in pos.items())
            net = getattr(self.k, "netted", 0.0) or 0.0           # sure payouts Kalshi already returned to cash
            pos_val -= net
            value = bal + pos_val
            used_mark = sum(collateral(p, marks.get(t, 0.5)) for t, p in pos.items())
            cost = getattr(self.k, "cost", {}) or {}
            # budget counts what positions COST (money actually tied up); mark value only if the cost field is missing
            used = max(0.0, sum(cost[t] if t in cost else collateral(p, marks.get(t, 0.5)) for t, p in pos.items() if p) - net)
            # size off a CONSERVATIVE value: positions at what they'd sell for now (the bid), not the mid
            liq = bal - net + sum(p * bids.get(t, (0, 0))[0] if p > 0 else -p * bids.get(t, (0, 0))[1] for t, p in pos.items())
            kv = getattr(self.k, "portfolio_value", None)
            today = trading_day()
            if self.day != today:
                self.day, self.day_start_value = today, day_baseline(today, value)
            pnl_today = value - self.day_start_value
            budget, max_loss = budget_for(liq), loss_limit(self.day_start_value)
            if time.time() - getattr(self, "_last_snap", 0) >= LOG_EVERY_S or pnl_today < -max_loss:
              self._last_snap = time.time()                # the stream makes cycles ~1s: log the account once a minute
              jlog("snapshots", balance=round(bal, 2), position_value=round(pos_val, 2), value=round(value, 2),
                 value_at_bid=round(liq, 2),
                 pnl_today=round(pnl_today, 2), positions={t: p for t, p in pos.items() if p}, n_resting=len(rest),
                 budget=round(budget, 2), loss_limit=round(max_loss, 2), used_cost=round(used, 2), used_mark=round(used_mark, 2),
                 netted=round(net, 2), kalshi_value=None if kv is None else round(bal_all + kv, 2),
                 lottery_cash=round(lot_cash, 2), lottery_value=round(lot_value, 2), lottery_positions=lot_pos or None)
            if pnl_today < -max_loss:
                self.cancel_all(f"daily loss {pnl_today:.2f}")
                log.warning("daily loss limit hit; pausing until midnight Central")
                notify(f"Daily loss limit hit ({pnl_today:+.2f}); orders cancelled, paused until midnight Central", prio="high")
                return "paused"
            if time.time() - getattr(self, "_last_fills", 0) > 10:
                self._last_fills = time.time()
                self.record_fills()
            if time.time() - getattr(self, "_last_settle", 0) > 3600:
                self._last_settle = time.time()
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
        quiet = quiet_now(now().minute)
        for tk, (y, n) in books.items():
            want = targets(y, n, pos.get(tk, 0.0), hours.get(tk, -1))
            if FAV_MIN > 0:
                want = favourite_only(want, pos.get(tk, 0.0))
            if quiet:                                          # observation window: only position-reducing quotes
                want = {sd: px for sd, px in want.items() if not adds_exposure(sd, pos.get(tk, 0.0))}
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
                            if "404" in str(e):                    # already filled or gone: nothing to cancel
                                log.info(f"cancel {o.get('order_id')}: already gone")
                            else:
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
        if placed or cancelled:
            self._acct_dirty = True                        # our own orders changed: re-read the account next cycle
        secs = round(time.time() - t0, 1)
        if not (placed or cancelled or time.time() - getattr(self, "_last_cyc", 0) >= LOG_EVERY_S):
            return "ok"                                    # quiet cycle: nothing to record (logs stay small)
        self._last_cyc = time.time()
        jlog("cycles", open_markets=len(mkts), books=len(books), placed=placed, cancelled=cancelled,
             resting_usd=round(resting_cost, 2), positions_usd=round(used, 2), secs=secs, quiet=quiet,
             from_ws=from_ws, ws=None if self.stream is None else dict(self.stream.stats))
        log.info(f"{len(mkts)} open mkts, {len(books)} books, placed {placed}, cancelled {cancelled}, "
                 f"resting ${resting_cost:,.2f}, in positions ${used:,.2f}, cycle {secs}s{' (quiet window)' if quiet else ''}"
                 + (f", {from_ws} books from stream" if self.stream is not None else ""))
        return "ok"

    def ws_check(self, tickers, got):
        """Compare the stream's copy of a few books with a normal request; log and drop any that disagree at the top."""
        bad = 0
        for tk in tickers:
            try:
                y, n = self.k.book(tk)
            except Exception:
                continue
            sy, sn = got[tk]
            top = lambda lv: lv[0][0] if lv else None
            ok = top(y) == top(sy) and top(n) == top(sn)
            jlog("ws_check", ticker=tk, ok=ok, rest=[top(y), top(n)], stream=[top(sy), top(sn)])
            if not ok:
                log.warning(f"ws_check {tk}: stream top {top(sy)}/{top(sn)} vs request {top(y)}/{top(n)} -- refreshing")
                with self.stream.lock:
                    self.stream.books.pop(tk, None)
                self.stream.stats["check_mismatch"] = self.stream.stats.get("check_mismatch", 0) + 1
                bad += 1
        if bad >= 2:                                   # not a one-off timing difference: rebuild every book from snapshots
            self.ws_bad = getattr(self, "ws_bad", 0) + 1
            if self.ws_bad >= 3:                       # keeps disagreeing after fresh snapshots: the stream can't be trusted
                log.error("ws_check: stream books keep disagreeing -- stream OFF, back to normal requests until restart")
                notify("Live order-book stream disagreed with Kalshi 3 checks running; switched off, bot continues on "
                       "normal requests. Run: kmm logs 50", prio="high")
                self.stream.stop(); self.stream = None; self.ws_disabled = True
                return
            log.warning("ws_check: several stream books disagree -- reconnecting the stream")
            self.stream.resync()
        else:
            self.ws_bad = 0

    def record_fills(self):
        try:
            fs = self.k.fills(self.last_fill_ts - 5)
        except Exception as e:
            log.warning(f"fills: {e}"); return
        path = f"{LOG_DIR}/fills.jsonl"
        if getattr(self, "_seen", None) is None:           # read the log once, then keep the set in memory
            self._seen = set()
            if os.path.exists(path):
                self._seen = {json.loads(l).get("trade_id") or (json.loads(l).get("fill") or {}).get("trade_id")
                              for l in open(path) if l.strip()}
        seen = self._seen
        for f in fs:
            if f.get("trade_id") in seen:
                continue
            seen.add(f.get("trade_id"))
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
                watchdog_ping()
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
                if self.stream is not None and self.stream.connected:
                    time.sleep(MIN_GAP_S)                      # then wait for the next book change (or LOOP_S at most)
                    self.stream.changed.wait(max(0.0, LOOP_S - MIN_GAP_S))
                    self.stream.changed.clear()
                else:
                    time.sleep(LOOP_S)
        finally:
            if self.stream is not None:
                self.stream.stop()
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
