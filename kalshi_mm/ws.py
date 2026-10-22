"""
kalshi_mm/ws.py -- live order books from Kalshi's WebSocket (orderbook_delta channel), for the market maker.

Instead of asking for every book every cycle (rate-limited: ~9s per pass over ~130 markets), the bot keeps a copy of
each book updated by Kalshi's push messages, and wakes the moment one changes. Safety:
  - a book is only served when it has a snapshot and the subscription's message sequence has no gap; a gap, a
    disconnect or a parse problem drops the copy and the bot falls back to a normal request for that market
  - the bot cross-checks a few stream books against normal requests every minute and logs mismatches (ws_check)
  - off unless USE_WS=yes
The first messages received are saved to LOG_DIR/ws_sample.jsonl so the message format can be verified.
Field formats: Kalshi has used both cent integers and dollar strings; both are accepted.
"""
import json
import os
import threading
import time

try:
    import websocket          # pip: websocket-client
except ImportError:           # optional dependency: the bot runs without it (USE_WS off)
    websocket = None

PATH = "/trade-api/ws/v2"
WS_HOSTS = {"prod": "wss://external-api-ws.kalshi.com", "demo": "wss://external-api-ws.demo.kalshi.co",
            "dry": "wss://api.elections.kalshi.com"}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _levels(msg, side):
    """[(price $, qty)] for 'yes' or 'no' from a snapshot, whichever field shape is present; None if no field is."""
    for key, scale in ((f"{side}_dollars_fp", 1), (f"{side}_dollars", 1), (side, 100)):
        if msg.get(key) is not None:
            out = []
            for lv in msg[key]:
                p, q = _num(lv[0]), _num(lv[1])
                if p is not None and q is not None:
                    out.append((round(p / scale, 4), q))
            return out
    return None


def _delta_fields(msg):
    """(side 'yes'/'no', price $, qty change) from a delta message, or None."""
    side = msg.get("side")
    if side not in ("yes", "no"):
        return None
    if msg.get("price_dollars") is not None:
        p = _num(msg["price_dollars"])
    else:
        p = _num(msg.get("price"))
        p = None if p is None else (p / 100 if p >= 1 else p)
    d = _num(msg.get("delta_fp", msg.get("delta")))
    if p is None or d is None:
        return None
    return side, round(p, 4), d


class BookStream:
    def __init__(self, kalshi, mode, log_dir, log=None):
        self.k, self.url = kalshi, os.environ.get("WS_URL") or WS_HOSTS[mode] + PATH
        self.log_dir, self.log = log_dir, log
        self.lock = threading.Lock()
        self.books = {}            # ticker -> {"yes": {price: qty}, "no": {...}}
        self.sid_seq = {}          # sid -> last seq
        self.tickers = []
        self.changed = threading.Event()
        self.connected = False
        self.ws = None
        self.samples = 0
        self.stats = {"msgs": 0, "gaps": 0, "reconnects": 0, "errors": 0}
        self._stop = False
        self._want = None          # tickers to (re)subscribe

    # --- public API (bot thread) -------------------------------------------------------
    def start(self, tickers):
        self.tickers = sorted(tickers)
        threading.Thread(target=self._run, daemon=True).start()

    def set_tickers(self, tickers):
        """Market list changed: reconnect with the new list (simplest correct way to resubscribe)."""
        t = sorted(tickers)
        if t != self.tickers:
            self.tickers = t
            self._drop_all()
            try:
                self.ws and self.ws.close()
            except Exception:
                pass

    def resync(self):
        """Throw away every copy and reconnect for fresh snapshots."""
        self._drop_all()
        try:
            self.ws and self.ws.close()
        except Exception:
            pass

    def book(self, ticker):
        """(yes levels best-first, no levels best-first) or None if this market has no trustworthy copy."""
        with self.lock:
            b = self.books.get(ticker) if self.connected else None
            if b is None:
                return None
            yes = sorted(((p, q) for p, q in b["yes"].items() if q > 0), reverse=True)
            no = sorted(((p, q) for p, q in b["no"].items() if q > 0), reverse=True)
        return yes, no

    def coverage(self):
        with self.lock:
            return len(self.books) if self.connected else 0

    def stop(self):
        self._stop = True
        try:
            self.ws and self.ws.close()
        except Exception:
            pass

    # --- stream thread -------------------------------------------------------------------
    def _drop_all(self):
        with self.lock:
            self.books.clear(); self.sid_seq.clear()

    def _note(self, msg):
        if self.log:
            self.log.info(msg)

    def _sample(self, raw):
        if self.samples < 40:
            self.samples += 1
            try:
                with open(os.path.join(self.log_dir, "ws_sample.jsonl"), "a") as f:
                    f.write(raw.strip()[:4000] + "\n")
            except Exception:
                pass

    def _run(self):
        back = 1.0
        while not self._stop:
            try:
                h = self.k._sign("GET", PATH)
                self.ws = websocket.create_connection(self.url, header=[f"{k}: {v}" for k, v in h.items()], timeout=30,
                                                      enable_multithread=True)
                self.ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                         "params": {"channels": ["orderbook_delta"], "market_tickers": self.tickers}}))
                self.connected, back = True, 1.0
                self._note(f"ws: connected, {len(self.tickers)} markets")
                last_ping = time.time()
                while not self._stop:
                    try:
                        raw = self.ws.recv()
                    except websocket.WebSocketTimeoutException:
                        raw = None
                    if time.time() - last_ping > 15:
                        self.ws.ping(); last_ping = time.time()
                    if not raw:
                        continue
                    self._handle(raw)
            except Exception as e:
                self.stats["errors"] += 1
                if not self._stop:
                    self._note(f"ws: {type(e).__name__}: {str(e)[:200]} -- reconnecting in {back:.0f}s")
            finally:
                self.connected = False
                self._drop_all()
                try:
                    self.ws and self.ws.close()
                except Exception:
                    pass
            if self._stop:
                return
            self.stats["reconnects"] += 1
            time.sleep(back); back = min(back * 2, 60)

    def _handle(self, raw):
        self._sample(raw)
        m = json.loads(raw)
        typ, msg = m.get("type"), m.get("msg") or {}
        if typ == "error":
            self.stats["errors"] += 1
            self._note(f"ws error: {msg}")
            return
        if typ not in ("orderbook_snapshot", "orderbook_delta"):
            return
        self.stats["msgs"] += 1
        sid, seq, tk = m.get("sid"), m.get("seq"), msg.get("market_ticker")
        with self.lock:
            if sid is not None and seq is not None:
                last = self.sid_seq.get(sid)
                if typ == "orderbook_delta" and last is not None and seq != last + 1:
                    # missed a message: every book on this subscription is suspect -> reconnect for fresh snapshots
                    self.stats["gaps"] += 1
                    raise RuntimeError(f"sequence gap on sid {sid}: {last} -> {seq}")
                self.sid_seq[sid] = seq
            if typ == "orderbook_snapshot":
                y, n = _levels(msg, "yes"), _levels(msg, "no")
                if y is None and n is None:            # unknown shape: never serve a made-up empty book
                    self.stats["unreadable"] = self.stats.get("unreadable", 0) + 1
                    self.books.pop(tk, None)
                else:                                  # a side with no field at all = no orders on that side
                    self.books[tk] = {"yes": dict(y or []), "no": dict(n or [])}
            else:
                b = self.books.get(tk)
                f = _delta_fields(msg)
                if b is None or f is None:
                    if b is not None:          # unreadable delta: stop trusting this market until a new snapshot
                        self.books.pop(tk, None)
                    return
                side, p, d = f
                b[side][p] = b[side].get(p, 0.0) + d
                if b[side][p] <= 1e-9:
                    b[side].pop(p, None)
        self.changed.set()
