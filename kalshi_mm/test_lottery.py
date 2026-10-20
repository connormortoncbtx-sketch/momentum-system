"""Offline tests for lottery.py with a fake Kalshi client (no network)."""
import datetime as dt, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LOG_DIR", tempfile.mkdtemp())
import lottery as L

UTC = dt.timezone.utc


class Fake:
    def __init__(self):
        self.orders, self.placed, self.result, self.n = {}, [], {}, 0
        self.close = dt.datetime(2026, 10, 27, 5, 0, tzinfo=UTC)

    def open_markets(self, s):
        if s != L.LOT_SERIES[0]:
            return []
        c = self.close.isoformat().replace("+00:00", "Z"); d = f"{self.close:%d}"
        return [dict(ticker=f"{s}-{d}-A", close_time=c, yes_bid_dollars="0.30", yes_ask_dollars="0.33"),
                dict(ticker=f"{s}-{d}-B", close_time=c, yes_bid_dollars="0.54", yes_ask_dollars="0.57"),
                dict(ticker=f"{s}-{d}-C", close_time=c, yes_bid_dollars="0.60", yes_ask_dollars="0.70")]

    def book(self, tk):
        return [(0.54, 100)], [(0.43, 100)]                   # yes bid 54, yes ask 57

    def place(self, tk, side, px, n, coid_prefix=""):
        assert side == "yes" and coid_prefix == "lot-"
        self.n += 1; oid = f"o{self.n}"
        self.orders[oid] = dict(order_id=oid, status="resting", fill_count=0, remaining_count=n, count=n)
        self.placed.append((tk, px, n)); return {"order_id": oid}

    def cancel(self, oid, tk=None):
        self.orders[oid]["status"] = "canceled"; self.orders[oid]["remaining_count"] = 0

    def fill(self, oid, k):
        o = self.orders[oid]; o["fill_count"] += k; o["remaining_count"] -= k
        if o["remaining_count"] == 0: o["status"] = "executed"

    def req(self, method, path, **kw):
        if path.startswith("/portfolio/orders/"):
            return {"order": self.orders[path.split("/")[-1]]}
        if path.startswith("/markets/"):
            tk = path.split("/")[-1]
            close = dt.datetime(2026, 10, int(tk.split("-")[-2]), 5, 0, tzinfo=UTC)
            return {"market": dict(ticker=tk, close_time=close.isoformat().replace("+00:00", "Z"),
                                   result=self.result.get(tk, ""))}
        raise AssertionError(path)


def test_full_week():
    k, st = Fake(), L.load()
    now = dt.datetime(2026, 10, 26, 15, 0, tzinfo=UTC)          # Monday 10am CT
    st = L.step(k, st, now)
    assert st["status"] == "buying" and st["leg"] == 1
    tk, px, n = k.placed[-1]
    assert tk.endswith("-B") and px == 0.55 and n == 9, k.placed      # spread 3c -> improve bid by 1c; $5 // 0.55
    k.fill("o1", 4)
    st = L.step(k, st, now + dt.timedelta(minutes=15))
    assert st["held"] == 4 and st["status"] == "buying"
    k.fill("o1", 5)
    st = L.step(k, st, now + dt.timedelta(minutes=30))
    assert st["held"] == 9 and st["status"] == "holding" and abs(st["cash"] - 0.05) < 1e-6, st
    k.result[tk] = "yes"
    k.close = dt.datetime(2026, 10, 28, 5, 0, tzinfo=UTC)          # next day's markets are listed
    st = L.step(k, st, now + dt.timedelta(hours=20))
    assert st["leg"] == 2 and st["status"] == "buying" and abs(st["cash"] - 9.05) < 1e-6, st
    # leg 2: placed with the rolled cash; misses -> week over
    st = L.step(k, st, now + dt.timedelta(hours=20, minutes=15))
    assert k.placed[-1][2] == 16                                   # $9.05 // 0.55
    oid = st["order_id"]; k.fill(oid, 16)
    st = L.step(k, st, now + dt.timedelta(hours=21))
    assert st["status"] == "holding"
    k.result[st["ticker"]] = "no"
    st = L.step(k, st, now + dt.timedelta(hours=40))
    assert st["status"] == "done"
    # still the same week: nothing new
    st = L.step(k, st, now + dt.timedelta(days=2))
    assert st["status"] == "done"
    # next Monday 9am CT: new roll
    k.close = dt.datetime(2026, 10, 31, 5, 0, tzinfo=UTC)
    st = L.step(k, st, dt.datetime(2026, 10, 30, 15, 30, tzinfo=UTC) + dt.timedelta(days=3))
    assert st["status"] == "buying" and st["leg"] == 1 and st["cash"] == L.STAKE


def test_stale_reprice_and_kill():
    k, st = Fake(), {"status": "idle", "week": None, "leg": 0, "cash": 0.0, "history": []}
    now = dt.datetime(2026, 10, 26, 15, 0, tzinfo=UTC)
    st = L.step(k, st, now)
    k.fill("o1", 2)
    st = L.step(k, st, now + dt.timedelta(hours=3))               # stale: cancel, count fills, re-place remainder
    assert k.orders["o1"]["status"] == "canceled" and st["held"] == 2 and k.placed[-1][2] == 7, (st, k.placed)
    open(L.B.KILL_FILE, "w").close()
    try:
        st = L.step(k, st, now + dt.timedelta(hours=3, minutes=15))
        assert st["order_id"] is None and k.orders["o2"]["status"] == "canceled"
    finally:
        os.remove(L.B.KILL_FILE)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
