"""Offline tests for kalshi_mm/bot.py quoting logic and request signing. Run: python kalshi_mm/test_bot.py"""
import base64, os, sys
os.environ.setdefault("MODE", "dry")
sys.path.insert(0, os.path.dirname(__file__))
import bot as B


def test_targets():
    y, n = [(0.40, 100), (0.38, 50)], [(0.55, 80)]           # yes ask = 0.45
    t = B.targets(y, n, 0, 10)
    assert t == {"yes": round(0.40 - B.OFFSET, 2), "no": round(0.55 - B.OFFSET, 2)}, t
    assert B.targets(y, n, 0, 0.5) == {}                       # too close to close
    assert B.targets(y, n, 0, 60) == {}                        # too far out
    assert "yes" not in B.targets(y, n, B.INV_CAP, 10)         # long cap: no more YES
    assert "no" not in B.targets(y, n, -B.INV_CAP, 10)         # short cap: no more NO
    assert B.targets([(0.01, 5)], [(0.98, 5)], 0, 10) == {}    # mid ~1.5c: outside band
    assert B.targets([(0.60, 5)], [(0.45, 5)], 0, 10) == {}    # crossed book
    assert B.targets([], n, 0, 10) == {}
    t = B.targets([(0.02, 5)], [(0.90, 5)], 0, 10)            # mid 6c: yes price would be 0 -> only NO
    assert t == {"no": round(0.90 - B.OFFSET, 2)}, t


def test_order_side():
    assert B.order_side({"book_side": "bid"}) == "yes"
    assert B.order_side({"book_side": "ask"}) == "no"
    assert B.order_side({"side": "yes", "action": "sell"}) == "no"
    assert B.order_price({"yes_price_dollars": "0.3000"}, "no") == 0.70


def test_order_price():
    assert B.order_price({"yes_price_dollars": "0.3700"}, "yes") == 0.37
    assert B.order_price({"no_price": 52}, "no") == 0.52


def test_sign_roundtrip():
    from cryptography.hazmat.primitives.asymmetric import ed25519, rsa, padding
    from cryptography.hazmat.primitives import hashes
    k = B.Kalshi.__new__(B.Kalshi)
    k.key_id = "test"
    for key in (ed25519.Ed25519PrivateKey.generate(), rsa.generate_private_key(public_exponent=65537, key_size=2048)):
        k.key = key
        h = k._sign("GET", "/trade-api/v2/portfolio/orders?status=resting")
        msg = (h["KALSHI-ACCESS-TIMESTAMP"] + "GET/trade-api/v2/portfolio/orders").encode()
        sig = base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"])
        if isinstance(key, ed25519.Ed25519PrivateKey):
            key.public_key().verify(sig, msg)
        else:
            key.public_key().verify(sig, msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                          salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())


def test_event_cap_helpers():
    from bot import event_of, adds_exposure
    assert event_of("KXHIGHPHIL-26OCT19-B66.5") == event_of("KXHIGHPHIL-26OCT19-T67") == "KXHIGHPHIL-26OCT19"
    assert event_of("KXHIGHPHIL-26OCT20-B72.5") != "KXHIGHPHIL-26OCT19"
    assert adds_exposure("yes", 0) and adds_exposure("no", 0)       # flat: any fill adds
    assert adds_exposure("yes", 5) and not adds_exposure("no", 5)   # long YES: buying NO reduces
    assert adds_exposure("no", -5) and not adds_exposure("yes", -5)


def test_day_baseline_survives_restart():
    import tempfile, datetime as dt
    p = os.path.join(tempfile.mkdtemp(), "day_start.json")
    assert B.day_baseline("2026-10-20", 380.0, p) == 380.0          # first sight of the day: saved
    assert B.day_baseline("2026-10-20", 367.0, p) == 380.0          # restart same day: original baseline kept
    assert B.day_baseline("2026-10-21", 366.0, p) == 366.0          # new day: new baseline
    t = dt.datetime(2026, 10, 21, 1, 30, tzinfo=dt.timezone.utc)    # 8:30pm Central on Oct 20
    assert B.trading_day(t) == "2026-10-20"


def test_fallback_mark():
    class K:
        calls = 0
        def req(self, m, path, **q):
            K.calls += 1
            tk = path.split("/")[-1]
            return {"market": {"T-YES": {"result": "yes"}, "T-NO": {"result": "no"},
                               "T-OPEN": {"result": "", "last_price_dollars": "0.07"}, "T-NONE": {"result": ""}}[tk]}
    b = B.Bot.__new__(B.Bot); b.k = K(); b.mark_cache = {}
    assert b.fallback_mark("T-YES") == 1.0 and b.fallback_mark("T-NO") == 0.0
    assert abs(b.fallback_mark("T-OPEN") - 0.07) < 1e-9 and b.fallback_mark("T-NONE") is None
    n = K.calls; b.fallback_mark("T-YES"); b.fallback_mark("T-OPEN")
    assert K.calls == n                                  # decided results cached for good, last trades for 5 min


def test_favourite_only():
    want = {"yes": 0.18, "no": 0.78}                     # a 20c bucket: bid YES 18c (long shot), bid NO 78c (favourite)
    assert B.favourite_only(want, 0.0) == {"no": 0.78}    # flat: only the favourite-side quote
    assert B.favourite_only(want, -10.0) == {"yes": 0.18, "no": 0.78}  # short YES (holding NO): buying YES reduces -> kept
    want = {"yes": 0.70, "no": 0.26}
    assert B.favourite_only(want, 0.0) == {"yes": 0.70}
    assert B.favourite_only(want, 8.0) == {"yes": 0.70, "no": 0.26}   # long YES: buying NO reduces -> kept


if __name__ == "__main__":
    for f in (test_targets, test_order_side, test_order_price, test_event_cap_helpers, test_sign_roundtrip,
              test_day_baseline_survives_restart, test_fallback_mark, test_favourite_only):
        f(); print("ok", f.__name__)
