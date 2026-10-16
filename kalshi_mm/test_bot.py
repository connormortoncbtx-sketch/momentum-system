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


if __name__ == "__main__":
    for f in (test_targets, test_order_price, test_sign_roundtrip):
        f(); print("ok", f.__name__)
