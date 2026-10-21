"""Probe: does Polymarket US expose public trade history (with aggressor side) or price history -- needed to test
market making there? Reads the docs index and tries likely endpoints on one active, liquid market."""
import json, urllib.parse, urllib.request
G = "https://gateway.polymarket.us"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def raw(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:200]
    except Exception as e:
        return None, str(e)[:200]


for idx in ("https://docs.polymarket.us/llms.txt", "https://docs.polymarket.us/sitemap.xml"):
    st, t = raw(idx)
    hits = [l.strip() for l in t.splitlines() if any(k in l.lower() for k in ("trade", "history", "candle", "price", "fill", "activity", "ohlc"))]
    print(f"::notice::{idx} -> {st}; {len(t)} chars; relevant: {' | '.join(hits[:25])[:1500]}")

st, t = raw(f"{G}/v1/markets?limit=200&active=true&closed=false")
mk = json.loads(t).get("markets", []) if st == 200 else []
mk = sorted(mk, key=lambda m: -float(m.get("volume24hr") or m.get("volumeNum") or m.get("volume") or 0))
slug = mk[0]["slug"] if mk else None
mid = mk[0].get("id") if mk else None
print(f"::notice::test market {slug} id {mid}; keys {sorted(mk[0].keys())[:50] if mk else None}")
for path in (f"/v1/markets/{slug}/trades", f"/v1/trades?marketSlug={slug}", f"/v1/trades?market={slug}",
             f"/v1/markets/{slug}/history", f"/v1/markets/{slug}/prices-history", f"/v1/markets/{slug}/candles",
             f"/v1/markets/{slug}/activity", f"/v1/activity?marketSlug={slug}", f"/v1/markets/{slug}/book"):
    st, t = raw(G + path)
    print(f"::notice::{path} -> {st}: {t[:400]}")
