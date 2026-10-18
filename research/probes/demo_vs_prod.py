"""Probe: does Kalshi's demo exchange mirror real prices? Same weather tickers, public orderbooks on both hosts."""
import json, urllib.request
H = {"prod": "https://api.elections.kalshi.com/trade-api/v2", "demo": "https://external-api.demo.kalshi.co/trade-api/v2"}
def get(host, path):
    try:
        with urllib.request.urlopen(urllib.request.Request(H[host] + path, headers={"Accept": "application/json"}), timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_err": str(e)[:80]}
def best(ob):
    o = ob.get("orderbook_fp") or ob.get("orderbook") or {}
    y = o.get("yes_dollars") or o.get("yes") or []; n = o.get("no_dollars") or o.get("no") or []
    f = lambda L: (max(float(p) for p, *_ in L), sum(float(q) for _, q, *_ in L)) if L else (None, 0)
    return f(y), f(n)
rows = 0
for s in ("KXHIGHNY", "KXHIGHCHI", "KXHIGHAUS", "KXHIGHMIA"):
    pm = {m["ticker"]: m for m in get("prod", f"/markets?series_ticker={s}&status=open&limit=50").get("markets", [])}
    dm = {m["ticker"]: m for m in get("demo", f"/markets?series_ticker={s}&status=open&limit=50").get("markets", [])}
    both = sorted(set(pm) & set(dm))
    print(f"::notice::{s}: prod open {len(pm)}, demo open {len(dm)}, same tickers {len(both)}")
    for tk in both[:6]:
        (py, pq), (pn, pnq) = best(get("prod", f"/markets/{tk}/orderbook"))
        (dy, dq), (dn, dnq) = best(get("demo", f"/markets/{tk}/orderbook"))
        print(f"::notice::{tk}: PROD yes bid {py} ({pq:.0f} deep) no bid {pn} | DEMO yes bid {dy} ({dq:.0f} deep) no bid {dn} | "
              f"vol24h prod {pm[tk].get('volume_24h_fp', pm[tk].get('volume_24h'))} demo {dm[tk].get('volume_24h_fp', dm[tk].get('volume_24h'))}")
        rows += 1
print(f"::notice::compared {rows} markets")
