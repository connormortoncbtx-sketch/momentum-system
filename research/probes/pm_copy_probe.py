"""Probe: Polymarket public data for a copy-the-top-traders test (Oct 20, 2026). Public data API only (no Kalshi:
its data terms prohibit harvesting the website). Prints which endpoints answer and their fields."""
import json, urllib.request, urllib.parse, traceback

UA = {"User-Agent": "Mozilla/5.0 research", "Accept": "application/json"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
        return json.loads(r.read())


def show(name, url):
    try:
        d = get(url)
        x = d[0] if isinstance(d, list) and d else d
        n = len(d) if isinstance(d, list) else (len(d.get("data", [])) if isinstance(d, dict) and "data" in d else "-")
        print(f"::notice::{name}: OK n={n} sample={json.dumps(x)[:700]}")
        return d
    except Exception as e:
        print(f"::warning::{name}: {type(e).__name__} {str(e)[:200]}")


D = "https://data-api.polymarket.com"
lb = None
for name, u in [("lb v1 profit month", f"{D}/v1/leaderboard?timePeriod=MONTH&orderBy=PNL&limit=5"),
                ("lb v1 vol all", f"{D}/v1/leaderboard?timePeriod=ALL&orderBy=VOL&limit=5"),
                ("lb-api profit", "https://lb-api.polymarket.com/profit?window=all&limit=5"),
                ("lb-api volume", "https://lb-api.polymarket.com/volume?window=all&limit=5")]:
    r = show(name, u)
    lb = lb or r
tr = show("global trades", f"{D}/trades?limit=5")
user = None
for src in (lb, tr):
    if isinstance(src, list) and src:
        for k in ("proxyWallet", "proxy_wallet", "user", "address"):
            if k in src[0]:
                user = src[0][k]; break
    if user: break
print(f"::notice::user for detail probes: {user}")
if user:
    show("user trades", f"{D}/trades?user={user}&limit=5")
    show("user activity", f"{D}/activity?user={user}&limit=5")
    show("user closed positions", f"{D}/closed-positions?user={user}&limit=5")
    a = show("user activity old", f"{D}/activity?user={user}&limit=5&offset=3000")
tok = None
if isinstance(tr, list) and tr:
    t0 = tr[0]
    tok = t0.get("asset"); cid = t0.get("conditionId")
    show("gamma market by condition", f"https://gamma-api.polymarket.com/markets?condition_ids={cid}")
    if tok:
        ts = int(t0.get("timestamp", 0))
        show("clob prices-history", f"https://clob.polymarket.com/prices-history?market={tok}&startTs={ts-3600}&endTs={ts+3600}&fidelity=1")
