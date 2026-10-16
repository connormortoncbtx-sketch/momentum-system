"""research/probes/kalshi_trades.py -- what trade-level history does Kalshi's public API give? Read-only."""
import json, urllib.request, urllib.parse
API = "https://api.elections.kalshi.com/trade-api/v2"


def get(path, **q):
    url = f"{API}{path}?{urllib.parse.urlencode(q)}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=60) as r:
            return r.status, json.loads(r.read())
    except Exception as e:
        return getattr(e, "code", str(e)[:60]), {}


for path, t in (("/historical/trades", "KXHIGHNY-26JUL02-B101.5"), ("/historical/trades", "KXHIGHNY-25JUN01-T66"),
                ("/historical/trades", "HIGHNY-24AUG03-B90.5"), ("/markets/trades", "KXHIGHNY-25JUN01-T66")):
    st, d = get(path, ticker=t, limit=1000)
    tr = d.get("trades", [])
    print(f"::notice::{path} {t}: status {st}, {len(tr)} trades, keys {sorted(tr[0].keys()) if tr else None}, "
          f"cursor {'yes' if d.get('cursor') else 'no'}; first {tr[0] if tr else ''}")
st, d = get("/series/KXHIGHNY")
print(f"::notice::series status {st}: fee_type {d.get('series', {}).get('fee_type')}, fee_multiplier {d.get('series', {}).get('fee_multiplier')}")
st, d = get("/historical/cutoff")
print(f"::notice::historical cutoff {st}: {d}")
