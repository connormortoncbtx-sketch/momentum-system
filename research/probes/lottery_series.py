"""Probe: which candidate weather series for the weekly roll have open markets, and their spreads/prices."""
import json, urllib.request
API = "https://api.elections.kalshi.com/trade-api/v2"
C = ("KXHIGHTHOU,KXHIGHTDC,KXHIGHTPHX,KXHIGHTSATX,KXHIGHTSEA,KXHIGHTMIN,KXHIGHTOKC,KXHIGHTNOLA,KXHIGHTSFO,KXHIGHTEWR,"
     "KXHIGHHOU,KXHIGHTEMPDEN,KXHIGHTLAX,KXHIGHTMIA,KXHIGHTNY,KXHIGHTCHI").split(",")
for s in C:
    try:
        d = json.loads(urllib.request.urlopen(f"{API}/markets?series_ticker={s}&status=open&limit=200", timeout=30).read())
        mk = [m for m in d.get("markets", []) if float(m.get("yes_bid_dollars") or 0) > 0]
        sp = sorted(float(m["yes_ask_dollars"]) - float(m["yes_bid_dollars"]) for m in mk)
        mids = sorted(round((float(m["yes_ask_dollars"]) + float(m["yes_bid_dollars"])) / 2, 2) for m in mk)
        vol = sum(float(m.get("volume_fp") or m.get("volume") or 0) for m in d.get("markets", []))
        print(f"::notice::{s}: {len(d.get('markets', []))} open, vol {vol:,.0f}, median spread "
              f"{sp[len(sp) // 2] if sp else None}, top mids {mids[-5:]}")
    except Exception as e:
        print(f"::warning::{s}: {e}")
