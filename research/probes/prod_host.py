"""Probe: which Kalshi production host answers (bot MODE=prod uses HOSTS['prod'])."""
import urllib.request
for h in ("https://external-api.kalshi.com", "https://api.elections.kalshi.com", "https://trading-api.kalshi.com"):
    for path in ("/trade-api/v2/exchange/status", "/trade-api/v2/portfolio/balance"):
        try:
            with urllib.request.urlopen(urllib.request.Request(h + path, headers={"Accept": "application/json"}), timeout=20) as r:
                print(f"::notice::{h}{path}: {r.status} {r.read()[:80]!r}")
        except urllib.error.HTTPError as e:
            print(f"::notice::{h}{path}: HTTP {e.code} {e.read()[:80]!r}")
        except Exception as e:
            print(f"::notice::{h}{path}: ERR {str(e)[:80]}")
