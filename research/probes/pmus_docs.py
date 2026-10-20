"""Probe: pull the Polymarket US API reference pages and print the endpoint paths/params we need."""
import re, urllib.request
pages = ["api-reference/markets/get-markets", "api-reference/markets/get-market-book", "api-reference/markets/get-market-bbo",
         "api-reference/events/get-events", "api-reference/incentives/get-incentive-programs", "incentives/liquidity",
         "api-reference/market/overview"]
for pg in pages:
    try:
        t = urllib.request.urlopen(urllib.request.Request(f"https://docs.polymarket.us/{pg}.md", headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read().decode()
    except Exception as e:
        print(f"::warning::{pg}: {e}"); continue
    keep = [l.strip() for l in t.splitlines() if re.search(r"(GET|POST) /|/v\\d|openapi|gateway\\.|query|param|slug|limit|offset|cursor|reward|target|discount|period|spread|distance|score", l, re.I)]
    print(f"::notice::{pg}: " + " | ".join(dict.fromkeys(keep))[:900])
