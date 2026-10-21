"""Probe: read the Polymarket US doc pages for trade search / price history / candles (are they public? fields?)."""
import re, urllib.request
for pg in ("api-reference/report/search-trades", "api-reference/report/download-trades-csv", "api-reference/price-history/get-price-history",
           "data-guide/candlestick-data", "api-reference/report/get-trade-stats"):
    try:
        t = urllib.request.urlopen(urllib.request.Request(f"https://docs.polymarket.us/{pg}.md", headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read().decode()
    except Exception as e:
        print(f"::warning::{pg}: {e}"); continue
    t = re.sub(r"\s+", " ", t)
    i = max(0, t.find("openapi") - 50)
    print(f"::notice::{pg} ({len(t)} chars): {t[:600]} ... {t[i:i+1600]}"[:3000])
