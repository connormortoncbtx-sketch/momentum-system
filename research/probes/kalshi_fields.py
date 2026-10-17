"""Probe: raw field names of settled markets (live + historical) for the scan's parser."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kalshi"))
import mm_scan as M
for s in ("KXHIGHNY", "KXNFLGAME", "KXBTCD"):
    for path in ("/markets", "/historical/markets"):
        q = {"series_ticker": s, "limit": 3}
        if path == "/markets":
            q["status"] = "settled"
        d = M.get(path, **q)
        ms = d.get("markets", [])
        m = ms[0] if ms else {}
        keep = {k: m.get(k) for k in m if any(w in k for w in ("vol", "result", "status", "close", "ticker", "settle"))}
        print(f"::notice::{s} {path}: n={len(ms)} keys={json.dumps(keep)[:900]}")
