"""Probe: why did mm_scan find no volume? Run markets_for / trades_for on known-busy series."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kalshi"))
import mm_scan as M
for s in ("KXNFLGAME", "KXHIGHNY"):
    t = time.time()
    mk = M.markets_for(s)
    print(f"::notice::{s}: markets_for -> {len(mk)} rows, cols {list(mk.columns)}, vol sum {mk.volume.sum() if len(mk) else 0:,.0f}, {time.time() - t:.0f}s")
    raw = M.get("/markets", series_ticker=s, status="settled", limit=1000)
    print(f"::notice::{s}: raw /markets limit=1000 -> keys {list(raw)[:5]}, n {len(raw.get('markets', []))}")
    raw = M.get("/historical/markets", series_ticker=s, limit=1000)
    print(f"::notice::{s}: raw /historical/markets limit=1000 -> keys {list(raw)[:5]}, n {len(raw.get('markets', []))}")
