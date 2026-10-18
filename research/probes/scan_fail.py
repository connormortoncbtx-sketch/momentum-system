"""Probe: why ~4,750 series never get cached by mm_scan. Runs scan_series on a sample of uncached series with full tracebacks."""
import os, sys, traceback, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kalshi import mm_scan as M
S = pd.read_parquet(f"{M.OUT}/series.parquet")
have = {f[3:-8] for f in os.listdir(M.OUT) if f.startswith("s2_")}
miss = S[~S.ticker.isin(have)].sample(40, random_state=3)
kinds = collections.Counter()
for s, fee in zip(miss.ticker, miss.get("fee_type", pd.Series("", index=miss.index))):
    info = ''
    try:
        mk = M.markets_for(s)
        info = f"markets {len(mk)}" + (f", vol sum {mk.volume.sum():,.0f}, vol NaN {mk.volume.isna().mean():.0%}, vol>0 {int((mk.volume > 0).sum())}" if len(mk) else "")
        t = M.scan_series(s, fee, np.random.default_rng(1))
        k = "ok" if t is not None else "none"
        print(f"::notice::{s}: {k} ({info})")
    except Exception as e:
        k = type(e).__name__ + ": " + str(e)[:80]
        tb = traceback.format_exc().strip().splitlines()
        print(f"::warning::{s}: {k} ({info}) at {tb[-3].strip()[:150]}")
    kinds[k.split(':')[0]] += 1
print(f"::notice::summary {dict(kinds)}")
