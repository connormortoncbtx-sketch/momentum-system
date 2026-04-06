"""
automation/backfill_sectors.py
================================
ONE-TIME SCRIPT — populates missing sector and industry data for all
symbols in universe.csv that currently have empty sector fields.

The inject_universe.py script adds symbols with blank sectors because
it skips the yfinance .info API call for speed. This script fills them in.

Runtime: ~2-3 hours for 2,800 symbols at 1 call/sec with rate limiting.
Run once via GitHub Actions, never needs to run again (weekly upsert
will maintain sectors for new symbols going forward).

Run:
    python automation/backfill_sectors.py
    python automation/backfill_sectors.py --limit 500  # partial run
"""

import argparse
import logging
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR = Path("data")
UNIVERSE = DATA_DIR / "universe.csv"
SLEEP_PER_TICKER = 1.0   # seconds between API calls — 0.5s caused rate limiting at scale
BATCH_SIZE       = 50    # commit progress every N tickers


def fetch_sector_batch(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch sector and industry for a batch of symbols via yfinance.
    Returns {symbol: {sector, industry}} for symbols that returned data.
    """
    results = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            sector   = info.get("sector",   "") or ""
            industry = info.get("industry", "") or ""
            if sector:
                results[sym] = {"sector": sector, "industry": industry}
        except Exception as e:
            log.debug(f"  {sym}: {e}")
            continue
        time.sleep(SLEEP_PER_TICKER)
    return results


def run(limit: int = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not UNIVERSE.exists():
        log.error("No universe.csv found — run Stage 1 first")
        return

    universe = pd.read_csv(UNIVERSE)

    # Ensure sector/industry are string columns — pandas may infer float64
    # when all values are NaN, which breaks string assignment
    universe["sector"]   = universe["sector"].astype(str).replace("nan", "")
    universe["industry"] = universe["industry"].astype(str).replace("nan", "")

    log.info(f"Universe: {len(universe):,} symbols")

    # Find symbols with missing sector data
    missing_mask = (
        universe["sector"].isna() |
        (universe["sector"] == "") |
        (universe["sector"] == "Unknown")
    )
    missing = universe[missing_mask]["symbol"].tolist()
    log.info(f"Symbols missing sector data: {len(missing):,}")

    if not missing:
        log.info("All symbols already have sector data — nothing to do")
        return

    if limit:
        missing = missing[:limit]
        log.info(f"Processing first {limit} symbols (--limit flag)")

    log.info(f"Starting sector backfill for {len(missing):,} symbols...")
    log.info(f"Estimated runtime: {len(missing) * SLEEP_PER_TICKER / 60:.0f} minutes")

    filled   = 0
    not_found = 0

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE

        log.info(f"  Batch {batch_num}/{total_batches} ({i}/{len(missing)} symbols)...")

        results = fetch_sector_batch(batch)

        # Update universe in memory
        for sym, data in results.items():
            idx = universe[universe["symbol"] == sym].index
            if len(idx):
                universe.loc[idx, "sector"]   = data["sector"]
                universe.loc[idx, "industry"] = data["industry"]
                filled += 1

        not_found += len(batch) - len(results)

        # Save progress after every batch — if the job times out
        # we keep whatever was fetched so far
        universe.to_csv(UNIVERSE, index=False)

        if i > 0 and i % 500 == 0:
            log.info(f"  Progress saved: {filled} sectors filled so far")

    log.info("=" * 60)
    log.info(f"Sector backfill complete")
    log.info(f"  Filled:    {filled:,} symbols")
    log.info(f"  Not found: {not_found:,} symbols (no yfinance data)")
    log.info(f"  Still empty: {(universe['sector'] == '').sum():,} symbols")
    log.info(f"Saved → {UNIVERSE}")

    # Print sector distribution
    sector_counts = universe[universe["sector"] != ""]["sector"].value_counts().head(15)
    log.info("\nTop sectors in universe:")
    for sector, count in sector_counts.items():
        log.info(f"  {sector:<35} {count:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-time sector data backfill")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N missing symbols (for testing)")
    args = parser.parse_args()
    run(limit=args.limit)
