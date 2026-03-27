"""
Stage 1 — Universe Builder
===========================
Pulls all NYSE + NASDAQ tickers, applies a liquidity/quality gate,
and writes two artifacts:

    data/universe_raw.csv      — every ticker with metadata
    data/universe.csv          — filtered, tradeable tickers only

Liquidity gate (all must pass):
    - Price >= $5.00
    - 20-day avg volume >= 500,000
    - Market cap >= $300M
    - Not a fund / ETF / warrant / unit (symbol heuristics)
    - Has at least 60 days of price history

Run:  python pipeline/01_universe.py
"""

import os
import time
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

MIN_PRICE       = 5.0
MIN_AVG_VOL     = 500_000
MIN_MARKET_CAP  = 300_000_000   # $300M
MIN_HISTORY_DAYS = 60
BATCH_SIZE      = 100           # tickers per yfinance batch call
BATCH_SLEEP     = 1.0           # seconds between batches (rate limit courtesy)

DATA_DIR   = Path("data")
OUTPUT_RAW = DATA_DIR / "universe_raw.csv"
OUTPUT     = DATA_DIR / "universe.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── TICKER SOURCES ────────────────────────────────────────────────────────────

def fetch_nasdaq_tickers() -> pd.DataFrame:
    """
    Pull NASDAQ-listed tickers from the NASDAQ FTP listing.
    Returns DataFrame with columns: symbol, name, exchange, sector, industry.
    """
    url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
    try:
        df = pd.read_csv(url)
        df = df.rename(columns={
            "Symbol": "symbol",
            "Company Name": "name",
        })
        df["exchange"] = "NASDAQ"
        df["sector"]   = ""
        df["industry"] = ""
        return df[["symbol", "name", "exchange", "sector", "industry"]]
    except Exception:
        log.warning("NASDAQ FTP failed, falling back to ftp.nasdaqtrader.com")
        return _fetch_nasdaq_ftp()


def _fetch_nasdaq_ftp() -> pd.DataFrame:
    """Fallback: parse nasdaqtrader.com pipe-delimited listing."""
    import urllib.request
    url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    with urllib.request.urlopen(url, timeout=15) as r:
        raw = r.read().decode("utf-8")
    rows = []
    for line in raw.strip().split("\n")[1:]:          # skip header
        parts = line.split("|")
        if len(parts) < 3 or parts[0] == "File Creation Time":
            continue
        symbol = parts[0].strip()
        name   = parts[1].strip()
        rows.append({"symbol": symbol, "name": name,
                     "exchange": "NASDAQ", "sector": "", "industry": ""})
    return pd.DataFrame(rows)


def fetch_nyse_tickers() -> pd.DataFrame:
    """
    Pull NYSE-listed tickers from nasdaqtrader otherlisted file
    (covers NYSE, ARCA, BATS, etc.).
    """
    import urllib.request
    url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            raw = r.read().decode("utf-8")
        rows = []
        for line in raw.strip().split("\n")[1:]:
            parts = line.split("|")
            if len(parts) < 5 or parts[0] == "File Creation Time":
                continue
            symbol   = parts[0].strip()
            name     = parts[1].strip()
            exchange = parts[2].strip()
            rows.append({"symbol": symbol, "name": name,
                         "exchange": exchange, "sector": "", "industry": ""})
        return pd.DataFrame(rows)
    except Exception as e:
        log.warning(f"NYSE fetch failed: {e}")
        return pd.DataFrame(columns=["symbol", "name", "exchange", "sector", "industry"])


def combine_and_clean(nasdaq: pd.DataFrame, nyse: pd.DataFrame) -> pd.DataFrame:
    """Merge, deduplicate, strip garbage symbols."""
    df = pd.concat([nasdaq, nyse], ignore_index=True)
    df["symbol"] = df["symbol"].str.strip().str.upper()

    # Drop blanks
    df = df[df["symbol"].str.len() > 0]

    # Drop obvious non-stocks: warrants (W suffix), units (U), rights (R),
    # preferred shares (contain a dash), test issues (ZZ prefix)
    bad = (
        df["symbol"].str.endswith("W")  |
        df["symbol"].str.endswith("U")  |
        df["symbol"].str.endswith("R")  |
        df["symbol"].str.contains(r"\.", regex=True)  |   # BRK.B style
        df["symbol"].str.contains("-",  regex=False)  |   # preferred shares
        df["symbol"].str.startswith("ZZ")
    )
    df = df[~bad]

    # Deduplicate — prefer NASDAQ listing if symbol appears on both
    df = df.drop_duplicates(subset="symbol", keep="first")
    df = df.reset_index(drop=True)

    log.info(f"Combined universe: {len(df):,} symbols after dedup + symbol filter")
    return df


# ── LIQUIDITY GATE ────────────────────────────────────────────────────────────

def gate_batch(symbols: list[str]) -> list[dict]:
    """
    Download 60d of daily OHLCV for a batch of symbols.
    Returns list of dicts for symbols that pass the gate.
    """
    end   = datetime.today()
    start = end - timedelta(days=90)   # extra buffer

    try:
        raw = yf.download(
            symbols,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        log.warning(f"yfinance batch error: {e}")
        return []

    # yfinance returns MultiIndex columns when >1 ticker
    if len(symbols) == 1:
        close  = raw["Close"].dropna()
        volume = raw["Volume"].dropna()
        if len(close) < MIN_HISTORY_DAYS:
            return []
        price    = float(close.iloc[-1])
        avg_vol  = float(volume.tail(20).mean())
        if price >= MIN_PRICE and avg_vol >= MIN_AVG_VOL:
            return [{"symbol": symbols[0], "last_price": round(price, 2),
                     "avg_vol_20d": int(avg_vol), "history_days": len(close)}]
        return []

    passed = []
    close_df  = raw["Close"]
    volume_df = raw["Volume"]

    for sym in symbols:
        try:
            close  = close_df[sym].dropna()
            volume = volume_df[sym].dropna()

            if len(close) < MIN_HISTORY_DAYS:
                continue

            price   = float(close.iloc[-1])
            avg_vol = float(volume.tail(20).mean())

            if price >= MIN_PRICE and avg_vol >= MIN_AVG_VOL:
                passed.append({
                    "symbol":       sym,
                    "last_price":   round(price, 2),
                    "avg_vol_20d":  int(avg_vol),
                    "history_days": len(close),
                })
        except Exception:
            continue

    return passed


def apply_liquidity_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Run all symbols through the gate in batches. Returns filtered DataFrame."""
    symbols  = df["symbol"].tolist()
    n        = len(symbols)
    passed   = []
    batches  = [symbols[i:i+BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    total_b  = len(batches)

    log.info(f"Running liquidity gate on {n:,} symbols in {total_b} batches...")

    for i, batch in enumerate(batches, 1):
        log.info(f"  Batch {i}/{total_b}  ({len(batch)} symbols)")
        results = gate_batch(batch)
        passed.extend(results)
        if i < total_b:
            time.sleep(BATCH_SLEEP)

    gate_df = pd.DataFrame(passed)
    if gate_df.empty:
        log.error("No symbols passed the liquidity gate — check network/API")
        return pd.DataFrame()

    # Merge back with metadata
    out = df.merge(gate_df, on="symbol", how="inner")
    log.info(f"Liquidity gate: {len(out):,} / {n:,} symbols passed")
    return out


# ── MARKET CAP FILTER ─────────────────────────────────────────────────────────

def fetch_market_caps(symbols: list[str]) -> dict[str, float]:
    """
    Fetch market cap for each symbol via yfinance .fast_info.
    Returns {symbol: market_cap}. Missing = 0.
    """
    caps = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).fast_info
            caps[sym] = getattr(info, "market_cap", 0) or 0
        except Exception:
            caps[sym] = 0
    return caps


def apply_market_cap_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch and apply market cap minimum. Adds market_cap column."""
    log.info(f"Fetching market caps for {len(df):,} symbols (this takes a few minutes)...")
    caps = fetch_market_caps(df["symbol"].tolist())
    df["market_cap"] = df["symbol"].map(caps).fillna(0)
    filtered = df[df["market_cap"] >= MIN_MARKET_CAP].copy()
    log.info(f"Market cap filter: {len(filtered):,} / {len(df):,} passed (>= ${MIN_MARKET_CAP/1e6:.0f}M)")
    return filtered


# ── SECTOR ENRICHMENT ─────────────────────────────────────────────────────────

def enrich_sectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort sector/industry fill via yfinance .fast_info / .info.
    Only run on the final filtered set (smaller).
    """
    log.info(f"Enriching sector data for {len(df):,} symbols...")
    sectors    = {}
    industries = {}

    for sym in df["symbol"]:
        try:
            info = yf.Ticker(sym).info
            sectors[sym]    = info.get("sector",   "")
            industries[sym] = info.get("industry", "")
        except Exception:
            sectors[sym]    = ""
            industries[sym] = ""

    df["sector"]   = df["symbol"].map(sectors)
    df["industry"] = df["symbol"].map(industries)
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    log.info("═" * 60)
    log.info("STAGE 1 — Universe Builder")
    log.info("═" * 60)

    # 1. Fetch raw ticker lists
    log.info("Fetching NASDAQ ticker list...")
    nasdaq = fetch_nasdaq_tickers()
    log.info(f"  NASDAQ: {len(nasdaq):,} symbols")

    log.info("Fetching NYSE/other ticker list...")
    nyse = fetch_nyse_tickers()
    log.info(f"  NYSE/other: {len(nyse):,} symbols")

    # 2. Combine and clean
    raw_df = combine_and_clean(nasdaq, nyse)
    raw_df.to_csv(OUTPUT_RAW, index=False)
    log.info(f"Raw universe saved → {OUTPUT_RAW}  ({len(raw_df):,} symbols)")

    # 3. Liquidity gate (price + volume + history)
    liquid_df = apply_liquidity_gate(raw_df)
    if liquid_df.empty:
        log.error("Pipeline halted — empty universe after liquidity gate")
        return

    # 4. Market cap filter
    final_df = apply_market_cap_filter(liquid_df)

    # 5. Sector enrichment (best-effort, non-blocking)
    try:
        final_df = enrich_sectors(final_df)
    except Exception as e:
        log.warning(f"Sector enrichment failed (non-fatal): {e}")

    # 6. Sort and save
    final_df = final_df.sort_values("avg_vol_20d", ascending=False).reset_index(drop=True)
    final_df["as_of"] = datetime.today().strftime("%Y-%m-%d")

    cols = ["symbol", "name", "exchange", "sector", "industry",
            "last_price", "avg_vol_20d", "market_cap", "history_days", "as_of"]
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    final_df.to_csv(OUTPUT, index=False)

    elapsed = time.time() - start_time
    log.info("═" * 60)
    log.info(f"Universe built in {elapsed/60:.1f} min")
    log.info(f"Final universe: {len(final_df):,} tradeable symbols")
    log.info(f"Output → {OUTPUT}")
    log.info("═" * 60)

    # Print summary
    if "sector" in final_df.columns:
        sector_counts = final_df["sector"].value_counts().head(10)
        log.info("Top sectors:")
        for sector, count in sector_counts.items():
            log.info(f"  {sector:<35} {count:>5}")

    return final_df


if __name__ == "__main__":
    run()
