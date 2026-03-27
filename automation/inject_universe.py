"""
automation/inject_universe.py
==============================
ONE-TIME (or occasional) script to inject an external ticker list
directly into data/universe.csv without running the full liquidity gate.

Designed for ticker lists sourced from NASDAQ FTP, other exchange feeds,
or curated lists where you trust the source quality.

Usage:
    python automation/inject_universe.py --file path/to/nasdaqlisted.csv
    python automation/inject_universe.py --file path/to/tickers.csv --symbol-col Ticker
    python automation/inject_universe.py --file path/to/list.xlsx

Supports: .csv, .txt, .xlsx, .xls

The script:
    1. Reads the file and identifies the symbol column
    2. Applies lightweight quality filters (no ETFs, no deficient/bankrupt,
       no warrants/units/rights by symbol suffix)
    3. Deduplicates against existing universe.csv
    4. Appends new symbols with placeholder price/vol/mcap data
       (these get refreshed on the next weekly upsert run)
    5. Reports what was added

After running:
    - New symbols appear in the ranked report next pipeline run
    - Price/vol/mcap data fills in automatically via Stage 1 upsert
    - No need to run the full universe rebuild
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR   = Path("data")
UNIVERSE   = DATA_DIR / "universe.csv"

# Columns we try to detect automatically
SYMBOL_COL_CANDIDATES = [
    "Symbol", "symbol", "Ticker", "ticker", "SYMBOL", "TICKER",
    "Stock", "stock", "Code", "code",
]
NAME_COL_CANDIDATES = [
    "Security Name", "Name", "name", "Company Name", "Company",
    "Description", "security_name", "company_name",
]
ETF_COL_CANDIDATES   = ["ETF", "etf", "Is ETF", "isETF"]
STATUS_COL_CANDIDATES = ["Financial Status", "financial_status", "Status", "status"]


# ── FILE READER ───────────────────────────────────────────────────────────────

def read_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext in (".csv", ".txt", ".tsv"):
        # Try comma first, then tab, then pipe
        for sep in [",", "\t", "|"]:
            try:
                df = pd.read_csv(path, sep=sep)
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv, .txt, .xlsx, .xls")

    log.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {p.name}")
    log.info(f"Columns: {df.columns.tolist()}")
    return df


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── SYMBOL EXTRACTION ─────────────────────────────────────────────────────────

def extract_symbols(df: pd.DataFrame, symbol_col: str | None = None,
                    name_col: str | None = None) -> pd.DataFrame:
    """
    Extract and clean symbols from the DataFrame.
    Returns clean DataFrame with columns: symbol, name, exchange
    """
    # Auto-detect symbol column if not specified
    if symbol_col is None:
        symbol_col = detect_column(df, SYMBOL_COL_CANDIDATES)
        if symbol_col is None:
            # Last resort: use first column
            symbol_col = df.columns[0]
            log.warning(f"Could not detect symbol column — using first column: '{symbol_col}'")
        else:
            log.info(f"Detected symbol column: '{symbol_col}'")

    if symbol_col not in df.columns:
        raise ValueError(f"Symbol column '{symbol_col}' not found. Available: {df.columns.tolist()}")

    # Auto-detect name column
    if name_col is None:
        name_col = detect_column(df, NAME_COL_CANDIDATES)

    # Build output
    out = pd.DataFrame()
    out["symbol"] = df[symbol_col].astype(str).str.strip().str.upper()

    if name_col and name_col in df.columns:
        out["name"] = df[name_col].astype(str).str.strip()
    else:
        out["name"] = ""

    out["exchange"] = "NASDAQ"   # default — adjust if file has exchange column

    return out


# ── QUALITY FILTERS ───────────────────────────────────────────────────────────

def apply_quality_filters(df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight quality filters appropriate for a trusted source.
    Removes ETFs, deficient/bankrupt companies, warrants/units/rights.
    """
    before = len(df)

    # Remove ETFs if flag column exists
    etf_col = detect_column(source_df, ETF_COL_CANDIDATES)
    if etf_col:
        etf_mask = source_df[etf_col].astype(str).str.upper() == "Y"
        df = df[~etf_mask.values[:len(df)]].copy()
        log.info(f"  Removed {before - len(df)} ETFs")
        before = len(df)

    # Remove financially distressed companies if status column exists
    status_col = detect_column(source_df, STATUS_COL_CANDIDATES)
    if status_col:
        # N = normal, D = deficient, E = delinquent, H = bankrupt, Q = question
        bad_status = source_df[status_col].astype(str).str.upper().isin(["D","E","H","Q"])
        df = df[~bad_status.values[:len(df)]].copy()
        removed = before - len(df)
        if removed:
            log.info(f"  Removed {removed} financially distressed/delinquent symbols")
        before = len(df)

    # Symbol-based filters — remove warrants, units, rights, preferreds
    bad_suffix = (
        df["symbol"].str.endswith("W")  |   # warrants
        df["symbol"].str.endswith("U")  |   # units
        df["symbol"].str.endswith("R")  |   # rights
        df["symbol"].str.contains(r"\.", regex=True) |  # BRK.B style
        df["symbol"].str.contains("-", regex=False)  |  # preferred shares
        df["symbol"].str.startswith("ZZ")
    )
    df = df[~bad_suffix].copy()
    removed = before - len(df)
    if removed:
        log.info(f"  Removed {removed} warrants/units/rights/preferreds by symbol pattern")
        before = len(df)

    # Remove blanks and very short symbols
    df = df[df["symbol"].str.len() >= 1].copy()
    df = df[df["symbol"] != "NAN"].copy()

    log.info(f"  After quality filters: {len(df):,} symbols remaining")
    return df.reset_index(drop=True)


# ── UPSERT ────────────────────────────────────────────────────────────────────

def upsert_into_universe(new_symbols: pd.DataFrame) -> dict:
    """
    Add new symbols to universe.csv.
    Skips any that already exist.
    Returns stats dict.
    """
    today = str(date.today())

    if UNIVERSE.exists():
        existing = pd.read_csv(UNIVERSE)
        existing_syms = set(existing["symbol"].str.upper().tolist())
        log.info(f"Existing universe: {len(existing):,} symbols")
    else:
        existing = pd.DataFrame()
        existing_syms = set()
        log.warning("No universe.csv found — creating from scratch")

    # Find truly new symbols
    new_df = new_symbols[~new_symbols["symbol"].isin(existing_syms)].copy()
    already = len(new_symbols) - len(new_df)

    if already:
        log.info(f"  {already} symbols already in universe — skipping")

    if new_df.empty:
        log.info("  Nothing new to add")
        return {"added": 0, "skipped": already, "total": len(existing)}

    # Build full rows with placeholder data
    # Price/vol/mcap will be filled in by Stage 1 upsert on next run
    new_rows = pd.DataFrame({
        "symbol":       new_df["symbol"].values,
        "name":         new_df["name"].values,
        "exchange":     new_df["exchange"].values,
        "sector":       "",
        "industry":     "",
        "last_price":   np.nan,    # filled by next upsert
        "avg_vol_20d":  np.nan,    # filled by next upsert
        "market_cap":   np.nan,    # filled by next upsert
        "history_days": np.nan,
        "gate_failures": 0,
        "as_of":        today,
    })

    # Combine and save
    if not existing.empty:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(UNIVERSE, index=False)

    log.info(f"  Added {len(new_rows):,} new symbols")
    log.info(f"  Universe now: {len(combined):,} symbols")

    return {
        "added":   len(new_rows),
        "skipped": already,
        "total":   len(combined),
        "sample":  new_df["symbol"].head(10).tolist(),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(file_path: str, symbol_col: str = None, name_col: str = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("UNIVERSE INJECTION")
    log.info("=" * 60)

    # Read file
    source_df = read_file(file_path)

    # Extract symbols
    symbols_df = extract_symbols(source_df, symbol_col, name_col)
    log.info(f"Extracted {len(symbols_df):,} symbols before filtering")

    # Apply quality filters
    log.info("Applying quality filters...")
    clean_df = apply_quality_filters(symbols_df, source_df)

    # Upsert
    log.info("Upserting into universe...")
    stats = upsert_into_universe(clean_df)

    log.info("\n" + "=" * 60)
    log.info(f"Injection complete")
    log.info(f"  Added:   {stats['added']:,} new symbols")
    log.info(f"  Skipped: {stats['skipped']:,} already in universe")
    log.info(f"  Total universe size: {stats['total']:,}")
    if stats.get("sample"):
        log.info(f"  Sample of new symbols: {stats['sample']}")
    log.info("")
    log.info("Next steps:")
    log.info("  New symbols have placeholder price/vol/mcap data.")
    log.info("  Run the pipeline to fill them in:")
    log.info("  → python run_pipeline.py --only 01   (upsert fills the data)")
    log.info("  → python run_pipeline.py              (full pipeline + score)")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inject external ticker list into universe.csv"
    )
    parser.add_argument("--file",       required=True,  help="Path to ticker file (.csv, .xlsx, .txt)")
    parser.add_argument("--symbol-col", default=None,   help="Column name containing ticker symbols")
    parser.add_argument("--name-col",   default=None,   help="Column name containing company names")
    args = parser.parse_args()

    run(
        file_path  = args.file,
        symbol_col = args.symbol_col,
        name_col   = args.name_col,
    )
