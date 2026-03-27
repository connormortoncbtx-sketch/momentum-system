"""
Stage 3 — Signal Engine
========================
Orchestrates all four signal modules against the full universe.
Reads:   data/universe.csv, data/regime.json, config/weights.json
Writes:  data/signals.csv

Signal modules (each independent, each scores 0.0-1.0):
    momentum      — RS rank, trend, volume, breakout
    catalyst      — Earnings proximity, insider buys, analyst upgrades
    fundamentals  — Growth, quality, profitability, valuation
    sentiment     — News tone, analyst trend, short interest

After all four run, this stage applies regime multipliers from
weights.json to produce a regime-adjusted score per signal,
which Stage 4 feeds into the model.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

DATA_DIR     = Path("data")
UNIVERSE_CSV = DATA_DIR / "universe.csv"
REGIME_JSON  = DATA_DIR / "regime.json"
WEIGHTS_JSON = Path("config/weights.json")
OUTPUT       = DATA_DIR / "signals.csv"


# ── LOAD CONFIG ───────────────────────────────────────────────────────────────

def load_regime() -> dict:
    with open(REGIME_JSON) as f:
        return json.load(f)


def load_weights() -> dict:
    with open(WEIGHTS_JSON) as f:
        return json.load(f)


# ── REGIME ADJUSTMENT ─────────────────────────────────────────────────────────

def apply_regime_weights(df: pd.DataFrame, regime: str, weights: dict) -> pd.DataFrame:
    """
    Multiply each signal's composite score by its regime multiplier.
    Produces sig_{name}_adj columns alongside the raw scores.
    """
    base_weights  = weights["signal_weights"]
    regime_mults  = weights["regime_multipliers"].get(regime, {})

    signal_map = {
        "momentum":     "sig_momentum",
        "catalyst":     "sig_catalyst",
        "fundamentals": "sig_fundamentals",
        "sentiment":    "sig_sentiment",
    }

    adj_cols = {}
    for name, col in signal_map.items():
        if col not in df.columns:
            continue
        base_w  = base_weights.get(name, 0.25)
        reg_m   = regime_mults.get(name, 1.0)
        adj_col = f"{col}_adj"
        df[adj_col] = df[col] * base_w * reg_m
        adj_cols[name] = adj_col

    # Technicals shares the momentum weight bucket
    if "sig_momentum_trend" in df.columns:
        tech_w = base_weights.get("technicals", 0.15)
        reg_m  = regime_mults.get("technicals", 1.0)
        df["sig_technicals_adj"] = df["sig_momentum_trend"] * tech_w * reg_m

    return df


# ── FILL MISSING VALUES ───────────────────────────────────────────────────────

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN signal values with sector median, then universe median.
    Ensures every ticker has a score even with partial data.
    """
    signal_cols = [c for c in df.columns if c.startswith("sig_")]

    for col in signal_cols:
        if df[col].isna().any():
            if "sector" in df.columns:
                sector_med = df.groupby("sector")[col].transform("median")
                df[col] = df[col].fillna(sector_med)
            universe_med = df[col].median()
            df[col] = df[col].fillna(universe_med if not np.isnan(universe_med) else 0.5)

    return df


# ── SUMMARY STATS ─────────────────────────────────────────────────────────────

def log_signal_stats(df: pd.DataFrame):
    signal_composites = ["sig_momentum", "sig_catalyst", "sig_fundamentals", "sig_sentiment"]
    for col in signal_composites:
        if col in df.columns:
            s = df[col].dropna()
            log.info(f"  {col:<25} mean={s.mean():.3f}  std={s.std():.3f}  "
                     f"p25={s.quantile(0.25):.3f}  p75={s.quantile(0.75):.3f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading universe...")
    universe = pd.read_csv(UNIVERSE_CSV)
    log.info(f"  {len(universe):,} symbols")

    log.info("Loading regime...")
    regime_data = load_regime()
    regime      = regime_data["regime"]
    log.info(f"  Regime: {regime}")

    log.info("Loading weights...")
    weights = load_weights()

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    log.info("Running momentum signal...")
    from pipeline.signals.momentum import score as momentum_score, fetch_history
    symbols = universe["symbol"].tolist()
    log.info(f"  Fetching price history for {len(symbols):,} symbols...")
    history = fetch_history(symbols)
    log.info(f"  Got history for {len(history):,} symbols")
    df = momentum_score(universe, history)
    log.info("  Momentum signal complete")

    # ── CATALYST ──────────────────────────────────────────────────────────────
    log.info("Running catalyst signal...")
    from pipeline.signals.catalyst import score as catalyst_score
    df = catalyst_score(df)
    log.info("  Catalyst signal complete")

    # ── FUNDAMENTALS ──────────────────────────────────────────────────────────
    log.info("Running fundamentals signal...")
    from pipeline.signals.fundamentals import score as fund_score
    df = fund_score(df)
    log.info("  Fundamentals signal complete")

    # ── SENTIMENT ─────────────────────────────────────────────────────────────
    log.info("Running sentiment signal...")
    from pipeline.signals.sentiment import score as sentiment_score
    df = sentiment_score(df)
    log.info("  Sentiment signal complete")

    # ── POST-PROCESSING ───────────────────────────────────────────────────────
    log.info("Filling missing values...")
    df = fill_missing(df)

    log.info(f"Applying regime adjustments (regime={regime})...")
    df = apply_regime_weights(df, regime, weights)

    # Timestamp
    df["signals_as_of"] = datetime.today().strftime("%Y-%m-%d")

    # Save
    df.to_csv(OUTPUT, index=False)

    log.info("Signal stats:")
    log_signal_stats(df)
    log.info(f"Output → {OUTPUT}  ({len(df):,} rows, {len(df.columns)} cols)")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
