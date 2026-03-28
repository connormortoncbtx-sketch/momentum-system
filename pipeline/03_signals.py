"""
Stage 3 — Signal Engine
========================
Orchestrates all four signal modules using a tiered approach
to keep runtime manageable regardless of universe size.

Tier 1 (full universe): Momentum signal only — fast batch downloads,
    no per-ticker API calls. Used to rank all tickers by RS.

Tier 2 (top N by momentum): Catalyst, fundamentals, sentiment — slow
    per-ticker API calls. Only run on top candidates to save time.
    Bottom tickers get median-filled signal values.

Tier cutoff is configurable via SLOW_SIGNAL_TIER below.

Reads:   data/universe.csv, data/regime.json, config/weights.json
Writes:  data/signals.csv
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

# Number of tickers to run slow signals on.
# Everything outside this tier gets median-filled values.
# At ~5s/ticker for catalyst, 600 tickers = ~50 min — fits in 3hr window.
SLOW_SIGNAL_TIER = 600


# ── LOAD CONFIG ───────────────────────────────────────────────────────────────

def load_regime() -> dict:
    with open(REGIME_JSON) as f:
        return json.load(f)


def load_weights() -> dict:
    with open(WEIGHTS_JSON) as f:
        return json.load(f)


# ── REGIME ADJUSTMENT ─────────────────────────────────────────────────────────

def apply_regime_weights(df: pd.DataFrame, regime: str, weights: dict) -> pd.DataFrame:
    base_weights  = weights["signal_weights"]
    regime_mults  = weights["regime_multipliers"].get(regime, {})

    signal_map = {
        "momentum":     "sig_momentum",
        "catalyst":     "sig_catalyst",
        "fundamentals": "sig_fundamentals",
        "sentiment":    "sig_sentiment",
    }

    for name, col in signal_map.items():
        if col not in df.columns:
            continue
        base_w  = base_weights.get(name, 0.25)
        reg_m   = regime_mults.get(name, 1.0)
        df[f"{col}_adj"] = df[col] * base_w * reg_m

    if "sig_momentum_trend" in df.columns:
        tech_w = base_weights.get("technicals", 0.15)
        reg_m  = regime_mults.get("technicals", 1.0)
        df["sig_technicals_adj"] = df["sig_momentum_trend"] * tech_w * reg_m

    return df


# ── FILL MISSING VALUES ───────────────────────────────────────────────────────

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    signal_cols = [c for c in df.columns if c.startswith("sig_")]
    for col in signal_cols:
        if df[col].isna().any():
            if "sector" in df.columns:
                sector_med = df.groupby("sector")[col].transform("median")
                df[col] = df[col].fillna(sector_med)
            universe_med = df[col].median()
            df[col] = df[col].fillna(
                universe_med if not np.isnan(universe_med) else 0.5)
    return df


# ── SUMMARY STATS ─────────────────────────────────────────────────────────────

def log_signal_stats(df: pd.DataFrame):
    for col in ["sig_momentum","sig_catalyst","sig_fundamentals","sig_sentiment"]:
        if col in df.columns:
            s = df[col].dropna()
            log.info(f"  {col:<25} mean={s.mean():.3f}  std={s.std():.3f}  "
                     f"p25={s.quantile(0.25):.3f}  p75={s.quantile(0.75):.3f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading universe...")
    universe = pd.read_csv(UNIVERSE_CSV)
    n_total  = len(universe)
    log.info(f"  {n_total:,} symbols")

    log.info("Loading regime...")
    regime_data = load_regime()
    regime      = regime_data["regime"]
    log.info(f"  Regime: {regime}")

    log.info("Loading weights...")
    weights = load_weights()

    # ── TIER 1: MOMENTUM (full universe) ─────────────────────────────────────
    log.info(f"Running momentum signal on full universe ({n_total:,} symbols)...")
    from pipeline.signals.momentum import score as momentum_score, fetch_history
    symbols = universe["symbol"].tolist()
    history = fetch_history(symbols)
    log.info(f"  Got history for {len(history):,} symbols")
    df = momentum_score(universe, history)
    log.info("  Momentum signal complete")

    # ── DETERMINE TIER 2 CUTOFF ───────────────────────────────────────────────
    tier_size = min(SLOW_SIGNAL_TIER, n_total)
    log.info(f"Tiered scoring: slow signals on top {tier_size:,} by momentum "
             f"({n_total - tier_size:,} will receive median-filled values)")

    # Rank by momentum score to pick tier 2 candidates
    if "sig_momentum" in df.columns:
        df["_mom_rank"] = df["sig_momentum"].rank(ascending=False, method="min")
        tier2_mask = df["_mom_rank"] <= tier_size
    else:
        tier2_mask = pd.Series([True] * len(df), index=df.index)

    tier2_df   = df[tier2_mask].copy()
    tier3_df   = df[~tier2_mask].copy()

    log.info(f"  Tier 2 (slow signals): {len(tier2_df):,} symbols")
    log.info(f"  Tier 3 (median fill):  {len(tier3_df):,} symbols")

    # ── TIER 2: CATALYST ─────────────────────────────────────────────────────
    log.info("Running catalyst signal on tier 2...")
    from pipeline.signals.catalyst import score as catalyst_score
    tier2_df = catalyst_score(tier2_df)
    log.info("  Catalyst signal complete")

    # ── TIER 2: FUNDAMENTALS ─────────────────────────────────────────────────
    log.info("Running fundamentals signal on tier 2...")
    from pipeline.signals.fundamentals import score as fund_score
    tier2_df = fund_score(tier2_df)
    log.info("  Fundamentals signal complete")

    # ── TIER 2: SENTIMENT ─────────────────────────────────────────────────────
    log.info("Running sentiment signal on tier 2...")
    from pipeline.signals.sentiment import score as sentiment_score
    tier2_df = sentiment_score(tier2_df)
    log.info("  Sentiment signal complete")

    # ── RECOMBINE ─────────────────────────────────────────────────────────────
    df = pd.concat([tier2_df, tier3_df], ignore_index=True)

    # Clean up helper column
    if "_mom_rank" in df.columns:
        df = df.drop(columns=["_mom_rank"])

    # ── POST-PROCESSING ───────────────────────────────────────────────────────
    log.info("Filling missing values (tier 3 median fill)...")
    df = fill_missing(df)

    log.info(f"Applying regime adjustments (regime={regime})...")
    df = apply_regime_weights(df, regime, weights)

    df["signals_as_of"] = datetime.today().strftime("%Y-%m-%d")
    df.to_csv(OUTPUT, index=False)

    log.info("Signal stats:")
    log_signal_stats(df)
    log.info(f"Output → {OUTPUT}  ({len(df):,} rows, {len(df.columns)} cols)")
    log.info(f"Tier breakdown: {len(tier2_df):,} fully scored  |  "
             f"{len(tier3_df):,} momentum-only (median filled)")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
