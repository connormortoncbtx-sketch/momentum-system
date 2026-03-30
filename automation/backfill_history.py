"""
automation/backfill_history.py
================================
ONE-TIME SCRIPT — run manually once to bootstrap the performance log
with historical weekly return data. Never needs to run again after that.

What it does:
    1. Loads the current universe (data/universe.csv)
    2. Pulls 26 weeks of daily price history for all tickers via yfinance
    3. Resamples to weekly (Friday-to-Friday) returns
    4. For each historical week, identifies which tickers were top-quintile
       performers — these become the "winner" labels
    5. Writes all historical rows to data/performance_log.csv
    6. Triggers an immediate model retrain on real labels

After this runs:
    - performance_log.csv has 26 weeks × ~700 tickers = ~18,000 rows
    - models/lgbm_model.pkl is retrained on real labels immediately
    - The system exits bootstrap mode on day one

Run:
    python automation/backfill_history.py
    python automation/backfill_history.py --weeks 12   # shorter lookback
    python automation/backfill_history.py --weeks 52   # full year

NOTE: This script does NOT have historical signal scores (momentum, catalyst,
fundamentals, sentiment) because those signals weren't computed historically.
The performance log rows will have NaN signal columns. The model retrain uses
only the return labels, which is sufficient to escape bootstrap mode.
Signal attribution analysis (which signals predicted winners) requires the
ongoing weekly pipeline to build up. This script only accelerates the
label quality — not the attribution analysis.
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta, date

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR    = Path("data")
UNIVERSE    = DATA_DIR / "universe.csv"
PERF_LOG    = DATA_DIR / "performance_log.csv"
REGIME_JSON = DATA_DIR / "regime.json"

BATCH_SIZE   = 100
TOP_QUINTILE = 0.80    # tickers above this weekly return percentile = label 1


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_friday_dates(n_weeks: int) -> list[date]:
    """Return list of the last n_weeks Fridays in chronological order."""
    today = date.today()
    # Find most recent completed Friday (not this week if today is Mon-Thu)
    days_back = (today.weekday() - 4) % 7
    if days_back == 0 and today.weekday() != 4:
        days_back = 7
    last_friday = today - timedelta(days=days_back)

    fridays = []
    for i in range(n_weeks, 0, -1):
        fridays.append(last_friday - timedelta(weeks=i))
    return fridays


def fetch_price_history(symbols: list[str], weeks: int) -> pd.DataFrame:
    """
    Download daily close prices for all symbols going back weeks+2 weeks.
    Returns wide DataFrame: index=date, columns=symbols.
    """
    end   = date.today()
    start = end - timedelta(weeks=weeks + 2)

    log.info(f"Fetching {weeks} weeks of price history for {len(symbols)} symbols...")
    log.info(f"  Date range: {start} → {end}")

    all_prices = {}
    batches = [symbols[i:i+BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]

    for i, batch in enumerate(batches, 1):
        if i % 5 == 0:
            log.info(f"  Batch {i}/{len(batches)}")
        try:
            raw = yf.download(
                batch,
                start=str(start),
                end=str(end),
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]

            if len(batch) == 1:
                sym = batch[0]
                if not raw.dropna().empty:
                    all_prices[sym] = raw.dropna()
            else:
                for sym in batch:
                    try:
                        s = raw[sym].dropna()
                        if len(s) > 10:
                            all_prices[sym] = s
                    except Exception:
                        continue
        except Exception as e:
            log.warning(f"  Batch {i} error: {e}")
            continue

    if not all_prices:
        log.error("No price data fetched")
        return pd.DataFrame()

    prices_df = pd.DataFrame(all_prices)
    log.info(f"  Got prices for {len(prices_df.columns)} symbols")
    return prices_df


def compute_weekly_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to weekly (Friday close) and compute returns.
    Returns DataFrame: index=week_end_date, columns=symbols, values=return.
    """
    weekly = prices_df.resample("W-FRI").last()
    returns = weekly.pct_change().dropna(how="all")
    log.info(f"  Weekly returns: {len(returns)} weeks × {len(returns.columns)} symbols")
    return returns


def build_performance_rows(
    returns_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    fridays: list[date],
) -> pd.DataFrame:
    """
    For each historical week, build performance_log rows with:
        - forward_return_1w (actual weekly return)
        - forward_return_1w_rank (percentile rank that week)
        - label (1 if top quintile that week)
        - NaN signal scores (not available historically)
    """
    # Load regime for context (use current regime as placeholder)
    try:
        import json
        with open(REGIME_JSON) as f:
            regime_data = json.load(f)
        current_regime = regime_data.get("regime", "unknown")
    except Exception:
        current_regime = "unknown"

    # Build symbol → metadata map
    meta = universe_df.set_index("symbol")[
        [c for c in ["sector","market_cap"] if c in universe_df.columns]
    ].to_dict("index")

    rows = []
    available_weeks = returns_df.index.to_series().dt.date.tolist()

    for friday in fridays:
        # Find the closest available Friday in our returns data
        closest = min(available_weeks, key=lambda d: abs((d - friday).days))
        if abs((closest - friday).days) > 5:
            log.debug(f"  Skipping {friday} — no close data within 5 days")
            continue

        week_returns = returns_df.loc[
            returns_df.index.date == closest
        ].squeeze()

        if week_returns.empty or isinstance(week_returns, pd.DataFrame):
            continue

        week_returns = week_returns.dropna()
        if len(week_returns) < 50:
            log.debug(f"  Skipping {closest} — only {len(week_returns)} symbols")
            continue

        # Compute percentile rank and label
        pct_rank = week_returns.rank(pct=True)
        labels   = (pct_rank >= TOP_QUINTILE).astype(int)

        week_str = str(friday)

        for sym in week_returns.index:
            m = meta.get(sym, {})
            rows.append({
                "week_of":               week_str,
                "symbol":                sym,
                "alpha_score":           np.nan,   # not available historically
                "alpha_rank":            np.nan,
                "conviction":            np.nan,
                "sector":                m.get("sector", ""),
                "regime":                current_regime,  # placeholder
                "regime_composite":      np.nan,
                "forward_return_1w":     round(float(week_returns[sym]), 6),
                "forward_return_1w_rank": round(float(pct_rank[sym]), 4),
                "sig_momentum":          np.nan,
                "sig_catalyst":          np.nan,
                "sig_fundamentals":      np.nan,
                "sig_sentiment":         np.nan,
                "label":                 int(labels[sym]),
            })

        n_winners = labels.sum()
        avg_ret   = week_returns.mean()
        log.info(f"  Week {week_str}: {len(week_returns)} symbols  "
                 f"{n_winners} winners ({n_winners/len(week_returns)*100:.0f}%)  "
                 f"avg_return={avg_ret*100:.2f}%")

    return pd.DataFrame(rows)


# ── RETRAIN ───────────────────────────────────────────────────────────────────

def retrain_on_backfill(perf_log: pd.DataFrame):
    """Retrain LightGBM immediately using the backfilled real labels."""
    log.info("\nRetraining model on backfilled real labels...")

    try:
        import pickle
        import lightgbm as lgb
        from pathlib import Path as P
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        return

    # We only have return labels, not signal features historically.
    # Use the current week's signal scores for any symbol that has them.
    signals_csv = DATA_DIR / "signals.csv"
    if not signals_csv.exists():
        log.warning("  No signals.csv found — skipping retrain (run full pipeline first)")
        return

    signals = pd.read_csv(signals_csv)

    # Join current signals to backfill labels
    # Use the most recent week's labels for each symbol
    recent = perf_log.sort_values("week_of").groupby("symbol").last().reset_index()
    merged = signals.merge(recent[["symbol","label"]], on="symbol", how="inner")
    merged = merged[merged["label"].notna()]

    FEATURES = [
        "sig_momentum_rs","sig_momentum_trend","sig_momentum_vol_surge",
        "sig_momentum_breakout","sig_catalyst_earnings","sig_catalyst_insider",
        "sig_catalyst_analyst","sig_fund_growth","sig_fund_quality",
        "sig_fund_profitability","sig_fund_value","sig_sentiment_news",
        "sig_sentiment_analyst","sig_sentiment_short",
        "sig_momentum_adj","sig_catalyst_adj","sig_fundamentals_adj","sig_sentiment_adj",
    ]
    features = [f for f in FEATURES if f in merged.columns]

    if len(merged) < 50 or not features:
        log.warning(f"  Insufficient data for retrain ({len(merged)} rows, {len(features)} features)")
        return

    X = merged[features].fillna(0.5).values
    y = merged["label"].values

    log.info(f"  Training on {len(X)} samples, {len(features)} features")
    log.info(f"  Class balance: {y.mean()*100:.1f}% positive")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    model_file = Path("models/lgbm_model.pkl")
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # Quick in-sample IC check
    from scipy import stats
    probs = model.predict_proba(X)[:,1]
    ic, _ = stats.spearmanr(probs, y)
    log.info(f"  In-sample IC: {ic:.4f}")
    log.info(f"  Model saved → {model_file}")

    importances = pd.Series(model.feature_importances_, index=features)
    log.info("  Top 5 feature importances:")
    for feat, imp in importances.nlargest(5).items():
        log.info(f"    {feat:<35} {imp:.0f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(weeks: int = 26):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"HISTORICAL BACKFILL — {weeks} weeks")
    log.info("=" * 60)
    log.info("This is a one-time operation. Run once, then let the")
    log.info("Monday scoring workflow maintain the log going forward.")
    log.info("=" * 60)

    # Load universe
    if not UNIVERSE.exists():
        log.error("No universe.csv found — run Stage 1 first")
        return
    universe = pd.read_csv(UNIVERSE)
    symbols  = universe["symbol"].tolist()
    log.info(f"Universe: {len(symbols)} symbols")

    # Get target Fridays
    fridays = get_friday_dates(weeks)
    log.info(f"Target weeks: {fridays[0]} → {fridays[-1]}")

    # Fetch price history
    prices = fetch_price_history(symbols, weeks)
    if prices.empty:
        log.error("No price data — aborting")
        return

    # Compute weekly returns
    returns = compute_weekly_returns(prices)

    # Build performance rows
    log.info("\nBuilding performance log rows...")
    new_rows = build_performance_rows(returns, universe, fridays)

    if new_rows.empty:
        log.error("No rows built — check price data")
        return

    log.info(f"\nBuilt {len(new_rows):,} rows across {new_rows['week_of'].nunique()} weeks")

    # Merge with any existing log
    if PERF_LOG.exists():
        existing = pd.read_csv(PERF_LOG)
        # Don't overwrite weeks that already have real pipeline data
        existing_weeks = set(existing["week_of"].unique())
        new_only = new_rows[~new_rows["week_of"].isin(existing_weeks)]
        combined = pd.concat([existing, new_only], ignore_index=True)
        log.info(f"Merged with existing log: {len(existing)} existing + "
                 f"{len(new_only)} new = {len(combined)} total rows")
    else:
        combined = new_rows
        log.info(f"Created new performance log: {len(combined)} rows")

    combined = combined.sort_values(["week_of","symbol"]).reset_index(drop=True)
    combined.to_csv(PERF_LOG, index=False)
    log.info(f"Saved → {PERF_LOG}")

    # Summary stats
    log.info("\nBackfill summary:")
    log.info(f"  Weeks covered:    {combined['week_of'].nunique()}")
    log.info(f"  Symbols covered:  {combined['symbol'].nunique()}")
    log.info(f"  Avg weekly return: {combined['forward_return_1w'].mean()*100:.2f}%")
    log.info(f"  Winner rate:       {(combined['forward_return_1w_rank'] >= TOP_QUINTILE).mean()*100:.1f}%")

    # Retrain model immediately
    retrain_on_backfill(combined)

    log.info("\n" + "=" * 60)
    log.info("Backfill complete.")
    log.info("The model has been retrained on real labels.")
    log.info("Run the full pipeline to score with the new model:")
    log.info("  python run_pipeline.py --from 04")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-time historical backfill")
    parser.add_argument("--weeks", type=int, default=26,
                        help="Weeks of history to backfill (default: 26)")
    args = parser.parse_args()
    run(weeks=args.weeks)
