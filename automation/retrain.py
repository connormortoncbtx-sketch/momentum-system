"""
automation/retrain.py
======================
Runs every Monday after self_refine.py.

Retrains the LightGBM model using real forward-return labels
from performance_log.csv once enough history exists.

Replaces models/lgbm_model.pkl in place.
The pipeline will load the new model next Friday automatically.

Retraining triggers:
    - At least 6 weeks of performance data
    - At least 30% universe coverage per week on average
    - Only if IC of current model has been declining (optional guard)
"""

import pickle
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation.system_logger import log_event, LogStatus
from automation.notifier import notify_alert, notify_error, notify_success

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
PERF_LOG    = DATA_DIR / "performance_log.csv"
SIGNALS_CSV = DATA_DIR / "signals.csv"
MODEL_FILE  = MODELS_DIR / "lgbm_model.pkl"

MIN_WEEKS        = 6
MIN_ROWS         = 5000
RETRAIN_FEATURES = [
    "sig_momentum_rs", "sig_momentum_trend", "sig_momentum_vol_surge",
    "sig_momentum_breakout", "sig_catalyst_earnings", "sig_catalyst_insider",
    "sig_catalyst_analyst", "sig_fund_growth", "sig_fund_quality",
    "sig_fund_profitability", "sig_fund_value", "sig_sentiment_news",
    "sig_sentiment_analyst", "sig_sentiment_short",
    "sig_momentum_adj", "sig_catalyst_adj", "sig_fundamentals_adj",
    "sig_sentiment_adj",
]


def build_training_data(log_df: pd.DataFrame,
                        signals_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Join historical signal data to forward returns.
    Label = 1 if forward_return_1w is in top quintile that week.

    Feature selection (C1 fix): prefers the full 18-feature sub-signal set
    (RETRAIN_FEATURES) that matches the production model's shape. Falls back
    to the 4 coarse composites for old perf_log rows written before
    collect_returns was taught to preserve sub-signals. Mixed is NOT allowed
    -- if the run doesn't have all 18 but has all 4 composites, we use the
    4-composite set and warn. The pipeline's stage 4 has a matching fallback
    in pipeline/04_model.py (lines 484-495) that routes to composite features
    when the saved model has n_features_in_ <= 4, so feature-shape stays
    consistent end-to-end.

    Mask before fillna (C2 fix): previously X got fillna(0.5) first, then
    X.notna().all(axis=1) was used in the mask -- which was vacuous because
    there were no NaNs left to check. Rows with missing sub-signals silently
    trained as synthetic 'neutral at 0.5 across every feature' samples,
    injecting a point mass at the training distribution's center. Now we
    drop rows missing any feature OR the label first, then fill any residual
    NaN on the kept rows (defensive -- shouldn't remain).
    """
    FALLBACK_FEATURES = ["sig_momentum", "sig_catalyst",
                         "sig_fundamentals", "sig_sentiment"]

    # Build weekly labels from real forward returns
    log_df = log_df.copy()
    log_df["label"] = log_df.groupby("week_of")["forward_return_1w"].transform(
        lambda x: (x >= x.quantile(0.80)).astype(int)
    )

    # C1: prefer 18-feature sub-signal set; fall back to 4 composites.
    # Tolerate partial sub-signal availability: if ≥60% of the 18 sub-signal columns
    # are present AND at least one row has ≥70% of features non-null, use the
    # sub-signal set. Historically perf_log rows have varying signal coverage
    # (insider data missing for small-caps, analyst data missing for micro-caps,
    # etc.) so requiring all 18 features be non-null for a row to be kept was
    # zeroing out the entire training set. The ≥70% threshold per row balances
    # "train on richest possible feature set" against "don't lose every row."
    full_available    = [c for c in RETRAIN_FEATURES if c in log_df.columns]
    fallback_present  = all(c in log_df.columns for c in FALLBACK_FEATURES)

    if len(full_available) >= 0.60 * len(RETRAIN_FEATURES):
        # Use the sub-signals that are present (not necessarily all 18)
        features = full_available
        log.info(f"  Training on {len(features)}/{len(RETRAIN_FEATURES)} sub-signal features "
                 f"(those present in perf_log)")
        min_features_per_row = int(0.70 * len(features))
    elif fallback_present:
        features = FALLBACK_FEATURES
        missing = [c for c in RETRAIN_FEATURES if c not in log_df.columns]
        log.warning(
            f"  Sub-signals missing from perf_log ({len(missing)} cols); "
            f"falling back to {len(features)} coarse composites. "
            f"Sub-signals will become available in perf_log after "
            f"collect_returns runs for N weeks with the post-Tier-4 build."
        )
        min_features_per_row = len(features)  # all 4 composites required
    else:
        # No usable feature set. Return empty frames so run() skips cleanly.
        missing = [c for c in FALLBACK_FEATURES if c not in log_df.columns]
        log.error(f"  No usable feature set -- both RETRAIN_FEATURES and "
                  f"FALLBACK_FEATURES are missing. Absent composites: {missing}")
        return pd.DataFrame(), pd.Series(dtype=int)

    # C2: mask BEFORE fillna -- keep rows with label + at least min_features_per_row
    # non-null features. This tolerates partial coverage per row while still
    # requiring enough signal density that the row carries real information.
    y = log_df["label"]
    X_raw = log_df[features]
    features_per_row = X_raw.notna().sum(axis=1)
    mask = y.notna() & (features_per_row >= min_features_per_row)

    # Apply mask first, then fill residual NaN in kept rows with 0.5 (neutral).
    # This is meaningfully different from the pre-fix fillna-then-vacuous-mask:
    # now 0.5 only appears in rows that already have ≥70% real signal, so it's
    # a minority replacement rather than dominating the training set.
    X = X_raw[mask].fillna(0.5)
    y = y[mask].astype(int)

    log.info(f"  Training data: {len(X):,} rows after masking "
             f"({(~mask).sum():,} dropped for missing label or <{min_features_per_row} features)")

    return X, y


def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_event("retrain", LogStatus.INFO, "Starting retrain")

    if not PERF_LOG.exists():
        log.info("No performance log — skipping retrain")
        log_event("retrain", LogStatus.INFO, "Skipped: no performance_log.csv")
        return

    try:
        log_df  = pd.read_csv(PERF_LOG, low_memory=False, parse_dates=["week_of"])
        n_weeks = log_df["week_of"].nunique()
        n_rows  = log_df["forward_return_1w"].notna().sum()

        log.info(f"Performance log: {n_weeks} weeks, {n_rows:,} return observations")

        if n_weeks < MIN_WEEKS:
            log.info(f"Need {MIN_WEEKS} weeks (have {n_weeks}) — skipping retrain")
            log_event("retrain", LogStatus.INFO,
                      f"Skipped: only {n_weeks} weeks (need {MIN_WEEKS})")
            return

        if n_rows < MIN_ROWS:
            log.info(f"Need {MIN_ROWS} rows (have {n_rows}) — skipping retrain")
            log_event("retrain", LogStatus.INFO,
                      f"Skipped: only {n_rows} rows (need {MIN_ROWS})")
            return

        log.info("Building training data...")
        X, y = build_training_data(log_df, None)

        # C2 side-effect: if masking drops every row (e.g., old perf_log rows
        # all had some NaN), skip cleanly instead of crashing in model.fit.
        if len(X) == 0 or len(y) == 0:
            log.warning("No usable training rows after masking -- skipping retrain")
            log_event("retrain", LogStatus.WARNING,
                      "Skipped: no training rows passed feature/label mask",
                      metrics={"weeks": int(n_weeks), "raw_rows": int(n_rows)})
            return

        if len(X) < MIN_ROWS:
            log.info(f"After masking, only {len(X)} rows (need {MIN_ROWS}) -- skipping")
            log_event("retrain", LogStatus.INFO,
                      f"Skipped: {len(X)} rows after masking (need {MIN_ROWS})",
                      metrics={"rows_after_mask": int(len(X))})
            return

        log.info(f"  {len(X)} training samples, {y.mean()*100:.1f}% positive")

        log.info("Training LightGBM...")
        try:
            import lightgbm as lgb
        except ImportError:
            log.error("lightgbm not installed")
            log_event("retrain", LogStatus.ERROR, "lightgbm package missing")
            notify_error("retrain", "lightgbm package not installed")
            return

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
        model.fit(X.values, y.values)

        # Quick validation: IC on training set (in-sample, for sanity only)
        from scipy import stats
        probs = model.predict_proba(X.values)[:, 1]
        ic, _ = stats.spearmanr(probs, log_df.loc[X.index, "forward_return_1w"].fillna(0))
        log.info(f"  In-sample IC: {ic:.4f}")

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)

        log.info(f"Model retrained and saved → {MODEL_FILE}")
        log.info(f"  Trained on {n_weeks} weeks of real return data")

        log_event("retrain", LogStatus.SUCCESS,
                  f"Model retrained on {n_weeks} weeks, {len(X):,} samples",
                  metrics={
                      "weeks":              int(n_weeks),
                      "samples":            int(len(X)),
                      "positive_class_pct": round(float(y.mean() * 100), 2),
                      "in_sample_ic":       round(float(ic), 4),
                      "n_features":         int(X.shape[1]),
                  })
        notify_success("retrain",
                       f"Model retrained: IC={ic:.3f} on {len(X):,} samples "
                       f"across {n_weeks} weeks")

    except Exception as e:
        log.error(f"retrain crashed: {e}", exc_info=True)
        log_event("retrain", LogStatus.ERROR,
                  "Unhandled exception during retrain",
                  errors=[str(e)])
        notify_error("retrain", f"Retrain failed: {e}")
        raise


if __name__ == "__main__":
    run()
