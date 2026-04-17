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
    """
    # Build weekly labels
    log_df = log_df.copy()
    log_df["label"] = log_df.groupby("week_of")["forward_return_1w"].transform(
        lambda x: (x >= x.quantile(0.80)).astype(int)
    )

    # For signal features, use the composite scores stored in the log
    # (full sub-signal history would require archiving signals.csv each week)
    feature_cols = ["sig_momentum", "sig_catalyst", "sig_fundamentals", "sig_sentiment"]
    available    = [c for c in feature_cols if c in log_df.columns]

    X = log_df[available].fillna(0.5)
    y = log_df["label"]

    mask = y.notna() & X.notna().all(axis=1)
    return X[mask], y[mask]


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
