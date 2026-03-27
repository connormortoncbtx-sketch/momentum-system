"""
Stage 4 — Scoring Model
========================
Reads signals.csv, runs LightGBM to produce a final alpha probability
score for every ticker, and writes scores.csv.

Two modes:
    TRAIN mode   — if no model exists, bootstrap from signal data using
                   a heuristic label (top quintile RS + catalyst = 1)
                   then train and save lgbm_model.pkl
    SCORE mode   — load existing model, score all tickers, write output

After several weeks of live data, the Monday retraining workflow
replaces the bootstrap labels with real forward-return labels,
making the model progressively more accurate over time.

Reads:   data/signals.csv, data/regime.json, config/weights.json
Writes:  data/scores.csv, models/lgbm_model.pkl (if training)
"""

import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
SIGNALS_CSV = DATA_DIR / "signals.csv"
REGIME_JSON = DATA_DIR / "regime.json"
PERF_LOG    = DATA_DIR / "performance_log.csv"
MODEL_FILE  = MODELS_DIR / "lgbm_model.pkl"
OUTPUT      = DATA_DIR / "scores.csv"

# Feature columns fed to the model
SIGNAL_FEATURES = [
    # Momentum sub-signals
    "sig_momentum_rs",
    "sig_momentum_trend",
    "sig_momentum_vol_surge",
    "sig_momentum_breakout",
    # Catalyst sub-signals
    "sig_catalyst_earnings",
    "sig_catalyst_insider",
    "sig_catalyst_analyst",
    # Fundamentals sub-signals
    "sig_fund_growth",
    "sig_fund_quality",
    "sig_fund_profitability",
    "sig_fund_value",
    # Sentiment sub-signals
    "sig_sentiment_news",
    "sig_sentiment_analyst",
    "sig_sentiment_short",
    # Composite signals (regime-adjusted)
    "sig_momentum_adj",
    "sig_catalyst_adj",
    "sig_fundamentals_adj",
    "sig_sentiment_adj",
]

# Metadata used for grouping/filtering but not as model features
META_COLS = ["symbol", "name", "exchange", "sector", "industry",
             "last_price", "avg_vol_20d", "market_cap"]


# ── MODEL TRAINING ────────────────────────────────────────────────────────────

def build_bootstrap_labels(df: pd.DataFrame) -> pd.Series:
    """
    Week 1 bootstrap: create heuristic labels before real return data exists.

    Label = 1 (winner) if ticker is in top quintile of:
        - Momentum RS rank AND
        - Has any catalyst signal above 0.5
    OR top decile of pure momentum RS.

    This gives the model a reasonable starting prior that it will
    progressively overwrite with real forward-return labels.
    """
    rs   = df["sig_momentum_rs"].fillna(0)
    cat  = df["sig_catalyst"].fillna(0)
    mom  = df["sig_momentum"].fillna(0)

    top_quintile_rs  = rs  >= rs.quantile(0.80)
    top_decile_mom   = mom >= mom.quantile(0.90)
    has_catalyst     = cat >= 0.50

    labels = ((top_quintile_rs & has_catalyst) | top_decile_mom).astype(int)
    log.info(f"  Bootstrap labels: {labels.sum()} positives / {len(labels)} total "
             f"({labels.mean()*100:.1f}%)")
    return labels


def build_real_labels(df: pd.DataFrame, perf_log: pd.DataFrame) -> pd.Series | None:
    """
    Use actual forward returns from performance_log.csv to create labels.
    Label = 1 if ticker's next-week return was in top quintile of that week's universe.
    Returns None if insufficient history.
    """
    if perf_log is None or len(perf_log) < 50:
        return None

    required = ["symbol", "week_of", "forward_return_1w"]
    if not all(c in perf_log.columns for c in required):
        return None

    # Use last 12 weeks of data
    recent = perf_log.sort_values("week_of").groupby("week_of").apply(
        lambda g: g.assign(
            label=(g["forward_return_1w"] >= g["forward_return_1w"].quantile(0.80)).astype(int)
        )
    ).reset_index(drop=True)

    recent = recent[recent["week_of"] >= recent["week_of"].max() - pd.Timedelta(weeks=12)]

    # Merge to current df on symbol
    label_map = recent.groupby("symbol")["label"].mean()  # avg label across weeks
    labels    = df["symbol"].map(label_map)

    coverage = labels.notna().mean()
    log.info(f"  Real labels coverage: {coverage*100:.1f}% of universe")

    if coverage < 0.30:
        log.info("  Coverage too low — falling back to bootstrap labels")
        return None

    # Fill missing with bootstrap
    bootstrap = build_bootstrap_labels(df)
    labels    = labels.fillna(bootstrap.astype(float))
    return (labels >= 0.5).astype(int)


def train_model(df: pd.DataFrame, labels: pd.Series) -> object:
    """Train LightGBM classifier on signal features."""
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed — run: pip install lightgbm")
        raise

    features = [f for f in SIGNAL_FEATURES if f in df.columns]
    X = df[features].fillna(0.5).values
    y = labels.values

    log.info(f"  Training on {len(X)} samples, {len(features)} features")
    log.info(f"  Class balance: {y.mean()*100:.1f}% positive")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
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

    # Feature importance log
    importances = pd.Series(model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=False)
    log.info("  Top 10 feature importances:")
    for feat, imp in importances.head(10).items():
        log.info(f"    {feat:<35} {imp:.0f}")

    return model


# ── FALLBACK SCORER (no LightGBM) ─────────────────────────────────────────────

def weighted_composite_score(df: pd.DataFrame, weights: dict, regime: str) -> pd.Series:
    """
    Pure weighted average fallback — used if LightGBM unavailable.
    Applies signal weights × regime multipliers directly.
    """
    base_w = weights["signal_weights"]
    reg_m  = weights["regime_multipliers"].get(regime, {})

    components = {
        "sig_momentum":     base_w.get("momentum",     0.30) * reg_m.get("momentum",     1.0),
        "sig_catalyst":     base_w.get("catalyst",     0.25) * reg_m.get("catalyst",     1.0),
        "sig_fundamentals": base_w.get("fundamentals", 0.15) * reg_m.get("fundamentals", 1.0),
        "sig_sentiment":    base_w.get("sentiment",    0.15) * reg_m.get("sentiment",    1.0),
    }

    total_w = sum(components.values())
    score   = pd.Series(0.0, index=df.index)

    for col, w in components.items():
        if col in df.columns:
            score += df[col].fillna(0.5) * (w / total_w)

    return score


# ── RANKING ───────────────────────────────────────────────────────────────────

def build_output(df: pd.DataFrame, raw_scores: pd.Series,
                 regime_data: dict) -> pd.DataFrame:
    """
    Build final scored + ranked DataFrame.
    """
    out = df[META_COLS + [c for c in df.columns if c.startswith("sig_")]].copy()

    out["alpha_score"]    = raw_scores.round(6)
    out["alpha_pct_rank"] = raw_scores.rank(pct=True).round(4)
    out["alpha_rank"]     = raw_scores.rank(ascending=False, method="min").astype(int)

    # Conviction tier
    p = out["alpha_pct_rank"]
    out["conviction"] = pd.cut(
        p,
        bins=[0, 0.50, 0.70, 0.85, 0.93, 1.01],
        labels=["low", "moderate", "elevated", "high", "very_high"],
    )

    # Regime context
    out["regime"]          = regime_data["regime"]
    out["regime_composite"] = regime_data["composite"]
    out["scored_at"]        = datetime.today().strftime("%Y-%m-%d")

    # Sort by rank
    out = out.sort_values("alpha_rank").reset_index(drop=True)

    return out


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading signals...")
    df = pd.read_csv(SIGNALS_CSV)
    log.info(f"  {len(df):,} symbols")

    log.info("Loading regime...")
    with open(REGIME_JSON) as f:
        regime_data = json.load(f)
    regime = regime_data["regime"]
    log.info(f"  Regime: {regime}")

    log.info("Loading weights...")
    with open(Path("config/weights.json")) as f:
        weights = json.load(f)

    # Load performance log if it exists
    perf_log = None
    if PERF_LOG.exists():
        try:
            perf_log = pd.read_csv(PERF_LOG, parse_dates=["week_of"])
            log.info(f"  Performance log: {len(perf_log):,} rows")
        except Exception as e:
            log.warning(f"  Could not load performance log: {e}")

    # ── TRAIN OR LOAD MODEL ───────────────────────────────────────────────────
    features = [f for f in SIGNAL_FEATURES if f in df.columns]

    if MODEL_FILE.exists():
        log.info(f"Loading existing model from {MODEL_FILE}...")
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        mode = "score"
    else:
        log.info("No model found — bootstrapping initial model...")

        # Prefer real labels if enough history exists
        labels = build_real_labels(df, perf_log)
        if labels is None:
            log.info("  Using bootstrap heuristic labels")
            labels = build_bootstrap_labels(df)
        else:
            log.info("  Using real forward-return labels")

        model = train_model(df, labels)

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        log.info(f"  Model saved → {MODEL_FILE}")
        mode = "train"

    # ── SCORE ─────────────────────────────────────────────────────────────────
    log.info(f"Scoring {len(df):,} tickers...")

    try:
        X          = df[features].fillna(0.5).values
        raw_scores = pd.Series(model.predict_proba(X)[:, 1], index=df.index)
        log.info(f"  LightGBM scores: mean={raw_scores.mean():.4f}  "
                 f"std={raw_scores.std():.4f}")
    except Exception as e:
        log.warning(f"  LightGBM scoring failed ({e}) — using weighted fallback")
        raw_scores = weighted_composite_score(df, weights, regime)

    # ── BUILD OUTPUT ──────────────────────────────────────────────────────────
    out = build_output(df, raw_scores, regime_data)

    out.to_csv(OUTPUT, index=False)

    log.info(f"Scores written → {OUTPUT}  ({len(out):,} rows)")
    log.info(f"Mode: {mode}")
    log.info("Top 10 by alpha score:")
    top = out.head(10)[["alpha_rank", "symbol", "sector",
                         "alpha_score", "alpha_pct_rank", "conviction"]]
    for _, row in top.iterrows():
        log.info(f"  #{row['alpha_rank']:<4} {row['symbol']:<8} "
                 f"{str(row['sector']):<25} "
                 f"score={row['alpha_score']:.4f}  "
                 f"conviction={row['conviction']}")

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
