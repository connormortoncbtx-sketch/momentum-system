"""
automation/weekend_refresh.py
==============================
Lightweight refresh that re-runs only the catalyst signal and LLM
synthesis against the existing universe and scores.

Designed to run:
    - Sunday 8:00 PM ET   — catches weekend news, FDA decisions,
                            M&A announcements, Sunday analyst notes
    - Monday 6:00 AM ET   — final refresh before pre-market monitor fires

Does NOT re-run:
    - Universe builder (universe doesn't change over a weekend)
    - Regime classifier (macro regime doesn't shift intraday)
    - Momentum signal (price-based, markets closed)
    - Fundamentals signal (quarterly data, doesn't change)
    - Sentiment signal (slow-moving analyst trends)

DOES re-run:
    - Catalyst signal     (earnings proximity, fresh insider filings,
                           breaking news, analyst upgrades)
    - LLM synthesis       (re-synthesizes top 50 with updated catalyst data)
    - Report generator    (rebuilds HTML with updated scores)
    - Index updater       (refreshes docs/index.html)

Net result: ranking can shift significantly if a major catalyst
event happened since Friday's run.
"""

import json
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ensure repo root is on path so pipeline modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automation.tz_utils import now_ct, format_ct, is_dst
from automation.system_logger import log_event, LogStatus
from automation.notifier import notify_alert, notify_error

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR    = Path("data")
SIGNALS_CSV = DATA_DIR / "signals.csv"
SCORES_CSV  = DATA_DIR / "scores_final.csv"
REGIME_JSON = DATA_DIR / "regime.json"
REFRESH_LOG = DATA_DIR / "refresh_log.json"


# ── VALIDATION ────────────────────────────────────────────────────────────────

def check_prerequisites() -> bool:
    """Ensure Friday's pipeline has run before attempting refresh."""
    missing = []
    for f in [SIGNALS_CSV, SCORES_CSV, REGIME_JSON]:
        if not f.exists():
            missing.append(str(f))
    if missing:
        log.error(f"Missing prerequisite files: {missing}")
        log.error("Run the weekly pipeline first before attempting a refresh")
        return False
    return True


# Top N tickers to re-run catalyst on — matches 03_signals.py tier size
SLOW_SIGNAL_TIER = 600


# ── CATALYST REFRESH ──────────────────────────────────────────────────────────

def refresh_catalyst(universe: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run catalyst signal on top N tickers by existing momentum score.
    Everything outside the tier keeps its existing catalyst signal values.
    """
    from pipeline.signals.catalyst import score as catalyst_score

    n_total   = len(universe)
    tier_size = min(SLOW_SIGNAL_TIER, n_total)

    # Determine tier 2 by existing momentum score
    if "sig_momentum" in signals.columns:
        ranked    = signals[["symbol","sig_momentum"]].copy()
        ranked["_rank"] = ranked["sig_momentum"].rank(ascending=False, method="min")
        tier2_syms = set(ranked[ranked["_rank"] <= tier_size]["symbol"].tolist())
    else:
        tier2_syms = set(universe["symbol"].tolist())

    tier2 = universe[universe["symbol"].isin(tier2_syms)].copy()
    tier3 = universe[~universe["symbol"].isin(tier2_syms)].copy()

    log.info(f"  Catalyst refresh: {len(tier2):,} tier-2 symbols "
             f"(+{len(tier3):,} keeping existing scores)")

    # Run catalyst only on tier 2
    fresh     = catalyst_score(tier2)
    cat_cols  = [c for c in fresh.columns if c.startswith("sig_catalyst")]

    # For tier 3, pull existing catalyst scores from signals
    existing_cats = [c for c in cat_cols if c in signals.columns]
    if existing_cats and len(tier3) > 0:
        tier3_cats = signals[signals["symbol"].isin(tier3["symbol"])][
            ["symbol"] + existing_cats].copy()
    else:
        tier3_cats = tier3[["symbol"]].copy()
        for col in cat_cols:
            tier3_cats[col] = float("nan")

    result = pd.concat([fresh[["symbol"] + cat_cols], tier3_cats], ignore_index=True)
    return result


def merge_refreshed_signals(signals: pd.DataFrame,
                            fresh_catalyst: pd.DataFrame) -> pd.DataFrame:
    """
    Replace catalyst signal columns in signals.csv with fresh values.
    All other signals remain from Friday's run.
    """
    # Drop old catalyst columns
    old_cat_cols = [c for c in signals.columns if c.startswith("sig_catalyst")]
    signals = signals.drop(columns=old_cat_cols)

    # Merge fresh catalyst
    signals = signals.merge(fresh_catalyst, on="symbol", how="left")

    # Re-apply regime adjustments to the new catalyst columns
    with open(Path("config/weights.json")) as f:
        weights = json.load(f)

    with open(REGIME_JSON) as f:
        regime_data = json.load(f)

    regime     = regime_data["regime"]
    base_w     = weights["signal_weights"]
    regime_m   = weights["regime_multipliers"].get(regime, {})

    cat_w = base_w.get("catalyst", 0.25)
    cat_r = regime_m.get("catalyst", 1.0)
    signals["sig_catalyst_adj"] = signals["sig_catalyst"].fillna(0.5) * cat_w * cat_r

    log.info(f"Catalyst signal refreshed — new mean: {signals['sig_catalyst'].mean():.4f}")
    return signals


# ── RESCORE ───────────────────────────────────────────────────────────────────

def rescore(signals: pd.DataFrame) -> pd.DataFrame:
    """Re-run the LightGBM model with updated signal features."""
    import pickle
    from pathlib import Path as P

    model_file = P("models/lgbm_model.pkl")
    if not model_file.exists():
        log.warning("No model found — using weighted fallback for rescoring")
        return weighted_rescore(signals)

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Detect how many features the model expects and use matching feature set
    n_expected = model.n_features_in_

    FEATURES_FULL = [
        "sig_momentum_rs", "sig_momentum_trend", "sig_momentum_vol_surge",
        "sig_momentum_breakout", "sig_catalyst_earnings", "sig_catalyst_insider",
        "sig_catalyst_analyst", "sig_fund_growth", "sig_fund_quality",
        "sig_fund_profitability", "sig_fund_value", "sig_sentiment_news",
        "sig_sentiment_analyst", "sig_sentiment_short",
        "sig_momentum_adj", "sig_catalyst_adj", "sig_fundamentals_adj",
        "sig_sentiment_adj",
    ]
    FEATURES_COMPOSITE = [
        "sig_momentum", "sig_catalyst", "sig_fundamentals", "sig_sentiment",
    ]

    # Pick feature set that matches model expectation
    if n_expected <= 4:
        feature_candidates = FEATURES_COMPOSITE
        log.info(f"  Model expects {n_expected} features — using composite signals")
    else:
        feature_candidates = FEATURES_FULL
        log.info(f"  Model expects {n_expected} features — using sub-signals")

    features = [f for f in feature_candidates if f in signals.columns]

    if len(features) < n_expected:
        log.warning(f"  Only {len(features)} of {n_expected} expected features available "
                    f"— falling back to weighted rescore")
        return weighted_rescore(signals)

    X      = signals[features].fillna(0.5).values
    scores = pd.Series(model.predict_proba(X)[:, 1], index=signals.index)

    log.info(f"Rescore complete — mean={scores.mean():.4f}  std={scores.std():.4f}")
    return scores


def weighted_rescore(signals: pd.DataFrame) -> pd.Series:
    """Fallback weighted composite if model unavailable."""
    with open(Path("config/weights.json")) as f:
        weights = json.load(f)
    with open(REGIME_JSON) as f:
        regime_data = json.load(f)

    regime  = regime_data["regime"]
    base_w  = weights["signal_weights"]
    regime_m = weights["regime_multipliers"].get(regime, {})

    score = pd.Series(0.0, index=signals.index)
    for sig, col in [("momentum","sig_momentum"),("catalyst","sig_catalyst"),
                     ("fundamentals","sig_fundamentals"),("sentiment","sig_sentiment")]:
        if col in signals.columns:
            w = base_w.get(sig, 0.25) * regime_m.get(sig, 1.0)
            score += signals[col].fillna(0.5) * w

    total_w = sum(base_w.get(s,0.25) * regime_m.get(s,1.0)
                  for s in ["momentum","catalyst","fundamentals","sentiment"])
    return score / total_w if total_w > 0 else score


# ── BUILD UPDATED SCORES ──────────────────────────────────────────────────────

def rebuild_scores(signals: pd.DataFrame, raw_scores: pd.Series,
                   regime_data: dict, previous_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild scores_final.csv with updated alpha scores.
    Preserves non-signal columns from previous run (name, sector, etc.)
    """
    # Start from previous scores, update the alpha columns
    # Preserve EV columns and composite rank -- these are not recomputed by weekend refresh
    # scored_at is preserved to reflect the original Friday scoring date (not the refresh date).
    # This is critical for collect_returns.py's 5-trading-day guard -- if we overwrote scored_at
    # to Sunday/Monday, the Friday learning loop would see elapsed<5 days and skip every week.
    #
    # Column-ratchet fix: if a column is missing from previous_scores (because a prior broken
    # refresh dropped it), fall back to pulling from signals.csv. This keeps sub-signal columns
    # from silently disappearing across weekend refreshes.
    meta_cols = ["symbol", "name", "exchange", "sector", "industry",
                 "last_price", "avg_vol_20d", "market_cap",
                 # EV / risk metadata
                 "ev_score", "ev_rank", "ev_pct_rank", "ev_conviction",
                 "avg_win_magnitude", "avg_loss_magnitude", "weekly_vol",
                 "composite_rank",
                 # Stop parameters — persisted by stage 4 so alpaca_trader can read them.
                 # Must be preserved across refreshes or alpaca_trader falls back to None
                 # and never places hard stops.
                 "suggested_hard_stop_pct", "suggested_activation_pct", "suggested_trail_pct",
                 # Timestamp metadata
                 "scored_at",
                 # Momentum sub-signals
                 "sig_momentum_rs", "sig_momentum_trend", "sig_momentum_vol_surge",
                 "sig_momentum_breakout",
                 # Catalyst sub-signals (including the ones previously omitted, causing
                 # asymmetric preservation where fundamentals/sentiment had _adj but catalyst didn't)
                 "sig_catalyst_earnings", "sig_catalyst_insider", "sig_catalyst_analyst",
                 "sig_catalyst_adj",
                 # Fundamentals sub-signals
                 "sig_fund_growth", "sig_fund_quality", "sig_fund_profitability",
                 "sig_fund_value",
                 # Sentiment sub-signals
                 "sig_sentiment_news", "sig_sentiment_analyst",
                 "sig_sentiment_short", "sig_sentiment_articles",
                 # Regime-adjusted composites
                 "sig_momentum_adj", "sig_fundamentals_adj", "sig_sentiment_adj"]

    # Build out from previous_scores; for any meta_col missing there, try signals.csv.
    # This prevents column ratchet (columns silently disappearing across refreshes).
    out = previous_scores[[c for c in meta_cols if c in previous_scores.columns]].copy()
    if "symbol" not in out.columns:
        out.insert(0, "symbol", previous_scores["symbol"].values)

    signals_indexed = signals.set_index("symbol") if "symbol" in signals.columns else None
    recovered_from_signals = []
    for col in meta_cols:
        if col in out.columns or col == "symbol":
            continue
        if signals_indexed is not None and col in signals_indexed.columns:
            out[col] = signals_indexed[col].reindex(out["symbol"]).values
            recovered_from_signals.append(col)

    if recovered_from_signals:
        log.info(f"  Recovered {len(recovered_from_signals)} columns from signals.csv "
                 f"(previous scores was missing them): {recovered_from_signals}")

    # Merge updated signal composites
    sig_cols = ["sig_momentum","sig_catalyst","sig_fundamentals","sig_sentiment"]
    for col in sig_cols:
        if col in signals.columns:
            out[col] = signals.set_index("symbol")[col].reindex(
                out["symbol"]).values

    out["alpha_score"]    = raw_scores.values
    out["alpha_pct_rank"] = raw_scores.rank(pct=True).round(4)
    out["alpha_rank"]     = raw_scores.rank(ascending=False, method="min").astype(int)

    p = out["alpha_pct_rank"]
    out["conviction"] = pd.cut(
        p,
        bins=[0, 0.50, 0.70, 0.85, 0.93, 1.01],
        labels=["low","moderate","elevated","high","very_high"],
    )

    out["regime"]           = regime_data["regime"]
    out["regime_composite"] = regime_data["composite"]
    # Record when this refresh ran -- distinct from scored_at (the original Friday scoring date).
    # The Friday learning loop's 5-trading-day guard reads scored_at, so we MUST NOT overwrite it.
    today_str = datetime.today().strftime("%Y-%m-%d")
    out["refreshed_at"] = today_str
    # Defensive fallback: if scored_at didn't make it through meta preservation (e.g., very
    # first run of this fix against a scores file lacking the column), seed it with today.
    # Next Friday's pipeline will write the correct Friday date.
    if "scored_at" not in out.columns or out["scored_at"].isna().all():
        out["scored_at"] = today_str

    # Flag what changed vs Friday
    if "alpha_rank" in previous_scores.columns:
        prev_ranks = previous_scores.set_index("symbol")["alpha_rank"]
        out["rank_change"] = out.apply(
            lambda r: int(prev_ranks.get(r["symbol"], r["alpha_rank"])) - int(r["alpha_rank"]),
            axis=1
        )
    else:
        out["rank_change"] = 0

    out = out.sort_values("alpha_rank").reset_index(drop=True)

    # Recompute composite rank if EV data is available (blends updated alpha rank with preserved EV rank)
    if "ev_pct_rank" in out.columns and out["ev_pct_rank"].notna().any():
        out["alpha_pct_rank_tmp"] = out["alpha_score"].rank(pct=True).round(4)
        out["composite_rank"] = (
            out["alpha_pct_rank_tmp"].fillna(0.5) * 0.50 +
            out["ev_pct_rank"].fillna(0.5) * 0.50
        ).rank(ascending=False, method="min").astype(int)
        out = out.drop(columns=["alpha_pct_rank_tmp"])

    return out


# ── CHANGED TICKERS SUMMARY ───────────────────────────────────────────────────

def log_notable_changes(scores: pd.DataFrame, run_label: str):
    """Log tickers that moved significantly in rank since Friday."""
    if "rank_change" not in scores.columns:
        return

    # Big movers — jumped 100+ ranks
    risers = scores[scores["rank_change"] >= 100].head(10)
    fallers = scores[scores["rank_change"] <= -100].head(5)

    if not risers.empty:
        log.info(f"\n{run_label} — Notable rank RISERS (new catalyst?):")
        for _, r in risers.iterrows():
            log.info(f"  #{int(r['alpha_rank']):<5} {r['symbol']:<8} "
                     f"↑{int(r['rank_change'])} ranks  "
                     f"cat={r.get('sig_catalyst',0):.3f}  "
                     f"{str(r.get('thesis',''))[:60]}")

    if not fallers.empty:
        log.info(f"\n{run_label} — Notable rank FALLERS:")
        for _, r in fallers.iterrows():
            log.info(f"  #{int(r['alpha_rank']):<5} {r['symbol']:<8} "
                     f"↓{abs(int(r['rank_change']))} ranks")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(run_label: str = "weekend_refresh"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_event("weekend_refresh", LogStatus.INFO,
              f"Starting weekend refresh ({run_label})")

    # Holiday check — skip if not a full 5-day trading week
    from automation.tz_utils import assert_normal_week
    if not assert_normal_week(run_label):
        log_event("weekend_refresh", LogStatus.INFO,
                  "Skipped: not a normal trading week")
        return

    # Idempotency guard — prevent double-run when both CDT and CST crons fire
    today    = now_ct().strftime("%Y-%m-%d")
    lock_key = f"{today}_{run_label}"
    lock_file = DATA_DIR / ".refresh_lock"

    if lock_file.exists():
        last_run = lock_file.read_text().strip()
        if last_run == lock_key:
            log.info(f"Already ran {run_label} today ({today}) — skipping duplicate cron trigger")
            log_event("weekend_refresh", LogStatus.INFO,
                      f"Skipped: {run_label} already ran today ({today})")
            return

    lock_file.write_text(lock_key)

    log.info("=" * 60)
    log.info(f"WEEKEND REFRESH — {run_label.upper()}")
    log.info("=" * 60)

    if not check_prerequisites():
        log_event("weekend_refresh", LogStatus.WARNING,
                  "Skipped: prerequisites check failed")
        notify_alert("weekend_refresh",
                     f"{run_label} prerequisites missing — likely means Friday's "
                     f"pipeline hasn't run yet")
        return

    try:
        # Load existing data
        log.info("Loading existing signals and scores...")
        signals        = pd.read_csv(SIGNALS_CSV)
        previous_scores = pd.read_csv(SCORES_CSV)
        universe       = signals[["symbol"]].copy()

        with open(REGIME_JSON) as f:
            regime_data = json.load(f)

        log.info(f"  {len(signals)} symbols  |  regime: {regime_data['regime']}")

        # Re-run catalyst signal
        fresh_catalyst = refresh_catalyst(universe, signals)
        signals        = merge_refreshed_signals(signals, fresh_catalyst)

        # Save updated signals
        signals.to_csv(SIGNALS_CSV, index=False)

        # Rescore
        log.info("Rescoring with updated catalyst data...")
        raw_scores = rescore(signals)

        # Rebuild scores
        scores = rebuild_scores(signals, raw_scores, regime_data, previous_scores)

        # Log notable changes
        log_notable_changes(scores, run_label)

        # Save updated scores (without thesis yet — LLM synthesis adds it)
        scores["thesis"]               = previous_scores.set_index("symbol")["thesis"].reindex(scores["symbol"]).values
        scores["risk_flag"]            = previous_scores.set_index("symbol").get("risk_flag", pd.Series()).reindex(scores["symbol"]).values
        scores["confidence"]           = previous_scores.set_index("symbol").get("confidence", pd.Series()).reindex(scores["symbol"]).values
        scores["thesis_source"]        = "rule_based"
        scores["conviction_adjustment"] = 0.0
        scores.to_csv(SCORES_CSV, index=False)

        # Re-run LLM synthesis on new top 50
        log.info("Re-running LLM synthesis on updated top 50...")
        import importlib
        synth = importlib.import_module("pipeline.05_llm_synthesis")
        synth.run()

        # Regenerate report -- write back to last Friday's file, not today's date
        log.info("Regenerating report...")
        from datetime import date as _date, timedelta as _timedelta
        _today = _date.today()
        _days_since_friday = (_today.weekday() - 4) % 7
        _last_friday = _today - _timedelta(days=_days_since_friday if _days_since_friday > 0 else 7)
        _friday_str = _last_friday.strftime("%Y-%m-%d")
        log.info(f"  Writing to Friday report: {_friday_str}.html")
        report = importlib.import_module("pipeline.06_report")
        report.run(date_override=_friday_str)

        # Update index
        log.info("Updating index...")
        idx = importlib.import_module("automation.update_index")
        idx.run()

        # Write refresh log
        refresh_entry = {
            "run_label":    run_label,
            "timestamp":    datetime.now().isoformat(),
            "regime":       regime_data["regime"],
            "top_5":        scores.head(5)[["symbol","alpha_score","alpha_rank"]].to_dict("records"),
        }
        existing_log = []
        if REFRESH_LOG.exists():
            with open(REFRESH_LOG) as f:
                existing_log = json.load(f)
        existing_log.append(refresh_entry)
        with open(REFRESH_LOG, "w") as f:
            json.dump(existing_log[-20:], f, indent=2)  # keep last 20 entries

        log.info("=" * 60)
        log.info(f"Refresh complete — {run_label}")
        log.info(f"Top 5:")
        for _, r in scores.head(5).iterrows():
            change = f"↑{int(r['rank_change'])}" if r.get('rank_change',0) > 0 else (
                     f"↓{abs(int(r['rank_change']))}" if r.get('rank_change',0) < 0 else "—")
            log.info(f"  #{int(r['alpha_rank']):<4} {r['symbol']:<8} "
                     f"score={r['alpha_score']:.4f}  {change}")
        log.info("=" * 60)

        top1 = scores.iloc[0]
        log_event("weekend_refresh", LogStatus.SUCCESS,
                  f"Refresh complete ({run_label})",
                  metrics={
                      "run_label":     run_label,
                      "symbols":       int(len(scores)),
                      "regime":        regime_data["regime"],
                      "top_pick":      str(top1["symbol"]),
                      "top_score":     round(float(top1["alpha_score"]), 4),
                  })

    except Exception as e:
        log.error(f"weekend_refresh crashed: {e}", exc_info=True)
        log_event("weekend_refresh", LogStatus.ERROR,
                  f"Unhandled exception during {run_label}",
                  errors=[str(e)])
        notify_error("weekend_refresh", f"{run_label} failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else "manual_refresh"
    run(run_label=label)
