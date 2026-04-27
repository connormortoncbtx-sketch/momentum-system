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

──────────────────────────────────────────────────────────────────────────────
2026-04-26 PATCH NOTES — Sunday weekend_refresh crash + latent issues:

1. alpha_rank crash: `.astype(int)` on rank() output crashes when raw_scores
   contains NaN (which it now does, since the upstream coverage gate in
   04_model.py marks ~30 tickers as NaN-scored each week). Same root cause
   and same one-character fix as the EV stage in 04_model.py: switch to
   nullable Int64. This was the proximate cause of the Sunday refresh
   exploding mid-run.

2. rank_change NA hazard: with alpha_rank now nullable Int64, the rank_change
   lambda's `int(r["alpha_rank"])` would crash on rows whose rank is <NA>.
   Rewrote as a vectorized symbol-aligned subtraction with explicit Int64
   handling — both faster and NA-safe.

3. raw_scores positional assignment: `out["alpha_score"] = raw_scores.values`
   assumes `out`'s row order matches `raw_scores`' row order. `out` is built
   from previous_scores; raw_scores is built from signals (via rescore's
   reindex). Symbol orderings are not guaranteed to match — if they ever
   diverge, alpha_scores get silently paired with wrong tickers. Fixed by
   aligning raw_scores to out's symbol order before assignment.
──────────────────────────────────────────────────────────────────────────────
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

# Max magnitude by which weekend refresh can shift any single ticker's
# alpha_score. Consistent with the LLM conviction_adjustment bound elsewhere
# in the system. Chosen to allow meaningful rank reshuffling when catalyst
# data changes substantially, while preventing the "top pick to rank 555"
# swings that previously occurred when full LightGBM rescoring was performed
# against a mix of Friday signals + Sunday catalyst + NaN-as-0.5 substitutions.
MAX_REFRESH_ADJUSTMENT = 0.15


def rescore(signals: pd.DataFrame,
            previous_scores: pd.DataFrame = None) -> pd.Series:
    """Compute refreshed alpha scores as BOUNDED TWEAKS to Friday's scores.

    Previous implementation: full LightGBM re-prediction on the updated
    feature matrix. This was architecturally unsound -- the model was trained
    on rows with real data, but the refresh feeds it a mix of Friday signals,
    Sunday catalyst, and NaN-replaced-as-0.5 values. Small input perturbations
    compound to large output swings because it is effectively extrapolating
    into a regime the model never saw in training. Result: the 2026-04-20
    trading incident, where WATT moved from alpha=0.93 to 0.097 purely from
    refresh-side noise with no corresponding real-world change.

    New approach: anchor to Friday's alpha_score. Compute a bounded adjustment
    based on how much the CATALYST signal moved between Friday and Sunday
    (the only signal the refresh actually updates). Apply that adjustment
    clipped to +/- MAX_REFRESH_ADJUSTMENT. Re-rank from the tweaked scores.

    This guarantees:
    - If catalyst didn't change, scores are identical to Friday
    - If catalyst changed modestly, scores shift modestly
    - If catalyst changed wildly, score shifts are still bounded so no ticker
      can vanish from the top 10 or appear from obscurity on refresh alone
    - The LightGBM model is never invoked in the refresh path, removing the
      NaN->0.5 extrapolation problem entirely
    """
    if previous_scores is None or "alpha_score" not in previous_scores.columns:
        log.warning("  rescore() called without previous_scores -- cannot anchor. "
                    "Falling back to weighted rescore (old behavior).")
        return weighted_rescore(signals)

    # Build a lookup: symbol -> Friday's alpha_score, Friday's sig_catalyst
    prev_indexed    = previous_scores.set_index("symbol") if "symbol" in previous_scores.columns else previous_scores
    signals_indexed = signals.set_index("symbol") if "symbol" in signals.columns else signals

    # Align indices on symbol
    common_symbols = prev_indexed.index.intersection(signals_indexed.index)
    log.info(f"  Anchor rescore: {len(common_symbols):,} symbols present in both "
             f"(previous={len(prev_indexed):,}, signals={len(signals_indexed):,})")

    friday_alpha = pd.to_numeric(
        prev_indexed.loc[common_symbols, "alpha_score"], errors="coerce"
    )

    # Friday's catalyst value: prefer sig_catalyst (composite) from previous_scores,
    # fall back to signals.csv if not present in previous_scores.
    if "sig_catalyst" in prev_indexed.columns and prev_indexed["sig_catalyst"].notna().any():
        friday_catalyst = pd.to_numeric(
            prev_indexed.loc[common_symbols, "sig_catalyst"], errors="coerce"
        )
    else:
        # No previous catalyst -- we can't compute a delta, so adjustment will be zero
        log.warning("  Previous scores missing sig_catalyst -- no adjustment possible this refresh")
        friday_catalyst = pd.Series(0.5, index=common_symbols)

    # Sunday's catalyst (just computed by refresh)
    sunday_catalyst = pd.to_numeric(
        signals_indexed.loc[common_symbols, "sig_catalyst"], errors="coerce"
    )

    # Catalyst delta. NaN handling: if either side is NaN, delta is 0 (no adjustment).
    # This is stricter than fillna(0.5) because we're saying "if we couldn't measure
    # the catalyst change, don't pretend we did."
    delta_catalyst = (sunday_catalyst - friday_catalyst).fillna(0.0)

    # Translate catalyst delta into alpha adjustment.
    # Scaling: the adjustment is delta_catalyst x catalyst_weight. E.g., if catalyst
    # weight is 0.25 and catalyst swung +0.4, raw adjustment is +0.10. A larger swing
    # (+0.8) would produce +0.20, which then gets clipped to +MAX_REFRESH_ADJUSTMENT.
    try:
        with open(Path("config/weights.json")) as f:
            weights = json.load(f)
        cat_weight = float(weights.get("signal_weights", {}).get("catalyst", 0.25))
    except Exception:
        cat_weight = 0.25

    raw_adjustment = delta_catalyst * cat_weight
    adjustment     = raw_adjustment.clip(-MAX_REFRESH_ADJUSTMENT, MAX_REFRESH_ADJUSTMENT)

    # Apply to Friday's alpha. Preserve NaN: if friday_alpha is NaN (gate-excluded
    # row from upstream), the result stays NaN -- we do NOT want to replace gate
    # exclusions with valid scores via the refresh path. Otherwise clip to [0,1].
    new_alpha_raw = friday_alpha + adjustment
    new_alpha = new_alpha_raw.where(friday_alpha.isna(), new_alpha_raw.clip(0.0, 1.0))

    # Stats for logging
    n_moved_up   = int((adjustment >  0.001).sum())
    n_moved_down = int((adjustment < -0.001).sum())
    n_clipped_up = int((raw_adjustment >  MAX_REFRESH_ADJUSTMENT).sum())
    n_clipped_dn = int((raw_adjustment < -MAX_REFRESH_ADJUSTMENT).sum())
    mean_abs_adj = float(adjustment.abs().mean())
    max_adj      = float(adjustment.abs().max())

    log.info(f"  Adjustments: {n_moved_up} moved up, {n_moved_down} moved down, "
             f"{len(common_symbols) - n_moved_up - n_moved_down} unchanged")
    log.info(f"  Adjustment magnitude: mean={mean_abs_adj:.4f}, max={max_adj:.4f}")
    if n_clipped_up or n_clipped_dn:
        log.info(f"  Clipped at bound: {n_clipped_up} upward, {n_clipped_dn} downward "
                 f"(bound = +/-{MAX_REFRESH_ADJUSTMENT})")

    # Return as a symbol-indexed Series. Callers must align by symbol, NOT
    # positionally -- previous_scores and signals do not necessarily share row
    # order. This contract is enforced in rebuild_scores via reindex on symbol.
    # Rows in signals but not in previous_scores get NaN here, which propagates
    # to alpha_score = NaN downstream -- they fall off the bottom of all
    # rankings, same as gate-excluded rows.
    result = pd.Series(np.nan, index=signals_indexed.index, name="alpha_score")
    result.loc[common_symbols] = new_alpha
    return result


def weighted_rescore(signals: pd.DataFrame) -> pd.Series:
    """Fallback weighted composite if model unavailable.

    Returns a symbol-indexed Series to match the contract of rescore() -- both
    must return symbol-indexed so rebuild_scores can align by symbol rather
    than by positional index. Previous version returned an integer-indexed
    Series, which silently misaligned when fed into the positional-assignment
    path in rebuild_scores.
    """
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
    score = score / total_w if total_w > 0 else score

    # Convert to symbol-indexed for contract consistency with rescore()
    if "symbol" in signals.columns:
        score.index = signals["symbol"].values
    return score


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
                 # Liquidity exclusion flags (set by stage 4). Must be preserved
                 # so weekend refresh can respect them when recomputing composite_rank.
                 "excluded_by_liquidity", "exclusion_reason",
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

    # ── ALPHA ASSIGNMENT (symbol-aligned, NOT positional) ────────────────────
    # raw_scores is symbol-indexed (per rescore()'s contract). out's row order
    # comes from previous_scores, which may not match. Reindex by symbol before
    # assignment to guarantee each ticker gets paired with its own alpha score.
    # Previous code did `raw_scores.values` (positional), which would silently
    # corrupt scores if the orderings ever diverged.
    if isinstance(raw_scores, pd.Series) and raw_scores.index.name != signals.index.name:
        # Symbol-indexed: align by symbol
        aligned_alpha = pd.to_numeric(
            raw_scores.reindex(out["symbol"]).values, errors="coerce"
        )
    else:
        # Positional fallback (only used if a caller bypassed rescore)
        aligned_alpha = raw_scores.values

    out["alpha_score"]    = aligned_alpha
    alpha_series          = pd.Series(aligned_alpha, index=out.index)
    out["alpha_pct_rank"] = alpha_series.rank(pct=True).round(4)
    # Int64 (nullable) to handle NaN alpha_scores from the upstream coverage
    # gate. Same fix as 04_model.py. With plain int, this crashed weekend
    # refresh whenever any rows were gate-excluded (i.e. every week).
    out["alpha_rank"]     = alpha_series.rank(ascending=False, method="min").astype("Int64")

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

    # ── RANK CHANGE (NA-safe vectorized) ─────────────────────────────────────
    # Previous implementation used `.apply(lambda r: int(r["alpha_rank"]) - ...)`
    # which crashes on the gate-excluded rows whose alpha_rank is now nullable
    # <NA>. Vectorized symbol-aligned subtraction handles NA correctly: the
    # arithmetic returns <NA> for rows missing on either side, then we coerce
    # those to 0 (no rank change recorded for unrankable tickers).
    if "alpha_rank" in previous_scores.columns:
        prev_ranks = pd.to_numeric(
            previous_scores.set_index("symbol")["alpha_rank"], errors="coerce"
        )
        prev_aligned = prev_ranks.reindex(out["symbol"]).values
        cur_aligned  = pd.to_numeric(out["alpha_rank"], errors="coerce").values
        # Both sides are nullable; arithmetic preserves NaN. fillna(0) keeps
        # the schema consistent for downstream consumers (the sorting and
        # logging expect rank_change to be a regular int column, not Int64).
        diff = pd.Series(prev_aligned - cur_aligned, index=out.index).fillna(0)
        out["rank_change"] = diff.astype(int)
    else:
        out["rank_change"] = 0

    # Sort by composite_rank if available (matches what alpaca_trader uses
    # for picking), falling back to alpha_rank if composite hasn't been
    # computed yet (gets computed below). Previously sorted by alpha_rank,
    # which caused update_index to display a different top-5 than the system
    # actually traded -- cosmetic bug fixed as part of the refresh rewrite.
    # We'll re-sort after composite_rank is recomputed below.
    # na_position="last" so gate-excluded rows (alpha_rank=<NA>) sort to the
    # bottom rather than crashing the sort or appearing at the top.
    out = out.sort_values("alpha_rank", na_position="last").reset_index(drop=True)

    # Recompute composite rank if EV data is available (blends updated alpha rank with preserved EV rank)
    # Respect the liquidity exclusion from stage 4 -- tickers marked as
    # excluded_by_liquidity should keep NaN composite_rank so downstream
    # trading logic (which filters on .notna()) continues to skip them.
    if "ev_pct_rank" in out.columns and out["ev_pct_rank"].notna().any():
        out["alpha_pct_rank_tmp"] = out["alpha_score"].rank(pct=True).round(4)
        composite_score = (
            out["alpha_pct_rank_tmp"].fillna(0.5) * 0.50 +
            out["ev_pct_rank"].fillna(0.5) * 0.50
        )
        # Null out composite score for liquidity-excluded tickers BEFORE ranking
        # so they don't occupy rank positions in the refreshed output.
        if "excluded_by_liquidity" in out.columns:
            excluded_mask = out["excluded_by_liquidity"].fillna(False).astype(bool)
            composite_score = composite_score.where(~excluded_mask)
        # Also exclude rows whose alpha_score is NaN (coverage-gated upstream)
        # from composite ranking. Same convention used in 04_model.py's
        # build_output: gated rows fall off the bottom of all rankings.
        composite_score = composite_score.where(out["alpha_score"].notna())
        out["composite_rank"] = composite_score.rank(
            ascending=False, method="min"
        ).astype("Int64")
        out = out.drop(columns=["alpha_pct_rank_tmp"])

        # Final sort: by composite_rank (ascending = best first).
        # NaN composite_rank values (liquidity-excluded) sort last, which is
        # correct behavior -- they should appear at the bottom of any display,
        # not at the top.
        out = out.sort_values("composite_rank", na_position="last").reset_index(drop=True)

    return out


# ── CHANGED TICKERS SUMMARY ───────────────────────────────────────────────────

def log_notable_changes(scores: pd.DataFrame, run_label: str):
    """Log tickers that moved significantly in rank since Friday."""
    if "rank_change" not in scores.columns:
        return

    # Big movers — jumped 100+ ranks. Cast alpha_rank to numeric for safe
    # int() conversion -- it's nullable Int64 now and pd.NA can't be int()'d
    # directly. dropna here is correct: gate-excluded rows have no rank, so
    # they can't be "movers" in any meaningful sense.
    rankable = scores.dropna(subset=["alpha_rank"]).copy()
    risers   = rankable[rankable["rank_change"] >= 100].head(10)
    fallers  = rankable[rankable["rank_change"] <= -100].head(5)

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

        # Rescore -- bounded-tweak adjustment, anchored to Friday's alpha_score
        log.info("Rescoring with bounded tweak from refreshed catalyst data...")
        raw_scores = rescore(signals, previous_scores=previous_scores)

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

        # Regenerate report -- write back to the most recent existing report,
        # not to a computed "last Friday" filename.
        #
        # Timezone trap: stage 6 in the Friday pipeline runs with UTC
        # datetime.today() on the GitHub runner. The pipeline fires Fri 8pm CT
        # which is Sat 00:00 UTC, so the pipeline names its report with
        # Saturday's UTC date (e.g. 2026-04-18.html for a Fri Apr 17 pipeline).
        # Meanwhile weekend_refresh computing "last Friday" via local weekday
        # arithmetic returns the CT-calendar Friday (Apr 17), which doesn't
        # match the existing file. The refresh then silently wrote to
        # 2026-04-17.html, creating an unreferenced duplicate while the actual
        # Pages-displayed 2026-04-18.html stayed stale.
        #
        # Solution: find the newest existing report file and write to it.
        log.info("Regenerating report...")
        from pathlib import Path as _Path
        reports_dir = _Path("docs/reports")
        existing = sorted(reports_dir.glob("????-??-??.html"), reverse=True)
        if existing:
            target = existing[0]
            target_str = target.stem  # YYYY-MM-DD without extension
            log.info(f"  Writing to most recent existing report: {target.name}")
        else:
            # Fallback: no existing reports, use today's UTC date to match
            # what stage 6 would produce on a fresh pipeline run.
            from datetime import datetime as _dt
            target_str = _dt.utcnow().strftime("%Y-%m-%d")
            log.warning(f"  No existing reports found; writing to UTC today: {target_str}.html")
        report = importlib.import_module("pipeline.06_report")
        report.run(date_override=target_str)

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
