# automation/execution_tracker.py
# Tracks execution quality metrics for scaling intelligence.
# Runs as part of the Friday learning loop after collect_returns.py.
#
# Captures per-trade:
#   - Entry/exit slippage (intended vs actual fill)
#   - ADV utilization (position size as % of avg daily volume)
#   - Composite rank at entry
#   - Forward return
#
# Captures per-week basket:
#   - Alpha concentration (% of total return from top 1 and top 2 names)
#   - Signal quality by rank bucket (1-3, 4-6, 7-10, 11+)
#   - Average slippage across all fills
#   - Average ADV utilization
#
# Output: data/execution_log.csv
#
# These metrics accumulate continuously and feed the scaling decision framework:
#   - When avg slippage > 0.5% -> consider TWAP
#   - When ADV utilization > 3% on any name -> consider volume floor increase
#   - When alpha concentration > 60% in top 1 name -> consider position count expansion
#   - When rank bucket 6-10 consistently underperforms 1-5 -> tighten position count

import json
import logging
import os
import sys
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR      = Path("data")
EXEC_LOG      = DATA_DIR / "execution_log.csv"
TRADE_STATE   = DATA_DIR / "alpaca_state.json"
PERF_LOG      = DATA_DIR / "performance_log.csv"
SCORES_CSV    = DATA_DIR / "scores_final.csv"

# Scaling alert thresholds -- when metrics cross these, log a warning
SLIPPAGE_WARN_PCT    = 0.005   # 0.5% avg slippage -> consider TWAP
ADV_UTIL_WARN_PCT    = 0.03    # 3% ADV utilization -> consider volume floor
CONCENTRATION_WARN   = 0.60    # 60% of return from top 1 name -> consider more positions
RANK_QUALITY_WARN    = 0.50    # rank bucket 6-10 returns less than 50% of rank 1-5


# ── ALPACA FILL DATA ──────────────────────────────────────────────────────────

def get_alpaca_fills(week_start: str, week_end: str) -> dict:
    """
    Query Alpaca for actual fill prices on orders placed this week.
    Returns dict: {symbol: {entry_fill, entry_qty, exit_fill, exit_qty}}
    Falls back gracefully if Alpaca is not configured.
    """
    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    paper  = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

    if not key or not secret:
        log.info("Alpaca keys not set -- skipping fill data (manual trades only)")
        return {}

    try:
        import alpaca_trade_api as tradeapi
        base_url = ("https://paper-api.alpaca.markets" if paper
                    else "https://api.alpaca.markets")
        api = tradeapi.REST(key, secret, base_url, api_version="v2")

        # Fetch orders in the week window
        orders = api.list_orders(
            status="filled",
            after=week_start,
            until=week_end,
            limit=500,
        )

        fills = {}
        for order in orders:
            sym  = order.symbol
            side = order.side
            fill_price = float(order.filled_avg_price or 0)
            fill_qty   = float(order.filled_qty or 0)

            if sym not in fills:
                fills[sym] = {}

            if side == "buy":
                fills[sym]["entry_fill"]  = fill_price
                fills[sym]["entry_qty"]   = fill_qty
                fills[sym]["entry_time"]  = str(order.filled_at or "")
            elif side == "sell":
                fills[sym]["exit_fill"]   = fill_price
                fills[sym]["exit_qty"]    = fill_qty
                fills[sym]["exit_time"]   = str(order.filled_at or "")

        log.info(f"Alpaca fills retrieved: {len(fills)} symbols")
        return fills

    except Exception as e:
        log.warning(f"Could not retrieve Alpaca fills: {e}")
        return {}


# ── EXECUTION METRICS ─────────────────────────────────────────────────────────

def compute_execution_metrics(
    week_str: str,
    fills: dict,
    state: dict,
    perf_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> list[dict]:
    """
    Build per-symbol execution metric rows for this week.
    Combines Alpaca fill data with state file intended prices and scores.
    """
    rows = []
    week_perf = perf_df[perf_df["week_of"].astype(str) == week_str] \
        if not perf_df.empty else pd.DataFrame()

    positions = state.get("positions", {})

    # Get all symbols traded this week -- union of fills and state
    all_symbols = set(fills.keys()) | set(positions.keys())

    for sym in all_symbols:
        pos_state  = positions.get(sym, {})
        fill_data  = fills.get(sym, {})

        # Intended prices from state file
        intended_entry = pos_state.get("entry_price_est")
        intended_exit  = None  # Friday MOC -- use fri_close from perf log

        # Actual fills from Alpaca
        actual_entry = fill_data.get("entry_fill")
        actual_exit  = fill_data.get("exit_fill")

        # Entry slippage
        entry_slippage_pct = None
        if intended_entry and actual_entry and intended_entry > 0:
            entry_slippage_pct = (actual_entry - intended_entry) / intended_entry

        # Exit slippage -- compare actual exit to Friday close
        exit_slippage_pct = None
        sym_perf = week_perf[week_perf["symbol"] == sym] if not week_perf.empty else pd.DataFrame()
        fri_close = float(sym_perf["fri_close"].iloc[0]) \
            if not sym_perf.empty and "fri_close" in sym_perf.columns else None

        if actual_exit and fri_close and fri_close > 0:
            # Negative slippage = sold below Friday close (stop triggered early)
            exit_slippage_pct = (actual_exit - fri_close) / fri_close

        # ADV utilization
        adv_utilization = None
        shares = pos_state.get("shares") or fill_data.get("entry_qty")
        if actual_entry and shares:
            position_dollar = float(shares) * float(actual_entry)
            # Get ADV from scores
            score_row = scores_df[scores_df["symbol"] == sym] \
                if not scores_df.empty else pd.DataFrame()
            avg_vol = float(score_row["avg_vol_20d"].iloc[0]) \
                if not score_row.empty and "avg_vol_20d" in score_row.columns else None
            if avg_vol and avg_vol > 0 and actual_entry:
                adv_dollar = avg_vol * actual_entry
                adv_utilization = position_dollar / adv_dollar if adv_dollar > 0 else None

        # Forward return from perf log
        forward_return = None
        phase2_sell_pct = pos_state.get("partial_sell_pct_actual")
        if not sym_perf.empty and "forward_return_1w" in sym_perf.columns:
            forward_return = float(sym_perf["forward_return_1w"].iloc[0])

        # Rank and scores at entry
        composite_rank = pos_state.get("composite_rank") or (
            int(score_row["composite_rank"].iloc[0])
            if not score_row.empty and "composite_rank" in score_row.columns else None
        ) if not scores_df.empty else None

        alpha_score = pos_state.get("alpha_score")
        weekly_vol  = pos_state.get("weekly_vol")

        # Phase info
        phase_reached = pos_state.get("phase", 1)

        rows.append({
            "week_of":               week_str,
            "symbol":                sym,
            "composite_rank":        composite_rank,
            "alpha_score":           alpha_score,
            "weekly_vol":            weekly_vol,
            "intended_entry_price":  intended_entry,
            "actual_entry_price":    actual_entry,
            "entry_slippage_pct":    round(entry_slippage_pct * 100, 4)
                                     if entry_slippage_pct is not None else None,
            "intended_exit_price":   fri_close,
            "actual_exit_price":     actual_exit,
            "exit_slippage_pct":     round(exit_slippage_pct * 100, 4)
                                     if exit_slippage_pct is not None else None,
            "adv_utilization_pct":   round(adv_utilization * 100, 4)
                                     if adv_utilization is not None else None,
            "forward_return_1w":     round(forward_return * 100, 4)
                                     if forward_return is not None else None,
            "phase_reached":         phase_reached,
            "partial_sell_pct":      round(phase2_sell_pct * 100, 2)
                                     if phase2_sell_pct is not None else None,
            "position_size_usd":     round(float(shares or 0) * float(actual_entry or 0), 2),
        })

    return rows


def compute_basket_metrics(rows: list[dict], week_str: str) -> dict:
    """
    Compute week-level basket metrics from per-symbol rows.
    """
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    df = df[df["forward_return_1w"].notna()]

    if df.empty:
        return {}

    # Alpha concentration -- what % of total basket return came from top 1 and top 2
    total_return = df["forward_return_1w"].sum()
    top1_return  = df["forward_return_1w"].max()
    top2_return  = df.nlargest(2, "forward_return_1w")["forward_return_1w"].sum()

    concentration_top1 = (top1_return / total_return) if total_return != 0 else None
    concentration_top2 = (top2_return / total_return) if total_return != 0 else None

    top1_sym = df.loc[df["forward_return_1w"].idxmax(), "symbol"] \
        if not df.empty else None

    # Signal quality by rank bucket
    def bucket_avg(min_rank, max_rank):
        sub = df[(df["composite_rank"] >= min_rank) &
                 (df["composite_rank"] <= max_rank)]
        return round(sub["forward_return_1w"].mean(), 4) \
            if not sub.empty else None

    rank_1_3   = bucket_avg(1, 3)
    rank_4_6   = bucket_avg(4, 6)
    rank_7_10  = bucket_avg(7, 10)
    rank_11_15 = bucket_avg(11, 15)
    rank_16_20 = bucket_avg(16, 20)

    # Slippage averages
    entry_slippage_cols = df["entry_slippage_pct"].dropna()
    exit_slippage_cols  = df["exit_slippage_pct"].dropna()
    avg_entry_slip = round(entry_slippage_cols.mean(), 4) \
        if not entry_slippage_cols.empty else None
    avg_exit_slip  = round(exit_slippage_cols.mean(), 4) \
        if not exit_slippage_cols.empty else None

    # ADV utilization
    adv_cols = df["adv_utilization_pct"].dropna()
    avg_adv_util = round(adv_cols.mean(), 4) if not adv_cols.empty else None
    max_adv_util = round(adv_cols.max(), 4)  if not adv_cols.empty else None

    return {
        "week_of":               week_str,
        "n_positions":           len(df),
        "basket_avg_return":     round(df["forward_return_1w"].mean(), 4),
        "alpha_concentration_1": round(concentration_top1, 4)
                                  if concentration_top1 is not None else None,
        "alpha_concentration_2": round(concentration_top2, 4)
                                  if concentration_top2 is not None else None,
        "top_contributor":       top1_sym,
        "top_contributor_ret":   round(top1_return, 4),
        "rank_1_3_avg_ret":      rank_1_3,
        "rank_4_6_avg_ret":      rank_4_6,
        "rank_7_10_avg_ret":     rank_7_10,
        "rank_11_15_avg_ret":    rank_11_15,
        "rank_16_20_avg_ret":    rank_16_20,
        "avg_entry_slippage_pct": avg_entry_slip,
        "avg_exit_slippage_pct":  avg_exit_slip,
        "avg_adv_utilization_pct": avg_adv_util,
        "max_adv_utilization_pct": max_adv_util,
    }


def log_scaling_alerts(basket: dict):
    """Log warnings when metrics cross scaling thresholds."""
    if not basket:
        return

    alerts = []

    slip = basket.get("avg_entry_slippage_pct")
    if slip and abs(slip) > SLIPPAGE_WARN_PCT * 100:
        alerts.append(f"SLIPPAGE ALERT: avg entry slippage {slip:.2f}% "
                      f"(threshold {SLIPPAGE_WARN_PCT*100:.1f}%) "
                      f"-- consider TWAP execution")

    adv = basket.get("max_adv_utilization_pct")
    if adv and adv > ADV_UTIL_WARN_PCT * 100:
        alerts.append(f"LIQUIDITY ALERT: max ADV utilization {adv:.2f}% "
                      f"(threshold {ADV_UTIL_WARN_PCT*100:.1f}%) "
                      f"-- consider raising volume floor")

    conc = basket.get("alpha_concentration_1")
    if conc and conc > CONCENTRATION_WARN:
        top = basket.get("top_contributor", "unknown")
        alerts.append(f"CONCENTRATION ALERT: {conc*100:.1f}% of return from {top} "
                      f"(threshold {CONCENTRATION_WARN*100:.1f}%) "
                      f"-- consider expanding position count")

    r1 = basket.get("rank_1_3_avg_ret")
    r2 = basket.get("rank_7_10_avg_ret")
    if r1 and r2 and r1 > 0:
        quality_ratio = r2 / r1
        if quality_ratio < RANK_QUALITY_WARN:
            alerts.append(f"SIGNAL QUALITY ALERT: rank 7-10 returns {r2:.2f}% "
                          f"vs rank 1-3 {r1:.2f}% "
                          f"(ratio {quality_ratio:.2f} < {RANK_QUALITY_WARN}) "
                          f"-- consider tightening position count")

    if alerts:
        log.info("\n=== SCALING ALERTS ===")
        for alert in alerts:
            log.warning(f"  {alert}")
        log.info("======================\n")
    else:
        log.info("Scaling metrics: no alerts -- system operating within normal parameters")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("EXECUTION TRACKER")
    log.info("=" * 60)

    # Determine week string
    today      = datetime.date.today()
    days_back  = (today.weekday() - 4) % 7
    last_fri   = today - datetime.timedelta(days=days_back if days_back > 0 else 7)
    week_str   = last_fri.strftime("%Y-%m-%d")
    week_start = (last_fri - datetime.timedelta(days=7)).isoformat()
    week_end   = last_fri.isoformat()

    log.info(f"Processing week: {week_str}")

    # Load existing execution log
    if EXEC_LOG.exists():
        exec_df = pd.read_csv(EXEC_LOG, low_memory=False)
        # Idempotency check
        if "week_of" in exec_df.columns and (exec_df["week_of"].astype(str) == week_str).any():
            log.info(f"Execution log already has week {week_str} -- skipping")
            return
    else:
        exec_df = pd.DataFrame()

    # Load supporting data
    state     = json.loads(TRADE_STATE.read_text()) if TRADE_STATE.exists() else {}
    perf_df   = pd.read_csv(PERF_LOG, low_memory=False) if PERF_LOG.exists() else pd.DataFrame()
    scores_df = pd.read_csv(SCORES_CSV) if SCORES_CSV.exists() else pd.DataFrame()

    # Get Alpaca fill data
    fills = get_alpaca_fills(week_start, week_end)

    # Compute per-symbol metrics
    rows = compute_execution_metrics(week_str, fills, state, perf_df, scores_df)

    if not rows:
        log.info("No execution data available for this week")
        return

    # Compute basket-level metrics
    basket = compute_basket_metrics(rows, week_str)

    # Log results
    log.info(f"\nExecution metrics for week {week_str}:")
    log.info(f"  Positions tracked: {len(rows)}")
    if basket:
        log.info(f"  Basket avg return:      {basket.get('basket_avg_return', 0)*100:+.2f}%")
        log.info(f"  Alpha concentration:    top1={basket.get('alpha_concentration_1', 0)*100:.1f}%  "
                 f"top2={basket.get('alpha_concentration_2', 0)*100:.1f}%")
        log.info(f"  Top contributor:        {basket.get('top_contributor')} "
                 f"+{basket.get('top_contributor_ret', 0)*100:.1f}%")
        log.info(f"  Signal quality by rank:")
        for bucket, key in [
            ("  Rank 1-3",   "rank_1_3_avg_ret"),
            ("  Rank 4-6",   "rank_4_6_avg_ret"),
            ("  Rank 7-10",  "rank_7_10_avg_ret"),
            ("  Rank 11-15", "rank_11_15_avg_ret"),
        ]:
            val = basket.get(key)
            if val is not None:
                log.info(f"    {bucket}: {val*100:+.2f}%")
        slip = basket.get("avg_entry_slippage_pct")
        adv  = basket.get("avg_adv_utilization_pct")
        if slip is not None:
            log.info(f"  Avg entry slippage:     {slip:.3f}%")
        if adv is not None:
            log.info(f"  Avg ADV utilization:    {adv:.3f}%")

    # Log scaling alerts
    log_scaling_alerts(basket)

    # Save per-symbol rows
    new_exec_df = pd.DataFrame(rows)
    combined = pd.concat([exec_df, new_exec_df], ignore_index=True) \
        if not exec_df.empty else new_exec_df
    combined.to_csv(EXEC_LOG, index=False)
    log.info(f"Execution log -> {EXEC_LOG} ({len(combined)} total rows)")

    # Save basket summary as a separate weekly summary file
    basket_log = DATA_DIR / "basket_metrics_log.csv"
    basket_df_existing = pd.read_csv(basket_log, low_memory=False) \
        if basket_log.exists() else pd.DataFrame()
    basket_new = pd.DataFrame([basket])
    basket_combined = pd.concat([basket_df_existing, basket_new], ignore_index=True) \
        if not basket_df_existing.empty else basket_new
    basket_combined.to_csv(basket_log, index=False)
    log.info(f"Basket metrics log -> {basket_log} ({len(basket_combined)} weeks)")


if __name__ == "__main__":
    run()
