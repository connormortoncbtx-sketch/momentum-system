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
#
# Migrated from `alpaca-trade-api` to `alpaca-py` (2026-04). The fills query
# now uses GetOrdersRequest with QueryOrderStatus.CLOSED + post-filter for
# OrderStatus.FILLED, since alpaca-py's QueryOrderStatus enum doesn't expose
# "filled" as a distinct query value (it's a subset of "closed").

import json
import logging
import os
import sys
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation.system_logger import log_event, LogStatus
from automation.notifier import notify_alert, notify_error

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
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus
        client = TradingClient(api_key=key, secret_key=secret, paper=paper)

        # Alpaca's `until` param is exclusive -- "before midnight UTC of that
        # date", which EXCLUDES same-day fills. Friday-close MOC sells fill at
        # 20:00-21:00 UTC on Friday, so passing week_end=last_fri.isoformat()
        # would drop every exit fill. Add one calendar day so the until-
        # boundary lands on Saturday 00:00 UTC, inclusively capturing
        # everything on Friday.
        from datetime import date as _date, timedelta as _td, datetime as _datetime, timezone as _tz
        try:
            until_date = _date.fromisoformat(week_end) + _td(days=1)
            until_dt   = _datetime.combine(until_date, _datetime.min.time(), tzinfo=_tz.utc)
            after_dt   = _datetime.combine(
                _date.fromisoformat(week_start), _datetime.min.time(), tzinfo=_tz.utc
            )
        except Exception:
            # Fall back to passing through whatever was given; alpaca-py also
            # accepts ISO strings via pydantic coercion in most versions.
            until_dt = week_end
            after_dt = week_start

        # alpaca-py only exposes OPEN/CLOSED/ALL as query statuses. "Filled"
        # is a subset of "closed" -- we filter post-hoc on order.status.
        all_orders = client.get_orders(
            filter=GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=after_dt,
                until=until_dt,
                limit=500,
            )
        )

        # Keep only filled orders. cancelled/expired/rejected etc. all ride
        # through CLOSED but have no fill data.
        orders = [o for o in all_orders if o.status == OrderStatus.FILLED]

        fills = {}
        for order in orders:
            sym  = order.symbol
            side = order.side
            fill_price = float(order.filled_avg_price or 0)
            fill_qty   = float(order.filled_qty or 0)

            if sym not in fills:
                fills[sym] = {}

            if side == OrderSide.BUY:
                fills[sym]["entry_fill"]  = fill_price
                fills[sym]["entry_qty"]   = fill_qty
                fills[sym]["entry_time"]  = str(order.filled_at or "")
            elif side == OrderSide.SELL:
                fills[sym]["exit_fill"]   = fill_price
                fills[sym]["exit_qty"]    = fill_qty
                fills[sym]["exit_time"]   = str(order.filled_at or "")

        log.info(f"Alpaca fills retrieved: {len(fills)} symbols "
                 f"({len(orders)} filled orders out of {len(all_orders)} closed)")
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

    all_symbols = set(fills.keys()) | set(positions.keys())

    for sym in all_symbols:
        pos_state  = positions.get(sym, {})
        fill_data  = fills.get(sym, {})

        intended_entry = pos_state.get("entry_price_est")
        intended_exit  = None  # Friday MOC -- use fri_close from perf log

        actual_entry = fill_data.get("entry_fill")
        actual_exit  = fill_data.get("exit_fill")

        entry_slippage_pct = None
        if intended_entry and actual_entry and intended_entry > 0:
            entry_slippage_pct = (actual_entry - intended_entry) / intended_entry

        exit_slippage_pct = None
        sym_perf = week_perf[week_perf["symbol"] == sym] if not week_perf.empty else pd.DataFrame()
        fri_close = float(sym_perf["fri_close"].iloc[0]) \
            if not sym_perf.empty and "fri_close" in sym_perf.columns else None

        if actual_exit and fri_close and fri_close > 0:
            exit_slippage_pct = (actual_exit - fri_close) / fri_close

        adv_utilization = None
        shares = pos_state.get("shares") or fill_data.get("entry_qty")
        if actual_entry and shares:
            position_dollar = float(shares) * float(actual_entry)
            score_row = scores_df[scores_df["symbol"] == sym] \
                if not scores_df.empty else pd.DataFrame()
            avg_vol = float(score_row["avg_vol_20d"].iloc[0]) \
                if not score_row.empty and "avg_vol_20d" in score_row.columns else None
            if avg_vol and avg_vol > 0 and actual_entry:
                adv_dollar = avg_vol * actual_entry
                adv_utilization = position_dollar / adv_dollar if adv_dollar > 0 else None

        forward_return = None
        phase2_sell_pct = pos_state.get("partial_sell_pct_actual")
        if not sym_perf.empty and "forward_return_1w" in sym_perf.columns:
            forward_return = float(sym_perf["forward_return_1w"].iloc[0])

        composite_rank = pos_state.get("composite_rank") or (
            int(score_row["composite_rank"].iloc[0])
            if not score_row.empty and "composite_rank" in score_row.columns else None
        ) if not scores_df.empty else None

        alpha_score = pos_state.get("alpha_score")
        weekly_vol  = pos_state.get("weekly_vol")

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

    total_return = df["forward_return_1w"].sum()
    top1_return  = df["forward_return_1w"].max()
    top2_return  = df.nlargest(2, "forward_return_1w")["forward_return_1w"].sum()

    concentration_top1 = (top1_return / total_return) if total_return != 0 else None
    concentration_top2 = (top2_return / total_return) if total_return != 0 else None

    top1_sym = df.loc[df["forward_return_1w"].idxmax(), "symbol"] \
        if not df.empty else None

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

    entry_slippage_cols = df["entry_slippage_pct"].dropna()
    exit_slippage_cols  = df["exit_slippage_pct"].dropna()
    avg_entry_slip = round(entry_slippage_cols.mean(), 4) \
        if not entry_slippage_cols.empty else None
    avg_exit_slip  = round(exit_slippage_cols.mean(), 4) \
        if not exit_slippage_cols.empty else None

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
            # H1 fix: r1 and r2 are stored as fractions (0.025 = 2.5%) because
            # they come from bucket_avg() which returns the mean of
            # forward_return_1w (already a fraction per collect_returns). Prior
            # format string {r1:.2f}% displayed the fraction verbatim, so an
            # actual 2.5% return was rendered as "0.02%". Multiply by 100 here
            # to match every other percent display in this module.
            alerts.append(f"SIGNAL QUALITY ALERT: rank 7-10 returns {r2*100:.2f}% "
                          f"vs rank 1-3 {r1*100:.2f}% "
                          f"(ratio {quality_ratio:.2f} < {RANK_QUALITY_WARN}) "
                          f"-- consider tightening position count")

    if alerts:
        log.info("\n=== SCALING ALERTS ===")
        for alert in alerts:
            log.warning(f"  {alert}")
        log.info("======================\n")
        log_event("execution_tracker", LogStatus.WARNING,
                  f"Scaling thresholds breached: {len(alerts)} alerts",
                  metrics={"alert_count": len(alerts)})
        notify_alert("execution_tracker",
                     "Scaling thresholds breached:\n\n" + "\n\n".join(alerts))
    else:
        log.info("Scaling metrics: no alerts -- system operating within normal parameters")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("EXECUTION TRACKER")
    log.info("=" * 60)
    log_event("execution_tracker", LogStatus.INFO, "Starting execution tracking")

    try:
        today      = datetime.date.today()
        days_back  = (today.weekday() - 4) % 7
        last_fri   = today - datetime.timedelta(days=days_back if days_back > 0 else 7)
        week_str   = last_fri.strftime("%Y-%m-%d")
        week_start = (last_fri - datetime.timedelta(days=7)).isoformat()
        week_end   = last_fri.isoformat()

        log.info(f"Processing week: {week_str}")

        if EXEC_LOG.exists():
            exec_df = pd.read_csv(EXEC_LOG, low_memory=False)
            if "week_of" in exec_df.columns and (exec_df["week_of"].astype(str) == week_str).any():
                log.info(f"Execution log already has week {week_str} -- skipping")
                log_event("execution_tracker", LogStatus.INFO,
                          f"Skipped: week {week_str} already in exec log")
                return
        else:
            exec_df = pd.DataFrame()

        state     = json.loads(TRADE_STATE.read_text()) if TRADE_STATE.exists() else {}
        perf_df   = pd.read_csv(PERF_LOG, low_memory=False) if PERF_LOG.exists() else pd.DataFrame()
        scores_df = pd.read_csv(SCORES_CSV) if SCORES_CSV.exists() else pd.DataFrame()

        fills = get_alpaca_fills(week_start, week_end)

        rows = compute_execution_metrics(week_str, fills, state, perf_df, scores_df)

        if not rows:
            log.info("No execution data available for this week")
            log_event("execution_tracker", LogStatus.INFO,
                      f"Skipped: no execution data for week {week_str}",
                      metrics={"fills_count": len(fills),
                               "state_positions": len(state.get("positions", {}))})
            return

        basket = compute_basket_metrics(rows, week_str)

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

        log_scaling_alerts(basket)

        new_exec_df = pd.DataFrame(rows)
        combined = pd.concat([exec_df, new_exec_df], ignore_index=True) \
            if not exec_df.empty else new_exec_df
        combined.to_csv(EXEC_LOG, index=False)
        log.info(f"Execution log -> {EXEC_LOG} ({len(combined)} total rows)")

        basket_log = DATA_DIR / "basket_metrics_log.csv"
        basket_df_existing = pd.read_csv(basket_log, low_memory=False) \
            if basket_log.exists() else pd.DataFrame()
        basket_new = pd.DataFrame([basket])
        basket_combined = pd.concat([basket_df_existing, basket_new], ignore_index=True) \
            if not basket_df_existing.empty else basket_new
        basket_combined.to_csv(basket_log, index=False)
        log.info(f"Basket metrics log -> {basket_log} ({len(basket_combined)} weeks)")

        log_event("execution_tracker", LogStatus.SUCCESS,
                  f"Tracked {len(rows)} positions for week {week_str}",
                  metrics={
                      "week_of":          week_str,
                      "positions":        int(len(rows)),
                      "fills_received":   len(fills),
                      "basket_avg_return": basket.get("basket_avg_return") if basket else None,
                      "top_contributor": basket.get("top_contributor") if basket else None,
                  })

    except Exception as e:
        log.error(f"execution_tracker crashed: {e}", exc_info=True)
        log_event("execution_tracker", LogStatus.ERROR,
                  "Unhandled exception during execution tracking",
                  errors=[str(e)])
        notify_error("execution_tracker", f"Unhandled error: {e}")
        raise


if __name__ == "__main__":
    run()
