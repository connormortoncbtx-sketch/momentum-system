# automation/alpaca_monitor.py
# Intraday position monitor -- checks Phase 2 activation and upgrades stops.
# Designed to run as a GitHub Actions job during market hours.
# Polls every 60 seconds for a specified duration then exits cleanly.
#
# Usage:
#   python automation/alpaca_monitor.py --duration 180  # run for 180 minutes
#
# Three overlapping jobs cover the full market session:
#   Job 1: 8:30 AM CT  -- runs 180 min -> ends 11:30 AM CT
#   Job 2: 11:15 AM CT -- runs 180 min -> ends 2:15 PM CT
#   Job 3: 2:00 PM CT  -- runs 65 min  -> ends 3:05 PM CT
#
# PAPER_MODE controlled by ALPACA_PAPER environment variable.
# Set ALPACA_PAPER=true for paper trading (default).

import argparse
import json
import logging
import os
import sys
import time
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation.system_logger import log_event, LogStatus
from automation.notifier import notify_alert, notify_error, notify_success

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

PAPER_MODE  = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
BASE_URL    = "https://paper-api.alpaca.markets" if PAPER_MODE else "https://api.alpaca.markets"
DATA_DIR    = Path("data")
TRADE_STATE = DATA_DIR / "alpaca_state.json"
POLL_INTERVAL_SECS = 60


# ── CLIENT ────────────────────────────────────────────────────────────────────

def get_alpaca():
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        log.error("alpaca-trade-api not installed")
        sys.exit(1)

    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        sys.exit(1)

    api = tradeapi.REST(key, secret, BASE_URL, api_version="v2")
    log.info(f"Alpaca connected [{'PAPER' if PAPER_MODE else 'LIVE'}]")
    return api


# ── STATE ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if TRADE_STATE.exists():
        with open(TRADE_STATE) as f:
            return json.load(f)
    return {"positions": {}}


def save_state(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRADE_STATE, "w") as f:
        json.dump(state, f, indent=2)


# ── PHASE 2 UPGRADE ───────────────────────────────────────────────────────────

def ensure_phase1_stop(api, symbol: str, state: dict) -> bool:
    """
    Ensure a Phase 1 position has a hard stop order on file.

    Called each poll. On first poll after the Monday MOC entry fills, this submits
    the stop order using the ACTUAL filled average price (not the pre-close estimate),
    so the stop is aligned with the realized entry.

    Why this lives in the monitor rather than alpaca_trader's run_entry:
    - alpaca_trader submits buy orders at 2:30pm CT with time_in_force="cls" -- they fill
      at 3:00pm CT market close. An inline stop submitted at 2:30pm would risk firing
      before the buy fills, creating a short position. Waiting until after fill is safer.
    - Monitor's first poll runs 8:30am CT Tuesday. The fill has been confirmed overnight.
      Accepting this Mon-overnight-Tue-open gap as a known risk for paper mode; will
      reassess for live.

    Returns True if a stop was placed this call, False otherwise.
    """
    pos_state = state.get("positions", {}).get(symbol, {})
    if not pos_state:
        return False

    # Only Phase 1 positions need a hard stop. Phase 2 uses the trailing stop instead.
    if pos_state.get("phase", 1) != 1:
        return False

    # Already placed? state records the order id
    if pos_state.get("hard_stop_order_id"):
        return False

    hard_stop_pct = pos_state.get("hard_stop_pct")
    if not hard_stop_pct or hard_stop_pct <= 0:
        log.warning(f"  {symbol}: no hard_stop_pct in state -- cannot place stop")
        return False

    # Check Alpaca for an existing stop order on this symbol. If one exists (e.g., placed
    # manually or by a previous run that didn't persist the id), adopt it rather than
    # creating a duplicate.
    try:
        open_orders = api.list_orders(status="open")
        for order in open_orders:
            if (order.symbol == symbol and
                    order.order_type in ("stop", "stop_limit") and
                    order.side == "sell"):
                log.info(f"  {symbol}: found existing stop {order.id} -- adopting")
                state["positions"][symbol]["hard_stop_order_id"] = order.id
                return True
    except Exception as e:
        log.warning(f"  {symbol}: list_orders check failed: {e} -- proceeding to submit")

    # Need the actual fill price. Use the live position's avg_entry_price -- this
    # reflects what Alpaca actually paid, including any slippage on the MOC fill.
    try:
        pos = api.get_position(symbol)
        actual_entry = float(pos.avg_entry_price)
        shares       = int(float(pos.qty))
    except Exception as e:
        log.warning(f"  {symbol}: get_position failed: {e} -- stop not placed this poll")
        return False

    if actual_entry <= 0 or shares < 1:
        return False

    stop_price = round(actual_entry * (1 - hard_stop_pct / 100), 2)

    try:
        order = api.submit_order(
            symbol        = symbol,
            qty           = shares,
            side          = "sell",
            type          = "stop",
            stop_price    = stop_price,
            time_in_force = "gtc",
        )
        log.info(f"  HARD STOP {symbol}: {shares} shares @ stop=${stop_price:.2f} "
                 f"(entry=${actual_entry:.2f} -{hard_stop_pct:.1f}%)  order_id={order.id}")
        state["positions"][symbol]["hard_stop_order_id"] = order.id
        state["positions"][symbol]["hard_stop_price"]    = stop_price
        state["positions"][symbol]["entry_price_actual"] = actual_entry
        log_event("alpaca_monitor", LogStatus.SUCCESS,
                  f"Hard stop placed: {symbol} @ ${stop_price:.2f}",
                  metrics={
                      "symbol":       symbol,
                      "shares":       shares,
                      "entry_price":  round(actual_entry, 2),
                      "stop_price":   round(stop_price, 2),
                      "stop_pct":     round(hard_stop_pct, 2),
                  })
        return True
    except Exception as e:
        log.error(f"  {symbol}: stop order submission failed: {e}")
        log_event("alpaca_monitor", LogStatus.ERROR,
                  f"Hard stop submission failed for {symbol}",
                  errors=[str(e)])
        # This is capital-at-risk. If a stop can't be placed, the position is
        # naked. Alert immediately.
        notify_error("alpaca_monitor",
                     f"Failed to place hard stop for {symbol}: {e}\n"
                     f"Position is NAKED until stop is placed.")
        return False


def check_and_upgrade(api, symbol: str, current_price: float, state: dict) -> bool:
    pos_state = state.get("positions", {}).get(symbol)
    if not pos_state:
        return False

    phase          = pos_state.get("phase", 1)
    entry          = pos_state.get("entry_price_est", 0)
    activation_pct = pos_state.get("activation_pct")
    trail_pct      = pos_state.get("trail_pct")

    if not entry or not activation_pct or not trail_pct:
        return False

    # Update high water mark
    hwm = pos_state.get("high_water_mark", entry)
    if current_price > hwm:
        state["positions"][symbol]["high_water_mark"] = current_price

    activation_price = entry * (1 + activation_pct / 100)

    # Same-day cooldown: Phase 2 cannot trigger on the calendar day the position
    # was entered. Monday MOC fills settle after close; any post-close monitor
    # run sees the entry price as "current price" and can mistakenly trigger
    # Phase 2 on slight fill-vs-last_price drift. Delaying eligibility until
    # Tuesday open ensures Phase 2 only fires on real post-entry price
    # movement, not fill-price variance.
    entry_date_str = state.get("entry_date")
    if entry_date_str:
        try:
            import datetime as _dt
            entry_date = _dt.date.fromisoformat(entry_date_str)
            if _dt.date.today() <= entry_date:
                # Today is the entry day -- Phase 2 not eligible yet
                return False
        except Exception:
            # Malformed entry_date -- proceed without cooldown guard rather
            # than refusing all upgrades
            pass

    if phase == 1 and current_price >= activation_price:
        log.info(f"PHASE 2 TRIGGERED: {symbol} @ ${current_price:.2f} "
                 f"(entry=${entry:.2f} activation=${activation_price:.2f} "
                 f"+{activation_pct:.1f}%)")
        return upgrade_to_phase2(api, symbol, current_price, trail_pct, state)

    return False


def compute_partial_sell_pct(alpha_score: float, weekly_vol: float) -> float:
    """
    Calculate what fraction of the position to sell at Phase 2 activation.
    Higher alpha score (model confidence) = sell less, let more run.
    Higher weekly volatility = sell more, lock in more gains.

    Formula:
      base = 0.75 - (alpha_score * 0.50)
      At alpha=0.50 -> sell 50%
      At alpha=0.75 -> sell 37.5%
      At alpha=0.90 -> sell 30%
      At alpha=1.00 -> sell 25%

    Vol adjustment:
      vol above 25% weekly adds to sell pct
      (weekly_vol - 0.25) * 0.50

    Result clamped to [0.25, 0.75]
    """
    alpha = max(0.0, min(1.0, float(alpha_score or 0.75)))
    base  = 0.75 - (alpha * 0.50)
    vol   = float(weekly_vol or 0.0)
    vol_adj = max(0.0, (vol - 0.25) * 0.50)
    pct = base + vol_adj
    return max(0.25, min(0.75, round(pct, 4)))


def upgrade_to_phase2(api, symbol: str, current_price: float,
                      trail_pct: float, state: dict) -> bool:
    """Phase 2 upgrade: lock in gains with partial sell + trailing stop.

    Order-of-operations redesign (2026-04-21):

    The 2026-04-20 incident surfaced two compounding failures:

    1. Coverage gap during upgrade: hard stop was cancelled FIRST, then
       trailing stop placed. If trailing stop submission failed, position
       had no stop coverage until manual intervention.

    2. Gap after partial-sell cancellation: partial market sell was placed
       for a portion of shares, trailing stop covered only the remainder.
       If user/system cancelled the partial sell, those shares were left
       un-stopped (hard stop already cancelled, trailing stop doesn't cover
       them).

    3. State persistence gap: state was saved at the END of upgrade. If any
       step failed with an exception, phase remained at 1 and the next
       monitor poll re-triggered Phase 2 -- which tried to cancel
       already-cancelled orders, place duplicate trailing stops, etc.
       This is why alpaca_monitor appeared to re-fire repeatedly on Monday.

    Redesigned order-of-operations:
      1. Place trailing stop FIRST (on full position quantity). Now every
         share has stop coverage via the trailing stop.
      2. Update state to phase=2 and save IMMEDIATELY. Even if later steps
         fail, the position won't be re-upgraded on next poll.
      3. Cancel the original hard stop(s). Position still covered by trail.
      4. Submit partial market sell. If this succeeds, it sells into the
         trailing stop (reducing trail qty by the sold amount at fill time).
         If this fails or gets cancelled, trailing stop still covers
         everything.

    Trailing stop always covers the FULL position in this design. Partial
    sell is an additive realization of gains, not a replacement for stop
    coverage. This makes every intermediate state safe.
    """
    try:
        pos_state   = state.get("positions", {}).get(symbol, {})
        alpha_score = pos_state.get("alpha_score", 0.75)
        weekly_vol  = pos_state.get("weekly_vol", 0.20)

        # Get current position qty
        try:
            pos = api.get_position(symbol)
            total_qty = int(float(pos.qty))
        except Exception:
            log.error(f"  Could not get position for {symbol} -- skipping")
            return False

        if total_qty < 1:
            log.warning(f"  Position for {symbol} has 0 shares -- skipping upgrade")
            return False

        # STEP 1: Place trailing stop on the FULL position.
        # Coverage exists before we touch anything else.
        trail_order = api.submit_order(
            symbol        = symbol,
            qty           = total_qty,
            side          = "sell",
            type          = "trailing_stop",
            time_in_force = "gtc",
            trail_percent = str(trail_pct),
        )
        log.info(f"  Step 1: Trailing stop placed: {symbol} {total_qty} shares "
                 f"trail={trail_pct}% order_id={trail_order.id}")

        # STEP 2: Mark state phase=2 and save IMMEDIATELY.
        # This prevents re-upgrade even if the remaining steps fail.
        state["positions"][symbol]["phase"]                = 2
        state["positions"][symbol]["trail_order_id"]       = trail_order.id
        state["positions"][symbol]["phase2_activated_at"]  = current_price
        state["positions"][symbol]["trail_qty"]            = total_qty
        state["positions"][symbol]["partial_order_id"]     = None
        state["positions"][symbol]["partial_sell_qty"]     = 0
        save_state(state)
        log.info(f"  Step 2: State marked phase=2 and saved")

        # STEP 3: Cancel old hard stops. Trailing stop already covers position.
        orders = api.list_orders(status="open")
        cancelled = []
        for order in orders:
            if (order.symbol == symbol and
                    order.order_type in ("stop", "stop_limit") and
                    order.side == "sell" and
                    order.id != trail_order.id):
                try:
                    api.cancel_order(order.id)
                    cancelled.append(order.id)
                except Exception as cancel_err:
                    log.warning(f"  Could not cancel stop {order.id}: {cancel_err}")
        if cancelled:
            log.info(f"  Step 3: Cancelled {len(cancelled)} old hard stop(s): {cancelled}")

        # STEP 4: Submit partial market sell (additive gain realization).
        # Trailing stop covers full position, so partial sell is optional --
        # if it fails, we still have complete stop coverage.
        sell_pct   = compute_partial_sell_pct(alpha_score, weekly_vol)
        sell_qty   = int(total_qty * sell_pct)

        # Guard: don't submit if split is degenerate
        if sell_qty < 1 or (total_qty - sell_qty) < 1:
            log.info(f"  Step 4: Partial sell skipped -- split {total_qty}×{sell_pct:.2f} "
                     f"produces degenerate qty ({sell_qty}/{total_qty - sell_qty})")
            partial_order_id = None
        else:
            try:
                partial_order = api.submit_order(
                    symbol        = symbol,
                    qty           = sell_qty,
                    side          = "sell",
                    type          = "market",
                    time_in_force = "day",
                )
                partial_order_id = partial_order.id
                log.info(f"  Step 4: Partial sell: {symbol} {sell_qty} shares @ market "
                         f"(~${current_price:.2f})  order_id={partial_order.id}  "
                         f"(alpha={alpha_score:.3f} vol={weekly_vol:.3f} pct={sell_pct*100:.1f}%)")
                state["positions"][symbol]["partial_order_id"]     = partial_order_id
                state["positions"][symbol]["partial_sell_qty"]     = sell_qty
                state["positions"][symbol]["partial_sell_pct_actual"] = round(
                    sell_qty / total_qty, 4) if total_qty > 0 else 0
                save_state(state)
            except Exception as sell_err:
                log.warning(f"  Step 4: Partial sell failed (non-fatal): {sell_err}")
                partial_order_id = None
                # Trailing stop already covers full position; partial sell
                # is gain-realization optimization, not safety-critical.

        # Phase 2 upgrade is the single most important intraday event -- send a
        # high-priority notification since it means the position hit the profit
        # threshold and we're locking in gains.
        entry = pos_state.get("entry_price_est") or pos_state.get("entry_price_actual", 0)
        gain_pct = ((current_price / entry) - 1) * 100 if entry else 0
        log_event("alpaca_monitor", LogStatus.SUCCESS,
                  f"Phase 2 activated: {symbol} @ ${current_price:.2f} ({gain_pct:+.1f}%)",
                  metrics={
                      "symbol":              symbol,
                      "current_price":       round(current_price, 2),
                      "gain_pct":            round(gain_pct, 2),
                      "partial_sell_qty":    sell_qty if partial_order_id else 0,
                      "trail_qty":           total_qty,
                      "partial_sell_pct":    round(sell_pct * 100, 1) if partial_order_id else 0,
                      "trail_pct":           trail_pct,
                  })
        notify_success("alpaca_monitor",
                       f"PHASE 2 ACTIVATED: {symbol} @ ${current_price:.2f} "
                       f"({gain_pct:+.1f}%)\n"
                       f"Trailing stop on {total_qty} shares @ {trail_pct}%"
                       + (f"\nPartial sell: {sell_qty} shares" if partial_order_id else ""))
        return True

    except Exception as e:
        log.error(f"Phase 2 upgrade failed for {symbol}: {e}")
        log_event("alpaca_monitor", LogStatus.ERROR,
                  f"Phase 2 upgrade failed: {symbol}",
                  errors=[str(e)])
        notify_error("alpaca_monitor",
                     f"PHASE 2 UPGRADE FAILED for {symbol}: {e}\n"
                     f"Position may now lack a stop. Investigate.")
        return False


# ── MAIN POLL LOOP ────────────────────────────────────────────────────────────

def run(duration_minutes: int = 180):
    log.info("=" * 60)
    log.info(f"ALPACA MONITOR starting -- duration={duration_minutes} min")
    log.info(f"Mode: {'PAPER' if PAPER_MODE else '*** LIVE ***'}")
    log.info("=" * 60)
    log_event("alpaca_monitor", LogStatus.INFO,
              f"Starting monitor (duration={duration_minutes}min, "
              f"mode={'PAPER' if PAPER_MODE else 'LIVE'})")

    # Holiday check -- no point monitoring on a closed market
    try:
        from automation.tz_utils import is_trading_day
        import datetime as dt
        today = dt.date.today()
        if not is_trading_day(today):
            log.info(f"Market holiday ({today}) -- monitor exiting cleanly")
            log_event("alpaca_monitor", LogStatus.INFO,
                      f"Skipped: market holiday {today}")
            return
    except Exception as e:
        log.debug(f"Holiday check unavailable: {e} -- proceeding")

    api        = get_alpaca()
    end_time   = datetime.datetime.now() + datetime.timedelta(minutes=duration_minutes)
    poll_count = 0
    upgrades   = 0
    stops_placed = 0
    poll_errors = 0

    while datetime.datetime.now() < end_time:
        poll_count += 1
        try:
            state   = load_state()
            tracked = state.get("positions", {})

            # Get live positions from Alpaca
            live_positions = {p.symbol: p for p in api.list_positions()}

            if not live_positions:
                log.info(f"Poll {poll_count}: No open positions in Alpaca account")
                time.sleep(POLL_INTERVAL_SECS)
                continue

            if not tracked:
                log.info(f"Poll {poll_count}: {len(live_positions)} Alpaca positions "
                         f"but no state file entries -- waiting for entry workflow")
                time.sleep(POLL_INTERVAL_SECS)
                continue

            changed = False
            for symbol, pos in live_positions.items():
                try:
                    # Use current_price from position object -- no extra API call needed
                    current_price = float(pos.current_price)

                    # Ensure Phase 1 positions have their hard stop placed. On first poll
                    # after entry, this submits the stop using the actual fill price. Called
                    # before check_and_upgrade because Phase 2 upgrade cancels the hard stop.
                    if ensure_phase1_stop(api, symbol, state):
                        changed = True
                        stops_placed += 1

                    upgraded = check_and_upgrade(api, symbol, current_price, state)
                    if upgraded:
                        upgrades += 1
                        changed = True
                    elif poll_count % 5 == 0:
                        pos_state = state.get("positions", {}).get(symbol, {})
                        phase     = pos_state.get("phase", 1)
                        entry     = pos_state.get("entry_price_est", 0)
                        ret_pct   = ((current_price / entry) - 1) * 100 if entry else 0
                        hwm       = pos_state.get("high_water_mark", entry)
                        log.info(f"  {symbol:<8} ${current_price:.2f}  "
                                 f"ret={ret_pct:+.1f}%  "
                                 f"hwm=${hwm:.2f}  "
                                 f"phase={phase}")
                except Exception as e:
                    log.warning(f"  {symbol}: price check error: {e}")
                    poll_errors += 1

            if changed:
                save_state(state)

            if poll_count % 5 == 0:
                remaining = (end_time - datetime.datetime.now()).total_seconds() / 60
                log.info(f"Poll {poll_count} -- {remaining:.0f} min remaining  "
                         f"{len(live_positions)} positions  {upgrades} upgrades")

        except Exception as e:
            log.error(f"Poll {poll_count} error: {e}")
            poll_errors += 1

        time.sleep(POLL_INTERVAL_SECS)

    log.info(f"Monitor complete -- {poll_count} polls, {upgrades} Phase 2 upgrades")

    # End-of-session summary. We don't notify on every monitor run (too noisy),
    # but do notify if anything material happened or if error rate was high.
    metrics = {
        "polls":        poll_count,
        "upgrades":     upgrades,
        "stops_placed": stops_placed,
        "poll_errors":  poll_errors,
    }
    if upgrades > 0 or stops_placed > 0:
        log_event("alpaca_monitor", LogStatus.SUCCESS,
                  f"Session: {upgrades} upgrades, {stops_placed} stops placed",
                  metrics=metrics)
    else:
        log_event("alpaca_monitor", LogStatus.INFO,
                  f"Session quiet: {poll_count} polls, no phase transitions",
                  metrics=metrics)

    # Alert if poll error rate is suspiciously high (>10% of polls failed)
    if poll_count > 10 and poll_errors > 0.10 * poll_count:
        notify_alert("alpaca_monitor",
                     f"High poll error rate: {poll_errors}/{poll_count} polls "
                     f"had errors. API issues or state file problems?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=180,
                        help="Duration in minutes (default: 180)")
    args = parser.parse_args()
    run(duration_minutes=args.duration)
