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
#
# Migrated from `alpaca-trade-api` to `alpaca-py` (2026-04). Polling-only
# (no streaming); the migration is largely mechanical for this file.

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
DATA_DIR    = Path("data")
TRADE_STATE = DATA_DIR / "alpaca_state.json"
POLL_INTERVAL_SECS = 60


# ── CLIENT ────────────────────────────────────────────────────────────────────

def get_alpaca():
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        log.error("alpaca-py not installed")
        sys.exit(1)

    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        sys.exit(1)

    client = TradingClient(api_key=key, secret_key=secret, paper=PAPER_MODE)
    log.info(f"Alpaca connected [{'PAPER' if PAPER_MODE else 'LIVE'}]")
    return client


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

def ensure_phase1_stop(client, symbol: str, state: dict) -> bool:
    """
    Ensure a Phase 1 position has a hard stop order on file.

    Called each poll. On first poll after the Monday MOC entry fills, this submits
    the stop order using the ACTUAL filled average price (not the pre-close estimate),
    so the stop is aligned with the realized entry.

    Why this lives in the monitor rather than alpaca_trader's run_entry:
    - alpaca_trader submits buy orders at 2:30pm CT with time_in_force=CLS -- they fill
      at 3:00pm CT market close. An inline stop submitted at 2:30pm would risk firing
      before the buy fills, creating a short position. Waiting until after fill is safer.
    - Monitor's first poll runs 8:30am CT Tuesday. The fill has been confirmed overnight.
      Accepting this Mon-overnight-Tue-open gap as a known risk for paper mode; will
      reassess for live.

    2026-04-28 update: this function now ALSO acts as the recovery path for failed
    Phase 2 upgrades. If state has `upgrade_in_progress: True` (set when a previous
    upgrade attempt cancelled the hard stop but failed to place the trailing stop),
    this function is responsible for placing a fresh hard stop to restore coverage.
    The phase=1 + no hard_stop_order_id check naturally catches this case because
    the in-progress upgrade cleared hard_stop_order_id when it cancelled.

    Returns True if a stop was placed this call, False otherwise.
    """
    from alpaca.trading.requests import StopOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus

    pos_state = state.get("positions", {}).get(symbol, {})
    if not pos_state:
        return False

    if pos_state.get("phase", 1) != 1:
        return False

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
        open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        for order in open_orders:
            if (order.symbol == symbol and
                    order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and
                    order.side == OrderSide.SELL):
                log.info(f"  {symbol}: found existing stop {order.id} -- adopting")
                state["positions"][symbol]["hard_stop_order_id"] = str(order.id)
                # If we were mid-upgrade-recovery, clear the flag now that we have coverage
                if state["positions"][symbol].get("upgrade_in_progress"):
                    state["positions"][symbol]["upgrade_in_progress"] = False
                    log.info(f"  {symbol}: upgrade_in_progress flag cleared (existing stop adopted)")
                return True
    except Exception as e:
        log.warning(f"  {symbol}: list_orders check failed: {e} -- proceeding to submit")

    # Need the actual fill price. Use the live position's avg_entry_price -- this
    # reflects what Alpaca actually paid, including any slippage on the MOC fill.
    try:
        pos = client.get_open_position(symbol)
        actual_entry = float(pos.avg_entry_price)
        shares       = int(float(pos.qty))
    except Exception as e:
        log.warning(f"  {symbol}: get_open_position failed: {e} -- stop not placed this poll")
        return False

    if actual_entry <= 0 or shares < 1:
        return False

    stop_price = round(actual_entry * (1 - hard_stop_pct / 100), 2)

    try:
        order = client.submit_order(
            order_data=StopOrderRequest(
                symbol        = symbol,
                qty           = shares,
                side          = OrderSide.SELL,
                time_in_force = TimeInForce.GTC,
                stop_price    = stop_price,
            )
        )
        log.info(f"  HARD STOP {symbol}: {shares} shares @ stop=${stop_price:.2f} "
                 f"(entry=${actual_entry:.2f} -{hard_stop_pct:.1f}%)  order_id={order.id}")
        state["positions"][symbol]["hard_stop_order_id"] = str(order.id)
        state["positions"][symbol]["hard_stop_price"]    = stop_price
        state["positions"][symbol]["entry_price_actual"] = actual_entry
        # Clear upgrade-in-progress flag if we're recovering from a failed upgrade
        was_recovering = state["positions"][symbol].get("upgrade_in_progress")
        if was_recovering:
            state["positions"][symbol]["upgrade_in_progress"] = False
            log.info(f"  {symbol}: upgrade_in_progress flag cleared (fresh hard stop placed)")
            log_event("alpaca_monitor", LogStatus.SUCCESS,
                      f"Phase 2 upgrade recovery: hard stop restored on {symbol}",
                      metrics={"symbol": symbol, "stop_price": round(stop_price, 2)})
        else:
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
        notify_error("alpaca_monitor",
                     f"Failed to place hard stop for {symbol}: {e}\n"
                     f"Position is NAKED until stop is placed.")
        return False


def check_and_upgrade(client, symbol: str, current_price: float, state: dict) -> bool:
    pos_state = state.get("positions", {}).get(symbol)
    if not pos_state:
        return False

    phase          = pos_state.get("phase", 1)
    entry          = pos_state.get("entry_price_est", 0)
    activation_pct = pos_state.get("activation_pct")
    trail_pct      = pos_state.get("trail_pct")

    if not entry or not activation_pct or not trail_pct:
        return False

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
                return False
        except Exception:
            pass

    if phase == 1 and current_price >= activation_price:
        log.info(f"PHASE 2 TRIGGERED: {symbol} @ ${current_price:.2f} "
                 f"(entry=${entry:.2f} activation=${activation_price:.2f} "
                 f"+{activation_pct:.1f}%)")
        return upgrade_to_phase2(client, symbol, current_price, trail_pct, state)

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


def upgrade_to_phase2(client, symbol: str, current_price: float,
                      trail_pct: float, state: dict) -> bool:
    """Phase 2 upgrade: lock in gains with partial sell + trailing stop.

    Order-of-operations (revised 2026-04-28 after MXL upgrade failure):

    PRIOR DESIGN (broken): place trailing stop FIRST, then cancel hard stop.
    The intent was zero-coverage-gap, but Alpaca rejects this with
    "insufficient qty available" because the existing hard stop reserves
    all shares (held_for_orders == position size). You CANNOT have two
    overlapping SELL orders on the same shares -- Alpaca's risk engine
    blocks it. Every Phase 2 upgrade since this redesign was failing
    silently with this error, which we discovered when MXL hit Phase 2
    on Tuesday 2026-04-28.

    CURRENT DESIGN: cancel-then-place-then-recover, with explicit
    in-progress flagging:

      1. Cancel the existing hard stop. Position briefly has no Alpaca-
         side coverage (typically <2 seconds; bounded by HTTP RTT).
      2. Set upgrade_in_progress=True in state and persist immediately.
         If the runner is killed (concurrency cancellation, OOM, network
         drop) between steps 1 and 3, the next monitor poll's
         ensure_phase1_stop will see phase=1 + no hard_stop_order_id +
         upgrade_in_progress=True and place a fresh hard stop to restore
         coverage. This is the recovery path for partial failures.
      3. Submit the trailing stop on full position quantity.
      4. On success: mark phase=2, clear upgrade_in_progress, save.
      5. On failure: notification fires; ensure_phase1_stop on next poll
         restores hard stop. Position is uncovered for ~60 seconds (one
         poll interval) in the worst case.
      6. Submit partial sell (additive; allowed to fail without affecting
         core coverage since the trailing stop now covers the position).

    The brief coverage gap in step 1-3 is an unavoidable consequence of
    Alpaca's no-overlap rule. The recovery logic in ensure_phase1_stop
    bounds the worst case to one poll interval, which is the same exposure
    window the system already has between any two polls.
    """
    from alpaca.trading.requests import (
        MarketOrderRequest, TrailingStopOrderRequest, GetOrdersRequest
    )
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus

    pos_state   = state.get("positions", {}).get(symbol, {})
    alpha_score = pos_state.get("alpha_score", 0.75)
    weekly_vol  = pos_state.get("weekly_vol", 0.20)

    # Read live position once up front
    try:
        pos = client.get_open_position(symbol)
        total_qty = int(float(pos.qty))
    except Exception as e:
        log.error(f"  Could not get position for {symbol} -- skipping upgrade: {e}")
        return False

    if total_qty < 1:
        log.warning(f"  Position for {symbol} has 0 shares -- skipping upgrade")
        return False

    # ── STEP 1: CANCEL EXISTING HARD STOP(S) ─────────────────────────────
    # Alpaca won't let us place a new SELL covering shares that are already
    # reserved by another SELL. So we must release the reservation first.
    try:
        orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    except Exception as e:
        log.error(f"  {symbol}: could not list open orders to find hard stop: {e}")
        return False

    hard_stops = [
        o for o in orders
        if o.symbol == symbol
        and o.side == OrderSide.SELL
        and o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
    ]

    cancelled_ids = []
    for order in hard_stops:
        try:
            client.cancel_order_by_id(order.id)
            cancelled_ids.append(str(order.id))
        except Exception as cancel_err:
            log.error(f"  {symbol}: failed to cancel hard stop {order.id}: {cancel_err}")
            # If we can't cancel, we can't upgrade. Return now without changing
            # state -- the existing hard stop remains in place and the position
            # is still covered. Next poll will retry the upgrade.
            return False

    log.info(f"  Step 1: Cancelled {len(cancelled_ids)} hard stop(s) for {symbol}: "
             f"{cancelled_ids}")

    # ── STEP 2: MARK UPGRADE IN PROGRESS (RECOVERY ANCHOR) ───────────────
    # If we crash between cancellation and trail placement, the next monitor
    # poll's ensure_phase1_stop() will see phase=1 + no hard_stop_order_id
    # and place a fresh hard stop. The upgrade_in_progress flag is mostly
    # informational/diagnostic -- ensure_phase1_stop's logic doesn't strictly
    # require it -- but it gives us a clear signal in state that this
    # position is in a transient state.
    state["positions"][symbol]["hard_stop_order_id"]   = None
    state["positions"][symbol]["upgrade_in_progress"]  = True
    state["positions"][symbol]["upgrade_started_at"]   = datetime.datetime.now().isoformat()
    save_state(state)
    log.info(f"  Step 2: State marked upgrade_in_progress=True and saved")

    # ── STEP 3: PLACE TRAILING STOP ON FULL POSITION ─────────────────────
    try:
        trail_order = client.submit_order(
            order_data=TrailingStopOrderRequest(
                symbol        = symbol,
                qty           = total_qty,
                side          = OrderSide.SELL,
                time_in_force = TimeInForce.GTC,
                trail_percent = str(trail_pct),
            )
        )
        log.info(f"  Step 3: Trailing stop placed: {symbol} {total_qty} shares "
                 f"trail={trail_pct}% order_id={trail_order.id}")
    except Exception as trail_err:
        # Trail submission failed. Position is now uncovered.
        # ensure_phase1_stop on next poll will detect phase=1 + no hard_stop_order_id
        # and place a fresh hard stop. Notify user immediately.
        log.error(f"  Step 3 FAILED for {symbol}: {trail_err}")
        log.error(f"  Position is uncovered until next poll (≤{POLL_INTERVAL_SECS}s)")
        log_event("alpaca_monitor", LogStatus.ERROR,
                  f"Phase 2 trail submission failed: {symbol} (recovering on next poll)",
                  errors=[str(trail_err)])
        notify_error("alpaca_monitor",
                     f"PHASE 2 UPGRADE FAILED for {symbol}: trail submission failed.\n"
                     f"Hard stop already cancelled. Position uncovered until next "
                     f"poll (~{POLL_INTERVAL_SECS}s) which will auto-restore hard stop. "
                     f"Error: {trail_err}")
        # Do NOT mark phase=2; do NOT clear upgrade_in_progress. The flag tells
        # the next poll this position is mid-upgrade. ensure_phase1_stop will
        # detect missing hard_stop_order_id and place a fresh hard stop.
        return False

    # ── STEP 4: MARK PHASE 2 SUCCESS ─────────────────────────────────────
    state["positions"][symbol]["phase"]                = 2
    state["positions"][symbol]["trail_order_id"]       = str(trail_order.id)
    state["positions"][symbol]["phase2_activated_at"]  = current_price
    state["positions"][symbol]["trail_qty"]            = total_qty
    state["positions"][symbol]["partial_order_id"]     = None
    state["positions"][symbol]["partial_sell_qty"]     = 0
    state["positions"][symbol]["upgrade_in_progress"]  = False
    state["positions"][symbol].pop("upgrade_started_at", None)
    save_state(state)
    log.info(f"  Step 4: State marked phase=2, upgrade_in_progress=False, saved")

    # ── STEP 5: SUBMIT PARTIAL SELL (ADDITIVE, ALLOWED TO FAIL) ──────────
    # Trailing stop covers the full position, so the partial sell is purely
    # gain realization. If it fails, position is fully covered by the trail.
    sell_pct   = compute_partial_sell_pct(alpha_score, weekly_vol)
    sell_qty   = int(total_qty * sell_pct)
    partial_order_id = None

    if sell_qty < 1 or (total_qty - sell_qty) < 1:
        log.info(f"  Step 5: Partial sell skipped -- split {total_qty}×{sell_pct:.2f} "
                 f"produces degenerate qty ({sell_qty}/{total_qty - sell_qty})")
    else:
        try:
            partial_order = client.submit_order(
                order_data=MarketOrderRequest(
                    symbol        = symbol,
                    qty           = sell_qty,
                    side          = OrderSide.SELL,
                    time_in_force = TimeInForce.DAY,
                )
            )
            partial_order_id = str(partial_order.id)
            log.info(f"  Step 5: Partial sell: {symbol} {sell_qty} shares @ market "
                     f"(~${current_price:.2f})  order_id={partial_order.id}  "
                     f"(alpha={alpha_score:.3f} vol={weekly_vol:.3f} pct={sell_pct*100:.1f}%)")
            state["positions"][symbol]["partial_order_id"]     = partial_order_id
            state["positions"][symbol]["partial_sell_qty"]     = sell_qty
            state["positions"][symbol]["partial_sell_pct_actual"] = round(
                sell_qty / total_qty, 4) if total_qty > 0 else 0
            save_state(state)
        except Exception as sell_err:
            # Partial sell failure is non-fatal -- trail covers position.
            log.warning(f"  Step 5: Partial sell failed (non-fatal, trail still covers): {sell_err}")

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


# ── MAIN POLL LOOP ────────────────────────────────────────────────────────────

def run(duration_minutes: int = 180):
    log.info("=" * 60)
    log.info(f"ALPACA MONITOR starting -- duration={duration_minutes} min")
    log.info(f"Mode: {'PAPER' if PAPER_MODE else '*** LIVE ***'}")
    log.info("=" * 60)
    log_event("alpaca_monitor", LogStatus.INFO,
              f"Starting monitor (duration={duration_minutes}min, "
              f"mode={'PAPER' if PAPER_MODE else 'LIVE'})")

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

    client     = get_alpaca()
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

            live_positions = {p.symbol: p for p in client.get_all_positions()}

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
                    current_price = float(pos.current_price)

                    if ensure_phase1_stop(client, symbol, state):
                        changed = True
                        stops_placed += 1

                    upgraded = check_and_upgrade(client, symbol, current_price, state)
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
