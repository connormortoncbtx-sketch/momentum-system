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
#
# ──────────────────────────────────────────────────────────────────────
# 2026-05-05 PATCH NOTES — state-vs-Alpaca reconciliation
#
# Two bugs that compounded to produce continuous error notifications:
#
# BUG A: ensure_phase1_stop's "adopt existing stop" reconciliation only
#        looked for STOP and STOP_LIMIT order types, missing TRAILING_STOP.
#        When a position had a live trail (e.g., from a Phase 2 upgrade
#        whose state save was lost due to concurrent runner overwrite),
#        the function would falsely conclude "no stop exists" and try to
#        place a hard stop. Alpaca rejected with "insufficient qty"
#        because the trail held all shares.
#
# BUG B: When a Phase 2 upgrade reaches Step 3 (trail placed) but is then
#        killed before Step 4 (state save), or has its phase=2 state save
#        overwritten by a concurrent monitor instance running with stale
#        in-memory state, the next poll sees phase=1 + activation_price
#        crossed and re-attempts the upgrade. Step 1 (cancel) is a no-op
#        on already-cancelled order. Step 3 (trail) fails because the
#        first attempt's trail is still live. Loop forever.
#
# FIX:
#   1. ensure_phase1_stop now recognizes ALL stop-style orders (STOP,
#      STOP_LIMIT, TRAILING_STOP) when reconciling. If a live trail is
#      found, the function reconciles state to phase=2 instead of trying
#      to place a hard stop.
#
#   2. upgrade_to_phase2 now checks for an existing trail BEFORE
#      attempting any state changes. If a trail already exists for the
#      symbol (because a prior upgrade succeeded but state was lost),
#      adopt it and mark phase=2 -- don't re-run the upgrade flow.
#
#   3. Step 1 (cancel) now treats "no hard stops to cancel" as a NORMAL
#      condition (not an error) since we already accept that prior
#      attempts may have cancelled it. Crucially, we do NOT proceed to
#      submit a new trail unless we've verified no trail already exists.
# ──────────────────────────────────────────────────────────────────────

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


def reconcile_state_with_broker(client, state: dict,
                                 live_symbols: set = None,
                                 log_orphans: bool = True) -> dict:
    """Detect and clean divergence between alpaca_state.json and Alpaca's live positions.

    Removes state entries for symbols no longer held in Alpaca ("ghosts" --
    state has them, broker doesn't). Logs but does NOT remove orphans (broker
    has them, state doesn't) since auto-adopting them would mask manual
    interventions or upstream bugs.

    Mutates state in place. Does NOT call save_state() -- caller is responsible
    for persisting.

    Returns: {"ghosts_removed": [...], "live_orphans": [...]}

    Monitor-specific kwargs (vs the alpaca_trader twin):
      live_symbols: pre-fetched set of live symbols. The monitor already
        fetches live_positions every poll, so passing the keys avoids
        a redundant API call inside this function.
      log_orphans: if False, the function returns orphan info but does not
        log a warning. The monitor sets this to False because the same
        orphan would otherwise warn every 60 seconds for the entire
        session; the monitor handles its own session-level dedupe.

    See alpaca_trader.reconcile_state_with_broker for the full rationale on
    when and why state and broker reality diverge.
    """
    result = {"ghosts_removed": [], "live_orphans": []}

    if not state.get("positions"):
        return result

    if live_symbols is None:
        try:
            live_symbols = {p.symbol for p in client.get_all_positions()}
        except Exception as e:
            log.warning(f"  reconcile: get_all_positions failed: {e} -- skipping")
            return result

    state_symbols = set(state["positions"].keys())
    ghosts = sorted(state_symbols - live_symbols)
    live_orphans = sorted(live_symbols - state_symbols)

    for sym in ghosts:
        log.info(f"  reconcile: removing state ghost {sym} (not in Alpaca)")
        state["positions"].pop(sym, None)

    if ghosts:
        log.info(f"  reconcile: cleaned {len(ghosts)} state ghost(s): {ghosts}")

    if live_orphans and log_orphans:
        log.warning(f"  reconcile: {len(live_orphans)} live orphan(s) "
                    f"in Alpaca without state metadata: {live_orphans} -- "
                    f"manual investigation required")

    result["ghosts_removed"] = ghosts
    result["live_orphans"] = live_orphans
    return result


# ── ORDER HELPERS ─────────────────────────────────────────────────────────────

def find_open_stop_orders(client, symbol: str) -> dict:
    """
    List open SELL stop-style orders for a symbol, classified by type.

    Returns a dict with keys:
      - 'hard':     list of orders with type STOP or STOP_LIMIT
      - 'trailing': list of orders with type TRAILING_STOP

    Used by both ensure_phase1_stop (reconciliation) and upgrade_to_phase2
    (pre-flight check).
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus

    result = {"hard": [], "trailing": []}
    try:
        open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    except Exception as e:
        log.warning(f"  {symbol}: list_orders failed: {e}")
        return result

    for order in open_orders:
        if order.symbol != symbol or order.side != OrderSide.SELL:
            continue
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            result["hard"].append(order)
        elif order.order_type == OrderType.TRAILING_STOP:
            result["trailing"].append(order)

    return result


# ── PHASE 1 STOP MANAGEMENT ───────────────────────────────────────────────────

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

    2026-05-05 update: reconciliation now recognizes TRAILING_STOP orders. If a
    trail is found for the symbol, state is reconciled to phase=2 instead of
    trying to place a duplicate hard stop. Fixes the scenario where a Phase 2
    upgrade succeeded but state got rolled back (concurrent runner overwrite,
    state-save race, etc.).

    Returns True if any state-changing action was taken this call, False otherwise.
    """
    from alpaca.trading.requests import StopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

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

    # ── RECONCILIATION: check Alpaca for any existing stop coverage ──────
    # If a STOP/STOP_LIMIT exists, adopt it as our hard stop.
    # If a TRAILING_STOP exists, the position is actually phase=2 and our
    # state is stale -- reconcile to phase=2.
    existing_stops = find_open_stop_orders(client, symbol)

    if existing_stops["trailing"]:
        # State says phase=1 but Alpaca has a live trail. Most likely cause:
        # a prior Phase 2 upgrade succeeded but the state save was lost
        # (runner kill mid-flow, concurrent runner overwrite, state file
        # corruption). Reconcile state to match reality rather than trying
        # to place a duplicate hard stop on shares the trail is reserving.
        trail = existing_stops["trailing"][0]
        log.warning(f"  {symbol}: STATE RECONCILIATION -- found live trail order "
                    f"{trail.id}, updating state from phase=1 to phase=2")

        # Get current position info for state fields. If position lookup
        # fails, still reconcile (the trail proves position exists) but
        # leave qty/price fields untouched.
        try:
            pos = client.get_open_position(symbol)
            total_qty = int(float(pos.qty))
            current_price = float(pos.current_price)
        except Exception as e:
            log.warning(f"  {symbol}: get_position failed during reconciliation: {e} "
                        f"-- proceeding with state update only")
            total_qty = pos_state.get("shares", int(float(getattr(trail, "qty", 0)) or 0))
            current_price = pos_state.get("entry_price_actual",
                                          pos_state.get("entry_price_est", 0))

        state["positions"][symbol]["phase"]                = 2
        state["positions"][symbol]["trail_order_id"]       = str(trail.id)
        state["positions"][symbol]["trail_qty"]            = total_qty
        state["positions"][symbol]["hard_stop_order_id"]   = None
        state["positions"][symbol]["hard_stop_price"]      = None
        state["positions"][symbol]["upgrade_in_progress"]  = False
        state["positions"][symbol]["reconciled_at"]        = datetime.datetime.now().isoformat()
        # Note: we don't claim to know phase2_activated_at price post-hoc; leave
        # whatever was there (or None) as-is.
        state["positions"][symbol].pop("upgrade_started_at", None)

        log_event("alpaca_monitor", LogStatus.SUCCESS,
                  f"State reconciled to phase=2 for {symbol} (live trail discovered)",
                  metrics={"symbol": symbol, "trail_order_id": str(trail.id)})
        # Don't notify -- this is housekeeping. The original Phase 2 activation
        # already notified when it succeeded; spamming a second notification
        # just because we noticed the state mismatch isn't useful.
        return True

    if existing_stops["hard"]:
        # Hard stop exists in Alpaca but not tracked in state. Adopt it.
        order = existing_stops["hard"][0]
        log.info(f"  {symbol}: found existing hard stop {order.id} -- adopting")
        state["positions"][symbol]["hard_stop_order_id"] = str(order.id)
        if state["positions"][symbol].get("upgrade_in_progress"):
            state["positions"][symbol]["upgrade_in_progress"] = False
            log.info(f"  {symbol}: upgrade_in_progress flag cleared (existing stop adopted)")
        return True

    # ── NO EXISTING COVERAGE: PLACE A FRESH HARD STOP ────────────────────
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


# ── PHASE 2 UPGRADE ───────────────────────────────────────────────────────────

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

    2026-05-05 redesign:

    The earlier design (cancel hard stop -> save state -> submit trail ->
    save phase=2 -> partial sell) had two failure modes that compounded
    to produce continuous error notifications:

    1. If the runner was killed between trail-submission and phase=2 save,
       state remained at phase=1 with no hard stop. Next poll would
       attempt the upgrade fresh, re-cancel the (already-cancelled) hard
       stop, then fail to place a new trail because the original trail
       was still live holding all shares.

    2. Concurrent monitor instances (despite the workflow concurrency
       fix) could read state, perform an upgrade, then have their phase=2
       save overwritten by another instance that started before the save
       but finished after. Same end state as failure mode 1.

    The redesign addresses these by:

    - Pre-flight check: BEFORE any state changes, look for an existing
      trail order in Alpaca. If one exists, adopt it and mark phase=2.
      This makes the upgrade idempotent against repeated calls.

    - Partial sell now happens BEFORE trail placement. The trail is then
      placed on the post-partial-sell quantity. This avoids the "trail
      reserves all shares -> partial sell can't execute" deadlock that
      affected the prior design (which placed the trail on full quantity
      first, leaving no shares free for the partial sell).

    - Each step's failure is handled distinctly:
        Step 0 (pre-flight) failure: log only, proceed cautiously
        Step 1 (cancel) failure: abort, hard stop still covers
        Step 2 (partial sell) failure: continue with full-position trail
        Step 3 (trail submission) failure: state recovery handles next poll
    """
    from alpaca.trading.requests import (
        MarketOrderRequest, TrailingStopOrderRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce

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

    # ── STEP 0: PRE-FLIGHT CHECK FOR EXISTING TRAIL ──────────────────────
    # If a trail already exists for this symbol, a prior upgrade attempt
    # already succeeded (its state save was lost). Adopt it and mark
    # phase=2 -- do not run the upgrade flow.
    existing = find_open_stop_orders(client, symbol)
    if existing["trailing"]:
        trail = existing["trailing"][0]
        log.warning(f"  {symbol}: pre-flight found existing trail {trail.id} -- "
                    f"adopting and marking phase=2 (no upgrade flow run)")
        state["positions"][symbol]["phase"]                = 2
        state["positions"][symbol]["trail_order_id"]       = str(trail.id)
        state["positions"][symbol]["trail_qty"]            = total_qty
        state["positions"][symbol]["hard_stop_order_id"]   = None
        state["positions"][symbol]["hard_stop_price"]      = None
        state["positions"][symbol]["upgrade_in_progress"]  = False
        state["positions"][symbol]["reconciled_at"]        = datetime.datetime.now().isoformat()
        state["positions"][symbol].pop("upgrade_started_at", None)
        save_state(state)
        log_event("alpaca_monitor", LogStatus.SUCCESS,
                  f"Adopted existing trail and marked phase=2 for {symbol}",
                  metrics={"symbol": symbol, "trail_order_id": str(trail.id)})
        return True

    # ── STEP 1: CANCEL EXISTING HARD STOP(S) ─────────────────────────────
    # Alpaca won't let us place a new SELL covering shares that are already
    # reserved by another SELL. So we must release the reservation first.
    cancelled_ids = []
    for order in existing["hard"]:
        try:
            client.cancel_order_by_id(order.id)
            cancelled_ids.append(str(order.id))
        except Exception as cancel_err:
            log.error(f"  {symbol}: failed to cancel hard stop {order.id}: {cancel_err}")
            # If we can't cancel, we can't upgrade. Return now without changing
            # state -- the existing hard stop remains in place and the position
            # is still covered. Next poll will retry the upgrade.
            return False

    if cancelled_ids:
        log.info(f"  Step 1: Cancelled {len(cancelled_ids)} hard stop(s) for "
                 f"{symbol}: {cancelled_ids}")
    else:
        log.info(f"  Step 1: No hard stops to cancel for {symbol} (already cleared)")

    # ── STEP 2: MARK UPGRADE IN PROGRESS (RECOVERY ANCHOR) ───────────────
    # The flag tells the next monitor poll's ensure_phase1_stop that this
    # position is mid-upgrade and a fresh hard stop should be placed if
    # the trail submission fails below. The reconciliation logic in
    # ensure_phase1_stop will adopt any trail that DOES get placed
    # successfully, even without this flag, but the flag remains useful
    # as diagnostic state.
    state["positions"][symbol]["hard_stop_order_id"]   = None
    state["positions"][symbol]["upgrade_in_progress"]  = True
    state["positions"][symbol]["upgrade_started_at"]   = datetime.datetime.now().isoformat()
    save_state(state)
    log.info(f"  Step 2: State marked upgrade_in_progress=True and saved")

    # ── STEP 3: PARTIAL SELL FIRST (BEFORE TRAIL) ────────────────────────
    # Submitting partial sell first means:
    #  - Partial sell uses concrete shares (the sell_qty it needs)
    #  - Trail then places against (total_qty - sell_qty) -- the remainder
    #  - No deadlock: the trail isn't holding all shares when the partial
    #    sell tries to execute
    #
    # If partial sell fails, we still proceed to trail placement on the
    # FULL quantity, since that's better than no coverage. Partial sell is
    # variance reduction; trail placement is core position protection.
    sell_pct = compute_partial_sell_pct(alpha_score, weekly_vol)
    sell_qty = int(total_qty * sell_pct)
    partial_order_id = None
    partial_succeeded = False

    if sell_qty < 1 or (total_qty - sell_qty) < 1:
        log.info(f"  Step 3: Partial sell skipped -- split {total_qty}×{sell_pct:.2f} "
                 f"produces degenerate qty ({sell_qty}/{total_qty - sell_qty})")
        trail_qty = total_qty
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
            partial_succeeded = True
            log.info(f"  Step 3: Partial sell submitted: {symbol} {sell_qty} shares @ market "
                     f"order_id={partial_order.id} "
                     f"(alpha={alpha_score:.3f} vol={weekly_vol:.3f} pct={sell_pct*100:.1f}%)")
            # Wait briefly for partial sell to fill so the trail places against
            # the post-sale quantity. Most market orders fill in <1 second; we
            # wait up to 5 seconds to be safe. If still pending after 5s, place
            # the trail on the full quantity anyway -- the trail's qty will be
            # auto-reduced by Alpaca's risk engine when the partial sell fills.
            wait_start = time.time()
            while time.time() - wait_start < 5.0:
                try:
                    refreshed = client.get_order_by_id(partial_order.id)
                    if str(refreshed.status).lower() in ("filled", "partially_filled"):
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            # Compute trail quantity. If partial sell filled, trail covers
            # the remainder. If still pending, trail covers full qty (Alpaca
            # will reconcile when partial sell fills).
            try:
                refreshed_pos = client.get_open_position(symbol)
                trail_qty = int(float(refreshed_pos.qty))
            except Exception:
                trail_qty = total_qty - sell_qty
        except Exception as sell_err:
            log.warning(f"  Step 3: Partial sell failed (proceeding with full-qty trail): "
                        f"{sell_err}")
            trail_qty = total_qty

    # ── STEP 4: PLACE TRAILING STOP ──────────────────────────────────────
    if trail_qty < 1:
        log.error(f"  Step 4: trail_qty is {trail_qty} -- cannot place trail. "
                  f"Investigate: total_qty={total_qty}, sell_qty={sell_qty}")
        return False

    try:
        trail_order = client.submit_order(
            order_data=TrailingStopOrderRequest(
                symbol        = symbol,
                qty           = trail_qty,
                side          = OrderSide.SELL,
                time_in_force = TimeInForce.GTC,
                trail_percent = str(trail_pct),
            )
        )
        log.info(f"  Step 4: Trailing stop placed: {symbol} {trail_qty} shares "
                 f"trail={trail_pct}% order_id={trail_order.id}")
    except Exception as trail_err:
        # Trail submission failed. Position is now uncovered (or at least
        # not optimally covered -- the partial sell may have already executed).
        # ensure_phase1_stop on next poll will detect phase=1 + no
        # hard_stop_order_id and place a fresh hard stop for whatever
        # quantity is currently held.
        log.error(f"  Step 4 FAILED for {symbol}: {trail_err}")
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

    # ── STEP 5: MARK PHASE 2 SUCCESS ─────────────────────────────────────
    state["positions"][symbol]["phase"]                = 2
    state["positions"][symbol]["trail_order_id"]       = str(trail_order.id)
    state["positions"][symbol]["phase2_activated_at"]  = current_price
    state["positions"][symbol]["trail_qty"]            = trail_qty
    state["positions"][symbol]["partial_order_id"]     = partial_order_id
    state["positions"][symbol]["partial_sell_qty"]     = sell_qty if partial_succeeded else 0
    state["positions"][symbol]["partial_sell_pct_actual"] = (
        round(sell_qty / total_qty, 4) if (partial_succeeded and total_qty > 0) else 0
    )
    state["positions"][symbol]["upgrade_in_progress"]  = False
    state["positions"][symbol].pop("upgrade_started_at", None)
    save_state(state)
    log.info(f"  Step 5: State marked phase=2, upgrade_in_progress=False, saved")

    entry = pos_state.get("entry_price_est") or pos_state.get("entry_price_actual", 0)
    gain_pct = ((current_price / entry) - 1) * 100 if entry else 0
    log_event("alpaca_monitor", LogStatus.SUCCESS,
              f"Phase 2 activated: {symbol} @ ${current_price:.2f} ({gain_pct:+.1f}%)",
              metrics={
                  "symbol":              symbol,
                  "current_price":       round(current_price, 2),
                  "gain_pct":            round(gain_pct, 2),
                  "partial_sell_qty":    sell_qty if partial_succeeded else 0,
                  "trail_qty":           trail_qty,
                  "partial_sell_pct":    round(sell_pct * 100, 1) if partial_succeeded else 0,
                  "trail_pct":           trail_pct,
              })
    notify_success("alpaca_monitor",
                   f"PHASE 2 ACTIVATED: {symbol} @ ${current_price:.2f} "
                   f"({gain_pct:+.1f}%)\n"
                   f"Trailing stop on {trail_qty} shares @ {trail_pct}%"
                   + (f"\nPartial sell: {sell_qty} shares" if partial_succeeded else ""))
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
    warned_orphans: set = set()  # session-level dedupe for live-orphan warnings

    while datetime.datetime.now() < end_time:
        poll_count += 1
        try:
            state          = load_state()
            live_positions = {p.symbol: p for p in client.get_all_positions()}

            # Reconcile state with broker BEFORE early-return checks. Mid-session
            # stop fires (hard stops, trailing stops) leave ghost entries in
            # state that aren't cleaned until next Friday's exit. Doing this
            # every poll keeps state aligned with broker reality at minute
            # granularity rather than weekly.
            recon = reconcile_state_with_broker(
                client, state,
                live_symbols=set(live_positions.keys()),
                log_orphans=False,
            )
            if recon["ghosts_removed"]:
                save_state(state)

            new_orphans = set(recon["live_orphans"]) - warned_orphans
            if new_orphans:
                log.warning(f"  reconcile: new live orphan(s) in Alpaca without "
                            f"state metadata: {sorted(new_orphans)} -- manual "
                            f"investigation required")
                warned_orphans.update(new_orphans)

            tracked = state.get("positions", {})

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
