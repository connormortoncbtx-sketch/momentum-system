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

    if phase == 1 and current_price >= activation_price:
        log.info(f"PHASE 2 TRIGGERED: {symbol} @ ${current_price:.2f} "
                 f"(entry=${entry:.2f} activation=${activation_price:.2f} "
                 f"+{activation_pct:.1f}%)")
        return upgrade_to_phase2(api, symbol, current_price, trail_pct, state)

    return False


def upgrade_to_phase2(api, symbol: str, current_price: float,
                      trail_pct: float, state: dict) -> bool:
    try:
        # Cancel existing hard stop orders
        orders = api.list_orders(status="open")
        for order in orders:
            if (order.symbol == symbol and
                    order.order_type in ("stop", "stop_limit") and
                    order.side == "sell"):
                api.cancel_order(order.id)
                log.info(f"  Cancelled hard stop {order.id} for {symbol}")

        # Get current position qty
        try:
            pos = api.get_position(symbol)
            qty = int(float(pos.qty))
        except Exception:
            log.error(f"  Could not get position for {symbol} -- skipping")
            return False

        # Submit trailing stop
        order = api.submit_order(
            symbol        = symbol,
            qty           = qty,
            side          = "sell",
            type          = "trailing_stop",
            time_in_force = "gtc",
            trail_percent = str(trail_pct),
        )
        log.info(f"  Trailing stop placed: {symbol} {qty} shares "
                 f"trail={trail_pct}% order_id={order.id}")

        # Update state
        state["positions"][symbol]["phase"]               = 2
        state["positions"][symbol]["trail_order_id"]      = order.id
        state["positions"][symbol]["phase2_activated_at"] = current_price
        save_state(state)
        return True

    except Exception as e:
        log.error(f"Phase 2 upgrade failed for {symbol}: {e}")
        return False


# ── MAIN POLL LOOP ────────────────────────────────────────────────────────────

def run(duration_minutes: int = 180):
    log.info("=" * 60)
    log.info(f"ALPACA MONITOR starting -- duration={duration_minutes} min")
    log.info(f"Mode: {'PAPER' if PAPER_MODE else '*** LIVE ***'}")
    log.info("=" * 60)

    api        = get_alpaca()
    end_time   = datetime.datetime.now() + datetime.timedelta(minutes=duration_minutes)
    poll_count = 0
    upgrades   = 0

    while datetime.datetime.now() < end_time:
        poll_count += 1
        try:
            state = load_state()
            tracked = state.get("positions", {})

            if not tracked:
                log.debug("No tracked positions -- waiting")
                time.sleep(POLL_INTERVAL_SECS)
                continue

            # Get live positions from Alpaca
            live_positions = {p.symbol: p for p in api.list_positions()}

            if not live_positions:
                log.info("No open positions in Alpaca account")
                time.sleep(POLL_INTERVAL_SECS)
                continue

            # Fetch current prices via snapshots
            symbols   = list(live_positions.keys())
            snapshots = api.get_snapshots(symbols)

            changed = False
            for symbol in symbols:
                if symbol not in snapshots:
                    continue
                try:
                    current_price = float(snapshots[symbol].latest_trade.price)
                    upgraded = check_and_upgrade(api, symbol, current_price, state)
                    if upgraded:
                        upgrades += 1
                        changed = True
                    elif poll_count % 5 == 0:
                        # Log status every 5 polls
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
                    log.debug(f"  {symbol}: {e}")

            if changed:
                save_state(state)

        except Exception as e:
            log.error(f"Poll error: {e}")

        remaining = (end_time - datetime.datetime.now()).total_seconds() / 60
        log.debug(f"Poll {poll_count} complete -- {remaining:.0f} min remaining")
        time.sleep(POLL_INTERVAL_SECS)

    log.info(f"Monitor complete -- {poll_count} polls, {upgrades} Phase 2 upgrades")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=180,
                        help="Duration in minutes (default: 180)")
    args = parser.parse_args()
    run(duration_minutes=args.duration)
