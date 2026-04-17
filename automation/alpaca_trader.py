# automation/alpaca_trader.py
# Autonomous trading execution layer for Momentum Alpha.
# Reads the weekly report scores, applies entry/exit logic,
# and places orders via the Alpaca API.
#
# Runs twice per week via GitHub Actions:
#   Monday  2:30 PM CT  -- entry orders (Monday close)
#   Friday  2:30 PM CT  -- exit orders (Friday close MOC)
#
# PAPER_MODE = True  uses Alpaca paper trading endpoint (default)
# PAPER_MODE = False uses Alpaca live trading endpoint (flip when ready)
#
# Required secrets in GitHub Actions:
#   ALPACA_API_KEY
#   ALPACA_SECRET_KEY
#
# Optional secrets:
#   ALPACA_WITHDRAWAL_ENABLED  -- set to "true" to enable weekly withdrawal
#   ALPACA_WITHDRAWAL_AMOUNT   -- flat dollar amount to withdraw (e.g. "5000")
#   ALPACA_WITHDRAWAL_PCT      -- % of weekly gains to withdraw (e.g. "0.50")
#   (if both set, flat amount takes precedence)

import json
import logging
import os
import sys
import datetime
import pandas as pd
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

PAPER_MODE = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL  = "https://api.alpaca.markets"
BASE_URL       = PAPER_BASE_URL if PAPER_MODE else LIVE_BASE_URL

DATA_DIR   = Path("data")
SCORES_CSV = DATA_DIR / "scores_final.csv"
REGIME_JSON = DATA_DIR / "regime.json"
TRADE_STATE = DATA_DIR / "alpaca_state.json"  # tracks open positions

# -- POSITION SIZING PARAMETERS (tune as capital scales) ----------------------

# Minimum dollar value per position
# Prevents trivially small positions as count expands
MIN_POSITION_SIZE = 500.0

# Maximum single position as % of total portfolio
# Prevents over-concentration in one name
MAX_POSITION_PCT = 0.15

# Minimum composite rank percentile to qualify for entry
# Names below this threshold are excluded regardless of position count
MIN_COMPOSITE_PERCENTILE = 0.70   # top 30% of universe

# Maximum position size as % of stock's average daily volume
# Prevents moving the market on entry/exit
# At current capital this won't bind -- matters at $500k+
MAX_ADV_PCT = 0.01   # 1% of average daily volume

# Minimum stock price -- hard gate, no sub-$1 stocks
MIN_PRICE = 1.00

# Minimum average daily volume -- ensures basic liquidity
MIN_AVG_VOLUME = 50000

# Circuit breaker -- if portfolio drops this % from week-open value, exit all
# Set to None to disable. Tune based on backtesting.
CIRCUIT_BREAKER_PCT = None   # e.g. 0.08 = exit if down 8% from Monday open

# -- WITHDRAWAL PARAMETERS ----------------------------------------------------
# Controlled by environment variables / GitHub secrets.
# ALPACA_WITHDRAWAL_ENABLED = "true" to activate
# ALPACA_WITHDRAWAL_AMOUNT  = flat dollar amount (e.g. "5000")
# ALPACA_WITHDRAWAL_PCT     = fraction of weekly gains (e.g. "0.50" for 50%)
# Minimum portfolio floor -- never withdraw if it would drop below this
WITHDRAWAL_FLOOR = 10000.0


# ── ALPACA CLIENT ─────────────────────────────────────────────────────────────

def get_alpaca():
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        log.error("alpaca-trade-api not installed. Add to requirements.txt:")
        log.error("  alpaca-trade-api>=3.0.0")
        sys.exit(1)

    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")

    if not key or not secret:
        log.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        sys.exit(1)

    api = tradeapi.REST(key, secret, BASE_URL, api_version="v2")
    mode = "PAPER" if PAPER_MODE else "LIVE"
    log.info(f"Alpaca connected [{mode}]")
    return api


# ── STATE MANAGEMENT ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if TRADE_STATE.exists():
        with open(TRADE_STATE) as f:
            return json.load(f)
    return {"week_open_value": None, "positions": {}, "entry_date": None}


def save_state(state: dict):
    with open(TRADE_STATE, "w") as f:
        json.dump(state, f, indent=2)


# ── POSITION SIZING ───────────────────────────────────────────────────────────

def compute_positions(scores: pd.DataFrame, portfolio_value: float) -> list[dict]:
    """
    Dynamically determine position list and sizing.

    Logic:
    1. Filter to tradable, qualifying names
    2. Compute max positions from capital / min position size
    3. Equal-weight capital across qualifying positions
    4. Apply individual position caps

    Returns list of dicts: {symbol, dollar_amount, shares, price}
    """
    # Filter gates
    df = scores.copy()

    # Price gate -- no sub-$1 stocks
    if "last_price" in df.columns:
        df = df[df["last_price"].notna() & (df["last_price"] >= MIN_PRICE)]

    # Volume gate
    if "avg_vol_20d" in df.columns:
        df = df[df["avg_vol_20d"].notna() & (df["avg_vol_20d"] >= MIN_AVG_VOLUME)]

    # Signal quality gate -- must be in top (1 - MIN_COMPOSITE_PERCENTILE) by composite rank.
    # We use composite_rank (ties alpha + EV 50/50) when available, falling back to alpha_pct_rank
    # only for legacy scores files that predate composite rank persistence.
    # The cutoff uses the FULL scored universe size (from scores), not the filtered df, because
    # composite_rank values reference the original universe-wide ranking.
    if "composite_rank" in df.columns and df["composite_rank"].notna().any():
        n_total     = int(pd.to_numeric(scores["composite_rank"], errors="coerce").max())
        cutoff_rank = max(1, int(n_total * (1.0 - MIN_COMPOSITE_PERCENTILE)))
        df = df[df["composite_rank"].notna() & (df["composite_rank"] <= cutoff_rank)]
        log.info(f"  Composite gate: keeping top {cutoff_rank} of {n_total} "
                 f"({(1-MIN_COMPOSITE_PERCENTILE)*100:.0f}%)")
    elif "alpha_pct_rank" in df.columns:
        df = df[df["alpha_pct_rank"] >= MIN_COMPOSITE_PERCENTILE]
        log.warning("  composite_rank missing from scores -- falling back to alpha_pct_rank gate")

    # Conviction gate -- very_high and high only
    if "conviction" in df.columns:
        df = df[df["conviction"].isin(["very_high", "high"])]

    # Sort by composite rank (ascending = best first)
    sort_col = "composite_rank" if "composite_rank" in df.columns else "alpha_rank"
    df = df.sort_values(sort_col).reset_index(drop=True)

    if df.empty:
        log.warning("No qualifying tickers after filters")
        return []

    # Dynamic position count
    # Max positions = how many MIN_POSITION_SIZE slots fit in portfolio
    max_by_capital = int(portfolio_value / MIN_POSITION_SIZE)

    # Also cap by liquidity -- position size must be < MAX_ADV_PCT of ADV
    # At current capital this won't bind but the logic is here for scaling
    if "avg_vol_20d" in df.columns and "last_price" in df.columns:
        # Estimate equal-weight position size
        est_position = portfolio_value / min(max_by_capital, len(df))
        df["adv_dollar"] = df["avg_vol_20d"] * df["last_price"]
        df["max_by_liquidity"] = df["adv_dollar"] * MAX_ADV_PCT
        # Flag names where our position would exceed liquidity cap
        df["liquidity_ok"] = df["max_by_liquidity"] >= est_position
        df = df[df["liquidity_ok"]]

    if df.empty:
        log.warning("No qualifying tickers after liquidity filter")
        return []

    # Target position count: minimum of capital-constrained max and qualified names
    n_positions = min(max_by_capital, len(df))

    # Build extended candidate pool -- up to 2x target count for fallback coverage.
    # If primary candidates fail (untradable, halted, etc.) fallbacks ensure capital
    # stays fully deployed rather than sitting idle.
    #
    # Each position is tagged is_primary/is_fallback in-place so run_entry doesn't
    # have to re-derive the target count from the returned list length. Previously
    # run_entry did `n_target = len(positions) // 2 or len(positions)` which under-
    # counted when the qualifying pool was <2x target.
    n_candidates = min(n_positions * 2, len(df))
    df_candidates = df.head(n_candidates)

    log.info(f"Position sizing: target={n_positions} positions, "
             f"candidates={n_candidates} (fallback pool)")
    log.info(f"  Capital: ${portfolio_value:,.0f}  "
             f"Per position: ${portfolio_value/n_positions:,.0f}")

    # Equal weight based on target count (not candidate count)
    position_size = portfolio_value / n_positions

    # Apply max position cap
    max_position = portfolio_value * MAX_POSITION_PCT
    if position_size > max_position:
        log.warning(f"Position size ${position_size:,.0f} exceeds "
                    f"{MAX_POSITION_PCT*100:.0f}% cap -- using ${max_position:,.0f}")
        position_size = max_position

    positions = []
    for idx, (_, row) in enumerate(df_candidates.iterrows()):
        price = float(row.get("last_price", 0))
        if price <= 0:
            continue
        shares = int(position_size / price)  # whole shares only
        if shares < 1:
            continue
        actual_size = shares * price
        positions.append({
            "symbol":      str(row["symbol"]),
            "shares":      shares,
            "price":       round(price, 2),
            "dollar_size": round(actual_size, 2),
            "conviction":  str(row.get("conviction", "")),
            "composite_rank": int(row.get("composite_rank", row.get("alpha_rank", 9999))),
            "alpha_score": float(row.get("alpha_score", 0) or 0),
            "ev_score":    float(row.get("ev_score", 0) or 0),
            "weekly_vol":  float(row.get("weekly_vol", 0) or 0),
            "suggested_hard_stop_pct":   row.get("suggested_hard_stop_pct"),
            "suggested_activation_pct":  row.get("suggested_activation_pct"),
            "suggested_trail_pct":       row.get("suggested_trail_pct"),
            # Primary / fallback flag. First n_positions are primaries; remainder are fallbacks.
            "is_primary":  idx < n_positions,
            "is_fallback": idx >= n_positions,
        })

    total_deployed = sum(p["dollar_size"] for p in positions)
    log.info(f"  Total deployed: ${total_deployed:,.0f} / ${portfolio_value:,.0f} "
             f"({total_deployed/portfolio_value*100:.1f}%)")

    return positions


# ── ENTRY (MONDAY 2:30 PM CT) ─────────────────────────────────────────────────

def run_entry():
    log.info("=" * 60)
    log.info("ALPACA TRADER -- MONDAY ENTRY")
    log.info(f"MODE: {'PAPER' if PAPER_MODE else '*** LIVE ***'}")
    log.info("=" * 60)

    api = get_alpaca()

    # Load scores
    if not SCORES_CSV.exists():
        log.error("No scores_final.csv -- run weekly pipeline first")
        return

    scores = pd.read_csv(SCORES_CSV)
    log.info(f"Loaded {len(scores):,} scored tickers")

    # Load regime
    regime = "unknown"
    if REGIME_JSON.exists():
        with open(REGIME_JSON) as f:
            regime_data = json.load(f)
        regime = regime_data.get("regime", "unknown")
    log.info(f"Regime: {regime}")

    # Get portfolio value
    account = api.get_account()
    portfolio_value = float(account.portfolio_value)
    cash            = float(account.cash)
    log.info(f"Portfolio: ${portfolio_value:,.2f}  Cash: ${cash:,.2f}")

    # MARGIN GUARD -- always deploy cash only, never margin
    # Cash < portfolio_value means margin is available but we never touch it
    deployable = cash
    if cash <= 0:
        log.warning("No settled cash available -- skipping entry")
        return
    if cash < portfolio_value * 0.45:
        # Sanity check -- if cash is less than 45% of portfolio value something is wrong
        log.warning(f"Cash ${cash:,.2f} is unusually low vs portfolio ${portfolio_value:,.2f} "
                    f"-- possible unsettled trades or margin usage. Skipping entry.")
        return
    log.info(f"Deployable (cash only, no margin): ${deployable:,.2f}")

    # Check for existing positions -- don't double-enter
    existing = {p.symbol for p in api.list_positions()}
    if existing:
        log.warning(f"Already holding {len(existing)} positions: {existing}")
        log.warning("Skipping entry -- close existing positions first")
        return

    positions  = compute_positions(scores, deployable)

    if not positions:
        log.warning("No positions to enter")
        return

    # Partition positions using the is_primary/is_fallback tags set by compute_positions.
    # Do not re-derive n_target from len(positions)//2 -- that silently under-counts when
    # the qualifying pool was smaller than 2x target (which is common on narrow universes
    # or after restrictive gates).
    primaries = [p for p in positions if p.get("is_primary")]
    fallbacks = [p for p in positions if p.get("is_fallback")]
    n_target  = len(primaries) if primaries else len(positions)

    log.info(f"Entry plan: {n_target} primary targets + {len(fallbacks)} fallbacks")

    # Place orders -- use fallbacks if primaries fail
    filled    = []
    failed    = []
    skipped   = []
    state     = load_state()
    state["entry_date"]       = datetime.date.today().isoformat()
    state["week_open_value"]  = portfolio_value
    state["positions"]        = {}

    candidates_queue = primaries + fallbacks

    for p in candidates_queue:
        if len(filled) >= n_target:
            break  # reached target -- stop even if fallbacks remain

        sym    = p["symbol"]
        shares = p["shares"]
        is_fallback = p.get("is_fallback", False)

        try:
            order = api.submit_order(
                symbol        = sym,
                qty           = shares,
                side          = "buy",
                type          = "market",
                time_in_force = "cls",
            )
            tag = " [FALLBACK]" if is_fallback else ""
            log.info(f"  ORDER{tag} {sym:<8} {shares:>4} shares @ ~${p['price']:.2f}  "
                     f"(${p['dollar_size']:,.0f})  "
                     f"rank={p['composite_rank']}  {p['conviction']}")
            filled.append(sym)
            state["positions"][sym] = {
                "shares":               shares,
                "entry_price_est":      p["price"],
                "hard_stop_pct":        p["suggested_hard_stop_pct"],
                "activation_pct":       p["suggested_activation_pct"],
                "trail_pct":            p["suggested_trail_pct"],
                "alpha_score":          p.get("alpha_score", 0.75),
                "weekly_vol":           p.get("weekly_vol", 0.20),
                "composite_rank":       p.get("composite_rank"),
                "is_fallback":          is_fallback,
                "phase":                1,
                "high_water_mark":      p["price"],
                "order_id":             order.id,
            }
        except Exception as e:
            log.error(f"  FAILED {sym}: {e}")
            failed.append(sym)
            if is_fallback:
                log.warning(f"  Fallback {sym} also failed -- continuing to next")

    # Log any unused fallbacks
    unused = [p["symbol"] for p in fallbacks if p["symbol"] not in filled and p["symbol"] not in failed]
    if unused:
        skipped = unused
        log.info(f"  Unused fallbacks (not needed): {skipped}")

    save_state(state)

    log.info(f"Entry complete: {len(filled)} filled, {len(failed)} failed, "
             f"{len(skipped)} fallbacks unused")
    fallback_used = [s for s in filled if state["positions"][s].get("is_fallback")]
    if fallback_used:
        log.info(f"  Fallbacks used: {fallback_used}")
    if failed:
        log.warning(f"  Failed: {failed}")


# ── EXIT (FRIDAY 2:30 PM CT) ──────────────────────────────────────────────────

def run_exit():
    log.info("=" * 60)
    log.info("ALPACA TRADER -- FRIDAY EXIT")
    log.info(f"MODE: {'PAPER' if PAPER_MODE else '*** LIVE ***'}")
    log.info("=" * 60)

    api = get_alpaca()

    positions = api.list_positions()
    if not positions:
        log.info("No open positions to close")
        return

    account          = api.get_account()
    portfolio_value  = float(account.portfolio_value)
    state            = load_state()
    week_open        = state.get("week_open_value") or portfolio_value
    week_return      = (portfolio_value - week_open) / week_open
    log.info(f"Week return: {week_return*100:+.2f}%  "
             f"(${week_open:,.0f} -> ${portfolio_value:,.0f})")

    # Circuit breaker check
    if CIRCUIT_BREAKER_PCT and week_return < -CIRCUIT_BREAKER_PCT:
        log.warning(f"CIRCUIT BREAKER: down {week_return*100:.1f}% -- exiting all")

    closed = []
    failed = []

    for pos in positions:
        sym    = pos.symbol
        shares = int(pos.qty)
        try:
            api.submit_order(
                symbol        = sym,
                qty           = shares,
                side          = "sell",
                type          = "market",
                time_in_force = "cls",   # market-on-close
            )
            ret_pct = float(pos.unrealized_plpc) * 100
            log.info(f"  CLOSE {sym:<8} {shares:>4} shares  "
                     f"return={ret_pct:+.2f}%  "
                     f"P&L=${float(pos.unrealized_pl):+,.2f}")
            closed.append(sym)
        except Exception as e:
            log.error(f"  FAILED {sym}: {e}")
            failed.append(sym)

    log.info(f"Exit complete: {len(closed)} closed, {len(failed)} failed")

    # Handle withdrawal if enabled
    run_withdrawal(api, portfolio_value, week_return, week_open)

    # Clear state
    state["positions"]       = {}
    state["entry_date"]      = None
    state["week_open_value"] = None
    save_state(state)


# ── WITHDRAWAL ────────────────────────────────────────────────────────────────
#
# Four modes controlled by GitHub secrets:
#
#   ALPACA_WITHDRAWAL_MODE = "off"    -- pure compounding, no withdrawals (default)
#   ALPACA_WITHDRAWAL_MODE = "debt"   -- withdraw 100% of weekly gains for debt payoff
#   ALPACA_WITHDRAWAL_MODE = "income" -- withdraw ALPACA_WITHDRAWAL_PCT of weekly gains
#
#   ALPACA_LUMP_SUM_AMOUNT = "5000"   -- one-time pull regardless of mode
#                                        clear this secret manually after it fires
#
# Floor: WITHDRAWAL_FLOOR = $10,000 -- never withdraw below this portfolio value
# Income pct: ALPACA_WITHDRAWAL_PCT = "0.25" for 25% of weekly gains

def execute_ach(api, amount: float, reason: str):
    """Execute ACH transfer to linked bank account."""
    try:
        log.info(f"WITHDRAWAL [{reason}]: ${amount:,.2f} -> linked bank account")
        log.info("  (ACH transfer API call placeholder -- uncomment when bank linked)")
        # Uncomment when ready:
        # api.initiate_transfer(
        #     transfer_type="ach",
        #     direction="outgoing",
        #     timing="immediate",
        #     amount=str(round(amount, 2)),
        # )
    except Exception as e:
        log.error(f"ACH transfer failed: {e}")


def apply_floor(amount: float, portfolio_value: float, reason: str) -> float:
    """Enforce portfolio floor -- never withdraw below WITHDRAWAL_FLOOR."""
    post = portfolio_value - amount
    if post < WITHDRAWAL_FLOOR:
        adjusted = portfolio_value - WITHDRAWAL_FLOOR
        if adjusted <= 0:
            log.warning(f"{reason}: skipped -- portfolio ${portfolio_value:,.0f} "
                        f"at or below floor ${WITHDRAWAL_FLOOR:,.0f}")
            return 0
        log.warning(f"{reason}: adjusted from ${amount:,.0f} to ${adjusted:,.0f} "
                    f"to protect ${WITHDRAWAL_FLOOR:,.0f} floor")
        return adjusted
    return amount


def run_withdrawal(api, portfolio_value: float, week_return: float, week_open: float):
    week_gains = portfolio_value - week_open

    # ── LUMP SUM (fires regardless of mode, one-time) ──────────────────────
    lump_str = os.environ.get("ALPACA_LUMP_SUM_AMOUNT", "").strip()
    if lump_str and float(lump_str) > 0:
        lump = apply_floor(float(lump_str), portfolio_value, "Lump sum")
        if lump > 0:
            execute_ach(api, lump, "lump sum")
            log.info("  Clear ALPACA_LUMP_SUM_AMOUNT secret manually to prevent re-firing")
            portfolio_value -= lump  # update for subsequent floor checks
    else:
        log.info("Lump sum: not set")

    # ── WEEKLY MODE ────────────────────────────────────────────────────────
    mode = os.environ.get("ALPACA_WITHDRAWAL_MODE", "off").lower().strip()
    log.info(f"Withdrawal mode: {mode}")

    if mode == "off":
        log.info("  Pure compounding -- no weekly withdrawal")
        return

    elif mode == "debt":
        if week_gains <= 0:
            log.info("  Debt mode: no gains this week -- skipping")
            return
        amount = apply_floor(week_gains, portfolio_value, "Debt payoff")
        if amount > 0:
            log.info(f"  Debt mode: withdrawing 100% of gains ${week_gains:,.0f}")
            execute_ach(api, amount, "debt payoff")

    elif mode == "income":
        pct_str = os.environ.get("ALPACA_WITHDRAWAL_PCT", "0.25").strip()
        pct = float(pct_str)
        if week_gains <= 0:
            log.info(f"  Income mode ({pct*100:.0f}%): no gains this week -- skipping")
            return
        raw = week_gains * pct
        amount = apply_floor(raw, portfolio_value, f"Income ({pct*100:.0f}%)")
        if amount > 0:
            log.info(f"  Income mode: withdrawing {pct*100:.0f}% of "
                     f"${week_gains:,.0f} gains = ${amount:,.0f}")
            execute_ach(api, amount, f"income {pct*100:.0f}%")

    else:
        log.warning(f"  Unknown withdrawal mode '{mode}' -- no withdrawal")
        log.warning("  Valid modes: off, debt, income")


# ── CIRCUIT BREAKER (MID-WEEK CHECK) ─────────────────────────────────────────

def run_circuit_breaker_check():
    if not CIRCUIT_BREAKER_PCT:
        return

    api   = get_alpaca()
    state = load_state()
    week_open = state.get("week_open_value")

    if not week_open:
        return

    account         = api.get_account()
    portfolio_value = float(account.portfolio_value)
    week_return     = (portfolio_value - week_open) / week_open

    log.info(f"Circuit breaker check: {week_return*100:+.2f}% "
             f"(threshold: -{CIRCUIT_BREAKER_PCT*100:.0f}%)")

    if week_return < -CIRCUIT_BREAKER_PCT:
        log.warning(f"CIRCUIT BREAKER TRIGGERED: {week_return*100:.1f}% drawdown")
        log.warning("Closing all positions...")
        positions = api.list_positions()
        for pos in positions:
            try:
                api.submit_order(
                    symbol        = pos.symbol,
                    qty           = int(pos.qty),
                    side          = "sell",
                    type          = "market",
                    time_in_force = "day",
                )
                log.info(f"  Emergency close: {pos.symbol}")
            except Exception as e:
                log.error(f"  Failed to close {pos.symbol}: {e}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(mode: str = "entry"):
    from automation.tz_utils import now_ct, is_entry_day, is_exit_day, get_entry_day, get_exit_day, is_trading_day
    ct      = now_ct()
    today   = ct.date()
    log.info(f"Alpaca trader running at {ct.strftime('%A %Y-%m-%d %H:%M CT')}")
    log.info(f"Mode: {mode}")

    if mode == "entry":
        # Check if today is the correct entry day accounting for holidays
        entry_day = get_entry_day(today)
        if entry_day == "skip":
            log.info("Both Monday and Tuesday are holidays -- skipping entry this week")
            return
        if not is_entry_day(today):
            log.info(f"Today is not the entry day for this week "
                     f"(expected {entry_day}) -- skipping")
            return
        log.info(f"Entry day confirmed: {entry_day}")
        run_entry()

    elif mode == "exit":
        # Check if today is the correct exit day accounting for holidays
        exit_day = get_exit_day(today)
        if exit_day == "skip":
            log.info("Both Friday and Thursday are holidays -- skipping exit this week")
            return
        if not is_exit_day(today):
            log.info(f"Today is not the exit day for this week "
                     f"(expected {exit_day}) -- skipping")
            return
        log.info(f"Exit day confirmed: {exit_day}")
        run_exit()

    elif mode == "circuit_breaker":
        if not is_trading_day(today):
            log.info("Market holiday -- skipping circuit breaker check")
            return
        run_circuit_breaker_check()

    else:
        log.error(f"Unknown mode: {mode}. Use 'entry', 'exit', or 'circuit_breaker'")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "entry"
    run(mode)
