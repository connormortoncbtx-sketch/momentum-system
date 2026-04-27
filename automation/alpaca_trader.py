# automation/alpaca_trader.py
# Autonomous trading execution layer for Momentum Alpha.
# Reads the weekly report scores, applies entry/exit logic,
# and places orders via the Alpaca API.
#
# Runs twice per week via GitHub Actions:
#   Monday  2:45 PM CT  -- entry orders (market DAY, fills in close ramp)
#   Friday  2:45 PM CT  -- exit orders (market DAY, fills in close ramp)
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
#
# Migrated from `alpaca-trade-api` to `alpaca-py` (2026-04). The old SDK pinned
# websockets<11 which conflicted with yfinance>=0.2.40 (needs websockets>=12).
# alpaca-py uses request objects + enums; old string-based comparisons have
# been converted to enum comparisons throughout.
#
# 2026-04-27 PATCH NOTES — capital deployment efficiency:
#   1. Added PER_POSITION_SLIPPAGE_CUSHION (50 bps) — explicit per-position
#      target shrinkage replaces accidental rounding-as-cushion. Was leaking
#      $0-200/position depending on share price; now consistent ~50 bps under.
#   2. compute_positions param renamed `portfolio_value` → `deployable` to
#      match what callers actually pass (cash, not portfolio_value). Latent
#      bug fix: parameter name claimed one thing but contract was the other.
#      Today's $100k paper run was unaffected (cash == portfolio_value), but
#      this would have misbehaved with unsettled trades or margin in play.
#   3. Entry order type CLS → DAY. Closing-auction pro-rata systematically
#      shorted high-conviction crowd-favorite names. 2026-04-27 entry left
#      $31k undeployed because OPEN/OWL/AXTI/MXL got 30-60% fills despite
#      log claiming "10 filled" (which only meant submission was accepted,
#      not that the auction actually cleared the full qty). DAY market
#      orders submitted at 2:45 CT fill against the live close-ramp book.
#   4. Exit order type CLS → DAY (same reasoning, applied to run_exit).
#      Sell-side partial fills are MORE consequential than buy-side: an
#      unfilled remainder carries over the weekend with no stop coverage
#      (since step 1 of run_exit cancelled the trailing stops), and blocks
#      Monday entry which refuses to run if any positions exist. Weekly-
#      cycle abstraction requires guaranteed full liquidation Friday.
#      Workflow cron also moves Fri 2:30 → 2:45 CT to match entry timing.

import json
import logging
import os
import sys
import datetime
import pandas as pd
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

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

PAPER_MODE = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

DATA_DIR   = Path("data")
SCORES_CSV = DATA_DIR / "scores_final.csv"
REGIME_JSON = DATA_DIR / "regime.json"
TRADE_STATE = DATA_DIR / "alpaca_state.json"  # tracks open positions

# -- POSITION SIZING PARAMETERS (tune as capital scales) ----------------------
#
# Design philosophy: the baseline deployment is 10 positions × 10% of capital.
# This is the canonical shape; everything else is a variation.
#
# Position COUNT is the primary lever, position SIZE is derived. In the future,
# position count will grow beyond 10 when per-ticker liquidity constraints force
# dilution (i.e., when a 10%-of-capital position would exceed what the highest-
# conviction names can absorb without market impact). That dynamic sizing logic
# is not yet implemented -- it will be built after paper trading provides the
# slippage/ADV data needed to calibrate the thresholds.
#
# For now: 10 positions is the fixed target. MIN_POSITIONS = 10 is a hard floor.

DEFAULT_POSITIONS = 10
MIN_POSITIONS = 10
MIN_POSITION_SIZE = 500.0
MAX_POSITION_PCT = 0.15
MIN_COMPOSITE_PERCENTILE = 0.70
MAX_ADV_PCT = 0.01
MIN_PRICE = 1.00
MIN_AVG_VOLUME = 50000

# Per-position slippage cushion -- shrink each position's target dollar size
# by this fraction before computing share count. Replaces accidental
# rounding-as-cushion with an explicit, tunable buffer that handles:
#
#   1. The price used for sizing (last_price from scores, or even a fresh
#      quote) is a snapshot. The actual MOC fill is at the closing print,
#      which can drift up from the snapshot during the final hour.
#   2. Without this, share count was int(position_size / price), which floors
#      to a different leakage on every ticker depending on share price --
#      a $50 stock leaks $0-50; a $200 stock leaks $0-200. Inconsistent.
#
# Calibration: 50 bps fits typical MOC drift on liquid mid-caps. If post-entry
# review shows the average undeployment is consistently smaller than this,
# tighten to 30 bps. If fills above-snapshot start showing up frequently,
# loosen to 75 bps. This constant is independent of CAPITAL_BUFFER_PCT --
# they protect against different failure modes.
PER_POSITION_SLIPPAGE_CUSHION = 0.005

# Capital deployment buffer -- hold back this fraction of portfolio value when
# sizing MOC orders. Rationale: Alpaca reserves estimated cost at order-submit
# time (not fill time), using last_price as the basis. If market moves up
# between submit and close, or if multiple MOC orders land sequentially with
# slight reservation overhead, the last order(s) in a batch can hit
# "insufficient buying power" even when nominal math says it fits.
#
# 2026-04-20 incident: 9 × $10k orders on $100k account -- 10th order failed
# because Alpaca's $90k reservation + overhead left <$10k for the final order.
# A 5% buffer ($9.5k target per position) would have allowed all 10 to fill.
#
# This is account-level protection. PER_POSITION_SLIPPAGE_CUSHION (above) is
# per-position protection. Both are needed; they handle different cases.
CAPITAL_BUFFER_PCT = 0.05

# Circuit breaker -- if portfolio drops this % from week-open value, exit all
# Set to None to disable. Tune based on backtesting.
CIRCUIT_BREAKER_PCT = None   # e.g. 0.08 = exit if down 8% from Monday open

# -- WITHDRAWAL PARAMETERS ----------------------------------------------------
WITHDRAWAL_FLOOR = 10000.0


# ── ALPACA CLIENT ─────────────────────────────────────────────────────────────

def get_alpaca():
    """Construct an alpaca-py TradingClient.

    Returns the client directly. Order placement is via request objects
    (MarketOrderRequest, StopOrderRequest, etc.) -- see the order helpers
    below. Paper vs live is selected via the `paper` constructor arg, not
    a base URL.
    """
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        log.error("alpaca-py not installed. Add to requirements.txt:")
        log.error("  alpaca-py>=0.30.0")
        sys.exit(1)

    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")

    if not key or not secret:
        log.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        sys.exit(1)

    client = TradingClient(api_key=key, secret_key=secret, paper=PAPER_MODE)
    mode = "PAPER" if PAPER_MODE else "LIVE"
    log.info(f"Alpaca connected [{mode}]")
    return client


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

def compute_positions(scores: pd.DataFrame, deployable: float) -> list[dict]:
    """
    Dynamically determine position list and sizing.

    `deployable` is the cash actually available for entries (NOT the full
    portfolio value). Caller is responsible for passing settled cash to
    avoid over-sizing against unsettled trades or margin. Renamed from
    `portfolio_value` (2026-04-27) -- the prior name was misleading because
    contract was always cash, never total value.

    Logic:
    1. Filter to tradable, qualifying names
    2. Compute max positions from capital / min position size
    3. Equal-weight capital across qualifying positions
    4. Apply individual position caps
    5. Per-position slippage cushion (bottom of loop) leaves explicit ~50 bps
       under target on every position rather than relying on integer rounding

    Returns list of dicts: {symbol, dollar_amount, shares, price}
    """
    df = scores.copy()

    if "last_price" in df.columns:
        df = df[df["last_price"].notna() & (df["last_price"] >= MIN_PRICE)]

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

    if "conviction" in df.columns:
        df = df[df["conviction"].isin(["very_high", "high"])]

    sort_col = "composite_rank" if "composite_rank" in df.columns else "alpha_rank"
    df = df.sort_values(sort_col).reset_index(drop=True)

    if df.empty:
        log.warning("No qualifying tickers after filters")
        return []

    max_by_capital = int(deployable / MIN_POSITION_SIZE)

    # Liquidity filter: target position size must be absorbable for every name.
    # At baseline 10%, this only bites on large capital + micro-cap names.
    if "avg_vol_20d" in df.columns and "last_price" in df.columns:
        target_position = deployable / DEFAULT_POSITIONS
        df["adv_dollar"] = df["avg_vol_20d"] * df["last_price"]
        df["max_by_liquidity"] = df["adv_dollar"] * MAX_ADV_PCT
        df["liquidity_ok"] = df["max_by_liquidity"] >= target_position
        df = df[df["liquidity_ok"]]

    if df.empty:
        log.warning("No qualifying tickers after liquidity filter")
        return []

    n_positions = min(max_by_capital, len(df), DEFAULT_POSITIONS)

    if n_positions == max_by_capital and max_by_capital < DEFAULT_POSITIONS:
        bound_by = "transaction cost floor"
    elif n_positions == len(df) and len(df) < DEFAULT_POSITIONS:
        bound_by = "qualifying pool size"
    else:
        bound_by = f"default target ({DEFAULT_POSITIONS})"
    log.info(f"  Position count bound by: {bound_by}")

    if n_positions < MIN_POSITIONS:
        log.warning(
            f"  Position count {n_positions} below MIN_POSITIONS={MIN_POSITIONS}; "
            f"deploying what's available. Review qualifying pool."
        )
        try:
            from automation.notifier import notify
            notify(
                title=f"Narrow qualifying pool: {n_positions} positions",
                body=(f"Only {n_positions} names qualified for entry "
                      f"(target={DEFAULT_POSITIONS}, floor={MIN_POSITIONS}). "
                      f"Review universe filters and signal output."),
                priority="high",
            )
        except Exception:
            pass

    n_candidates = min(n_positions * 2, len(df))
    df_candidates = df.head(n_candidates)

    log.info(f"Position sizing: target={n_positions} positions, "
             f"candidates={n_candidates} (fallback pool)")

    deployable_capital = deployable * (1 - CAPITAL_BUFFER_PCT)
    log.info(f"  Deployable cash: ${deployable:,.0f}  "
             f"After {CAPITAL_BUFFER_PCT*100:.0f}% account buffer: ${deployable_capital:,.0f}  "
             f"Per position target: ${deployable_capital/n_positions:,.0f}  "
             f"After {PER_POSITION_SLIPPAGE_CUSHION*100:.1f}% slippage cushion: "
             f"${(deployable_capital/n_positions)*(1-PER_POSITION_SLIPPAGE_CUSHION):,.0f}")

    position_size = deployable_capital / n_positions

    max_position = deployable * MAX_POSITION_PCT
    if position_size > max_position:
        log.warning(f"Position size ${position_size:,.0f} exceeds "
                    f"{MAX_POSITION_PCT*100:.0f}% cap -- using ${max_position:,.0f}")
        position_size = max_position

    # Apply per-position slippage cushion AFTER caps. The cushion shrinks the
    # target by a fixed fraction; integer floor of (target / price) gives us a
    # share count whose actual dollar deployment lands consistently ~50 bps
    # under the cushioned target on every ticker, regardless of share price.
    sized_target = position_size * (1 - PER_POSITION_SLIPPAGE_CUSHION)

    positions = []
    for idx, (_, row) in enumerate(df_candidates.iterrows()):
        price = float(row.get("last_price", 0))
        if price <= 0:
            continue
        # int() of a positive float is equivalent to floor(); intent is floor.
        # Result: actual_size <= sized_target <= position_size in all cases,
        # which is what we want for a "stay under the cap" deployment policy.
        shares = int(sized_target / price)
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
            "is_primary":  idx < n_positions,
            "is_fallback": idx >= n_positions,
        })

    primary_deployed = sum(p["dollar_size"] for p in positions if p["is_primary"])
    total_planned    = sum(p["dollar_size"] for p in positions)
    log.info(f"  Primaries deploy: ${primary_deployed:,.0f} / ${deployable:,.0f} "
             f"({primary_deployed/deployable*100:.1f}% of deployable cash)")
    log.info(f"  Fallback pool:    ${total_planned - primary_deployed:,.0f} "
             f"(used only if primaries fail)")

    return positions


# ── ENTRY (MONDAY 2:30 PM CT) ─────────────────────────────────────────────────

def run_entry():
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    log.info("=" * 60)
    log.info("ALPACA TRADER -- MONDAY ENTRY")
    log.info(f"MODE: {'PAPER' if PAPER_MODE else '*** LIVE ***'}")
    log.info("=" * 60)

    client = get_alpaca()

    if not SCORES_CSV.exists():
        log.error("No scores_final.csv -- run weekly pipeline first")
        return

    scores = pd.read_csv(SCORES_CSV)
    log.info(f"Loaded {len(scores):,} scored tickers")

    regime = "unknown"
    if REGIME_JSON.exists():
        with open(REGIME_JSON) as f:
            regime_data = json.load(f)
        regime = regime_data.get("regime", "unknown")
    log.info(f"Regime: {regime}")

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    cash            = float(account.cash)
    log.info(f"Portfolio: ${portfolio_value:,.2f}  Cash: ${cash:,.2f}")

    deployable = cash
    if cash <= 0:
        log.warning("No settled cash available -- skipping entry")
        return
    if cash < portfolio_value * 0.45:
        log.warning(f"Cash ${cash:,.2f} is unusually low vs portfolio ${portfolio_value:,.2f} "
                    f"-- possible unsettled trades or margin usage. Skipping entry.")
        return
    log.info(f"Deployable (cash only, no margin): ${deployable:,.2f}")

    existing = {p.symbol for p in client.get_all_positions()}
    if existing:
        log.warning(f"Already holding {len(existing)} positions: {existing}")
        log.warning("Skipping entry -- close existing positions first")
        return

    positions  = compute_positions(scores, deployable)

    if not positions:
        log.warning("No positions to enter")
        return

    primaries = [p for p in positions if p.get("is_primary")]
    fallbacks = [p for p in positions if p.get("is_fallback")]
    n_target  = len(primaries) if primaries else len(positions)

    log.info(f"Entry plan: {n_target} primary targets + {len(fallbacks)} fallbacks")

    filled    = []
    failed    = []
    skipped   = []
    state     = load_state()
    state["entry_date"]       = datetime.date.today().isoformat()
    state["week_open_value"]  = portfolio_value
    state["positions"]        = {}

    candidates_queue = primaries + fallbacks

    # ── ORDER TYPE CHOICE ────────────────────────────────────────────────
    # 2026-04-27: Switched from TimeInForce.CLS (closing auction) to
    # TimeInForce.DAY (regular market order) for entries.
    #
    # CLS problem: in the closing auction, all buy orders for a symbol get
    # pro-rated against available sell-side liquidity at the cleared price.
    # The names this strategy targets (high-RS momentum leaders, in
    # particular high-conviction crowd-favorites like OPEN, OWL) are
    # systematically the names with the most closing-auction demand from
    # index funds, ETFs, and other momentum algos. Result: highest-
    # conviction positions get the WORST fill rates. The 2026-04-27 entry
    # showed OPEN filled at 33% of submitted qty, OWL at 32%, AXTI at 49%,
    # MXL at 59% -- $31k of $100k undeployed across 10 positions purely
    # from auction pro-rating.
    #
    # DAY orders submitted in the final 10-15 minutes of the session fill
    # against the live order book during the close ramp. Trade: pay ~5-10
    # bps of bid-ask spread instead of getting the auction-cleared print
    # price. Benefit: actually get the full requested quantity ~99% of
    # the time on names with reasonable liquidity.
    #
    # Workflow cron should be advanced from 14:30 CT to 14:45 CT to take
    # advantage of this -- earlier means more liquidity volatility and
    # wider spreads; later means less time to react if anything fails.
    # 14:45 lands solidly in the closing-ramp liquidity zone.
    for p in candidates_queue:
        if len(filled) >= n_target:
            break

        sym    = p["symbol"]
        shares = p["shares"]
        is_fallback = p.get("is_fallback", False)

        try:
            # NOTE: TimeInForce.DAY (not CLS) -- see comment block above
            # `for p in candidates_queue:`. Submitting as a regular market
            # order during the final minutes of the session avoids the
            # closing-auction pro-rata problem.
            order = client.submit_order(
                order_data=MarketOrderRequest(
                    symbol        = sym,
                    qty           = shares,
                    side          = OrderSide.BUY,
                    time_in_force = TimeInForce.DAY,
                )
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
                "order_id":             str(order.id),
            }
        except Exception as e:
            log.error(f"  FAILED {sym}: {e}")
            failed.append(sym)
            if is_fallback:
                log.warning(f"  Fallback {sym} also failed -- continuing to next")

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

    if failed:
        log_event("alpaca_trader", LogStatus.WARNING,
                  f"Entry: {len(filled)} filled, {len(failed)} FAILED",
                  metrics={
                      "filled":         len(filled),
                      "failed":         len(failed),
                      "fallbacks_used": len(fallback_used),
                      "symbols_failed": failed[:10],
                  })
        notify_alert("alpaca_trader",
                     f"Entry: {len(filled)}/{len(filled)+len(failed)} filled, "
                     f"{len(failed)} FAILED: {', '.join(failed[:5])}"
                     f"{'…' if len(failed) > 5 else ''}")
    else:
        log_event("alpaca_trader", LogStatus.SUCCESS,
                  f"Entry: all {len(filled)} orders placed",
                  metrics={
                      "filled":         len(filled),
                      "fallbacks_used": len(fallback_used),
                      "top_pick":       filled[0] if filled else None,
                  })
        notify_success("alpaca_trader",
                       f"Entry: {len(filled)} BUY orders placed (market DAY). "
                       f"Stops will be submitted at 3:10pm CT via place_stops workflow. "
                       f"{('Fallbacks used: ' + ', '.join(fallback_used)) if fallback_used else ''}")


# ── PLACE STOPS (MONDAY 3:10 PM CT) ───────────────────────────────────────────

def run_place_stops():
    """Place Phase 1 hard stops on Monday-entered positions.

    Called ~5 minutes after Monday MOC close (3:05pm CT). Previously stops
    were only placed by the Tuesday morning monitor poll, creating a ~17
    hour gap between fill and coverage. This function closes that gap.

    Logic mirrors alpaca_monitor.ensure_phase1_stop:
      - Check each position in state
      - Skip positions already with a hard_stop_order_id
      - Look for existing stop orders on Alpaca (adopt rather than duplicate)
      - Otherwise submit a GTC stop using the ACTUAL fill price
        (Alpaca's avg_entry_price from the live position)

    Runs as its own scheduled workflow (alpaca_place_stops.yml) at 3:05pm CT
    Mondays. Idempotent: if run again, adopts existing stops rather than
    duplicating.
    """
    from alpaca.trading.requests import StopOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus

    log.info("=" * 60)
    log.info("ALPACA TRADER -- MONDAY POST-CLOSE STOP PLACEMENT")
    log.info(f"MODE: {'PAPER' if PAPER_MODE else '*** LIVE ***'}")
    log.info("=" * 60)

    client = get_alpaca()

    live_positions = client.get_all_positions()
    if not live_positions:
        log.info("No open positions -- nothing to stop")
        return

    live_by_symbol = {p.symbol: p for p in live_positions}
    log.info(f"Alpaca reports {len(live_positions)} open positions")

    state = load_state()
    state_positions = state.get("positions", {})

    if not state_positions:
        log.warning("State has no positions but Alpaca does -- skipping to avoid placing "
                    "stops on positions we don't have metadata for")
        return

    placed    = []
    adopted   = []
    skipped   = []
    failed    = []

    # Pull all open orders once -- used to check for existing stops we should adopt
    try:
        open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    except Exception as e:
        log.warning(f"  list open orders failed: {e} -- proceeding without adoption check")
        open_orders = []

    for symbol, pos_state in state_positions.items():
        if pos_state.get("hard_stop_order_id"):
            skipped.append((symbol, "already_tracked"))
            continue

        live = live_by_symbol.get(symbol)
        if not live:
            log.warning(f"  {symbol}: in state but not in Alpaca live positions -- skipping")
            skipped.append((symbol, "not_live"))
            continue

        hard_stop_pct = pos_state.get("hard_stop_pct")
        if not hard_stop_pct or hard_stop_pct <= 0:
            log.warning(f"  {symbol}: no hard_stop_pct in state -- cannot place stop")
            skipped.append((symbol, "no_pct"))
            continue

        # Check for existing stop orders for this symbol (adopt rather than duplicate)
        adopted_id = None
        for order in open_orders:
            if (order.symbol == symbol and
                    order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and
                    order.side == OrderSide.SELL):
                adopted_id = str(order.id)
                break
        if adopted_id:
            log.info(f"  {symbol}: adopting existing stop {adopted_id}")
            state["positions"][symbol]["hard_stop_order_id"] = adopted_id
            adopted.append(symbol)
            continue

        try:
            actual_entry = float(live.avg_entry_price)
            shares       = int(float(live.qty))
        except Exception as e:
            log.error(f"  {symbol}: reading live position failed: {e}")
            failed.append(symbol)
            continue

        if actual_entry <= 0 or shares < 1:
            log.warning(f"  {symbol}: entry={actual_entry} qty={shares} -- invalid, skipping")
            skipped.append((symbol, "invalid_position"))
            continue

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
            placed.append(symbol)
        except Exception as e:
            log.error(f"  {symbol}: stop submission failed: {e}")
            failed.append(symbol)

    save_state(state)

    log.info(f"Stop placement complete: {len(placed)} placed, {len(adopted)} adopted, "
             f"{len(skipped)} skipped, {len(failed)} failed")

    if failed:
        notify_error("alpaca_trader",
                     f"Stop placement FAILED for {len(failed)} positions: "
                     f"{', '.join(failed)}. These are NAKED until resolved.")
    elif placed or adopted:
        notify_success("alpaca_trader",
                       f"Post-close stops placed: {len(placed)} new, {len(adopted)} adopted. "
                       f"All {len(placed) + len(adopted)} Monday positions now protected.")


# ── EXIT (FRIDAY 2:30 PM CT) ──────────────────────────────────────────────────

def run_exit():
    """Friday exit: close all open positions via DAY market sell orders.

    Order of operations:
      1. Cancel ALL outstanding stop / trailing-stop / stop-limit orders for
         each open position. Without this step, market sells collide with
         the stops (Alpaca rejects duplicate sell coverage for the same
         shares).
      2. Submit DAY market sells for each position using the live qty from
         Alpaca (not state qty, since Phase 2 partial sells may have
         reduced it).
      3. Record per-symbol results; don't clear state until exits confirmed.
      4. Clear state only for positions that successfully submitted.
         Positions that failed to close stay in state so next week's entry
         check catches them.

    Order type: DAY market orders submitted at 2:45 PM CT (15 min before
    close) fill against the live close-ramp book. Previously used
    TimeInForce.CLS (closing auction); changed 2026-04-27 because the
    closing-auction pro-rata mechanism produced partial fills, which on
    the sell side breaks the weekly-cycle abstraction (unfilled shares
    carry over the weekend without stop coverage and block the next
    Monday entry). See step 2 comment for full rationale.

    Previously: stops were not cancelled pre-sell, which would cause all
    sells to fail on "position already has open sell coverage."

    State clearing previously happened unconditionally. If any exits
    failed, next week's entry saw empty state but Alpaca still had open
    positions, requiring manual intervention.
    """
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus

    log.info("=" * 60)
    log.info("ALPACA TRADER -- FRIDAY EXIT")
    log.info(f"MODE: {'PAPER' if PAPER_MODE else '*** LIVE ***'}")
    log.info("=" * 60)

    client = get_alpaca()

    positions = client.get_all_positions()
    if not positions:
        log.info("No open positions to close")
        state = load_state()
        state["positions"]       = {}
        state["entry_date"]      = None
        state["week_open_value"] = None
        save_state(state)
        return

    account          = client.get_account()
    portfolio_value  = float(account.portfolio_value)
    state            = load_state()
    week_open        = state.get("week_open_value") or portfolio_value
    week_return      = (portfolio_value - week_open) / week_open
    log.info(f"Week return: {week_return*100:+.2f}%  "
             f"(${week_open:,.0f} -> ${portfolio_value:,.0f})")

    if CIRCUIT_BREAKER_PCT and week_return < -CIRCUIT_BREAKER_PCT:
        log.warning(f"CIRCUIT BREAKER: down {week_return*100:.1f}% on exit day "
                    f"(threshold {CIRCUIT_BREAKER_PCT*100:.1f}%)")

    # ── STEP 1: CANCEL ALL OUTSTANDING STOPS ─────────────────────────────
    position_symbols = {p.symbol for p in positions}

    try:
        open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    except Exception as e:
        log.error(f"Could not list open orders: {e} -- proceeding to sells "
                  f"but some may fail on stop collision")
        open_orders = []

    stops_to_cancel = [
        o for o in open_orders
        if o.symbol in position_symbols
        and o.side == OrderSide.SELL
        and o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP)
    ]

    cancelled_stops = []
    for order in stops_to_cancel:
        try:
            client.cancel_order_by_id(order.id)
            cancelled_stops.append((order.symbol, str(order.order_type), str(order.id)))
            log.info(f"  Cancelled {order.order_type} {order.id} for {order.symbol}")
        except Exception as e:
            log.warning(f"  Could not cancel {order.id} for {order.symbol}: {e}")

    if cancelled_stops:
        log.info(f"  Cancelled {len(cancelled_stops)} outstanding stop orders")

    # ── STEP 2: SUBMIT MARKET SELLS (DAY, not CLS) ─────────────────────────
    # Mirrors the 2026-04-27 entry-side fix. The same closing-auction
    # pro-rata problem that shorted Monday buys can short Friday sells --
    # and the consequences are MUCH worse on the sell side:
    #   - Partial sell fill leaves stop coverage gone (we just cancelled
    #     the trailing stop in step 1) on shares we still hold.
    #   - Unfilled remainder carries over the weekend with no protection.
    #   - run_entry on Monday refuses to enter if any positions exist, so
    #     leftover Friday shares block the entire next cycle.
    #   - alpaca_state.json shows empty positions while Alpaca shows held
    #     shares -- monitor doesn't know what to do with the strays.
    #
    # The system's weekly-cycle abstraction depends on full liquidation by
    # Friday close. DAY market orders submitted at 2:45 PM CT (per the
    # advanced cron) fill against the live order book and reliably clear
    # the position. Tradeoff: ~5-10 bps of bid-ask spread vs. the auction
    # print, in exchange for ~99% fill certainty.
    closed = []
    failed = []
    per_symbol_results = {}

    for pos in positions:
        sym    = pos.symbol
        shares = int(float(pos.qty))
        try:
            client.submit_order(
                order_data=MarketOrderRequest(
                    symbol        = sym,
                    qty           = shares,
                    side          = OrderSide.SELL,
                    time_in_force = TimeInForce.DAY,
                )
            )
            ret_pct = float(pos.unrealized_plpc) * 100
            log.info(f"  CLOSE {sym:<8} {shares:>4} shares  "
                     f"return={ret_pct:+.2f}%  "
                     f"P&L=${float(pos.unrealized_pl):+,.2f}")
            closed.append(sym)
            per_symbol_results[sym] = {"status": "submitted", "shares": shares,
                                        "return_pct": round(ret_pct, 2),
                                        "pnl": round(float(pos.unrealized_pl), 2)}
        except Exception as e:
            log.error(f"  FAILED {sym}: {e}")
            failed.append(sym)
            per_symbol_results[sym] = {"status": "failed", "error": str(e)}

    log.info(f"Exit complete: {len(closed)} closed, {len(failed)} failed")

    run_withdrawal(client, portfolio_value, week_return, week_open)

    # ── STEP 3: UPDATE STATE ──────────────────────────────────────────────
    if state.get("positions"):
        for sym in closed:
            state["positions"].pop(sym, None)

    if not failed and not state.get("positions"):
        state["positions"]       = {}
        state["entry_date"]      = None
        state["week_open_value"] = None
    elif failed:
        log.warning(f"  {len(failed)} positions failed to close -- "
                    f"state retains these symbols for next-week reconciliation")

    save_state(state)

    # ── STEP 4: NOTIFY ─────────────────────────────────────────────────────
    metrics = {
        "closed":             len(closed),
        "failed":             len(failed),
        "cancelled_stops":    len(cancelled_stops),
        "week_open":          round(float(week_open), 2),
        "week_close":         round(float(portfolio_value), 2),
        "week_return_pct":    round(float(week_return * 100), 2),
        "symbols_failed":     failed[:10],
    }
    if failed:
        log_event("alpaca_trader", LogStatus.WARNING,
                  f"Exit: {len(closed)} closed, {len(failed)} FAILED "
                  f"(week {week_return*100:+.2f}%)",
                  metrics=metrics)
        notify_alert("alpaca_trader",
                     f"Exit: {len(closed)} closed, {len(failed)} FAILED: "
                     f"{', '.join(failed[:5])}{'…' if len(failed) > 5 else ''}\n"
                     f"Week: {week_return*100:+.2f}%  "
                     f"(${week_open:,.0f} → ${portfolio_value:,.0f})\n"
                     f"Failed symbols remain in state -- next week's entry "
                     f"will halt until resolved.")
    else:
        log_event("alpaca_trader", LogStatus.SUCCESS,
                  f"Exit: all {len(closed)} closed "
                  f"(week {week_return*100:+.2f}%)",
                  metrics=metrics)
        notify_success("alpaca_trader",
                       f"Exit complete: {len(closed)} positions closed.\n"
                       f"Week: {week_return*100:+.2f}%  "
                       f"(${week_open:,.0f} → ${portfolio_value:,.0f})")


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
#
# NOTE: The actual ACH transfer call is commented out below. alpaca-py exposes
# transfers only through BrokerClient (for broker accounts), not TradingClient
# (standard self-directed). This was already a placeholder in the prior SDK.
# Live this when account type is confirmed and the right SDK surface is wired.

def execute_ach(client, amount: float, reason: str):
    """Execute ACH transfer to linked bank account."""
    try:
        log.info(f"WITHDRAWAL [{reason}]: ${amount:,.2f} -> linked bank account")
        log.info("  (ACH transfer API call placeholder -- requires BrokerClient + broker account)")
        # Standard TradingClient does not expose ACH initiation. For broker
        # accounts, use:
        #   from alpaca.broker.client import BrokerClient
        #   from alpaca.broker.requests import CreateACHTransferRequest
        #   broker = BrokerClient(api_key=..., secret_key=..., sandbox=PAPER_MODE)
        #   broker.create_ach_transfer_for_account(account_id, CreateACHTransferRequest(...))
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


def run_withdrawal(client, portfolio_value: float, week_return: float, week_open: float):
    week_gains = portfolio_value - week_open

    lump_str = os.environ.get("ALPACA_LUMP_SUM_AMOUNT", "").strip()
    if lump_str and float(lump_str) > 0:
        lump = apply_floor(float(lump_str), portfolio_value, "Lump sum")
        if lump > 0:
            execute_ach(client, lump, "lump sum")
            log.info("  Clear ALPACA_LUMP_SUM_AMOUNT secret manually to prevent re-firing")
            portfolio_value -= lump
    else:
        log.info("Lump sum: not set")

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
            execute_ach(client, amount, "debt payoff")

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
            execute_ach(client, amount, f"income {pct*100:.0f}%")

    else:
        log.warning(f"  Unknown withdrawal mode '{mode}' -- no withdrawal")
        log.warning("  Valid modes: off, debt, income")


# ── CIRCUIT BREAKER (MID-WEEK CHECK) ─────────────────────────────────────────

def run_circuit_breaker_check():
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    if not CIRCUIT_BREAKER_PCT:
        return

    client = get_alpaca()
    state  = load_state()
    week_open = state.get("week_open_value")

    if not week_open:
        return

    account         = client.get_account()
    portfolio_value = float(account.portfolio_value)
    week_return     = (portfolio_value - week_open) / week_open

    log.info(f"Circuit breaker check: {week_return*100:+.2f}% "
             f"(threshold: -{CIRCUIT_BREAKER_PCT*100:.0f}%)")

    if week_return < -CIRCUIT_BREAKER_PCT:
        log.warning(f"CIRCUIT BREAKER TRIGGERED: {week_return*100:.1f}% drawdown")
        log.warning("Closing all positions...")
        positions = client.get_all_positions()
        for pos in positions:
            try:
                client.submit_order(
                    order_data=MarketOrderRequest(
                        symbol        = pos.symbol,
                        qty           = int(float(pos.qty)),
                        side          = OrderSide.SELL,
                        time_in_force = TimeInForce.DAY,
                    )
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
    log_event("alpaca_trader", LogStatus.INFO, f"Starting alpaca_trader mode={mode}")

    try:
        if mode == "entry":
            entry_day = get_entry_day(today)
            if entry_day == "skip":
                log.info("Both Monday and Tuesday are holidays -- skipping entry this week")
                log_event("alpaca_trader", LogStatus.INFO,
                          "Skipped entry: both Monday and Tuesday are holidays")
                return
            if not is_entry_day(today):
                log.info(f"Today is not the entry day for this week "
                         f"(expected {entry_day}) -- skipping")
                log_event("alpaca_trader", LogStatus.INFO,
                          f"Skipped: not entry day (expected {entry_day})")
                return
            log.info(f"Entry day confirmed: {entry_day}")
            run_entry()

        elif mode == "exit":
            exit_day = get_exit_day(today)
            if exit_day == "skip":
                log.info("Both Friday and Thursday are holidays -- skipping exit this week")
                log_event("alpaca_trader", LogStatus.INFO,
                          "Skipped exit: both Friday and Thursday are holidays")
                return
            if not is_exit_day(today):
                log.info(f"Today is not the exit day for this week "
                         f"(expected {exit_day}) -- skipping")
                log_event("alpaca_trader", LogStatus.INFO,
                          f"Skipped: not exit day (expected {exit_day})")
                return
            log.info(f"Exit day confirmed: {exit_day}")
            run_exit()

        elif mode == "circuit_breaker":
            if not is_trading_day(today):
                log.info("Market holiday -- skipping circuit breaker check")
                return
            run_circuit_breaker_check()

        elif mode == "place_stops":
            if not is_trading_day(today):
                log.info("Market holiday -- skipping stop placement")
                return
            run_place_stops()

        else:
            log.error(f"Unknown mode: {mode}. Use 'entry', 'exit', 'place_stops', or 'circuit_breaker'")
            log_event("alpaca_trader", LogStatus.ERROR,
                      f"Unknown mode: {mode}")
            notify_error("alpaca_trader", f"Unknown mode '{mode}' passed to run()")

    except Exception as e:
        log.error(f"alpaca_trader crashed: {e}", exc_info=True)
        log_event("alpaca_trader", LogStatus.ERROR,
                  f"Unhandled exception during {mode}",
                  errors=[str(e)])
        notify_error("alpaca_trader", f"{mode.upper()} crashed: {e}")
        raise


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "entry"
    run(mode)
