# automation/shadow_phase2.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 "full ride vs partial sell" shadow tracker.
#
# Question this answers
# ---------------------
# At Phase 2 activation, the live system sells a fraction f of the position at
# ~the activation price and trails the remaining (1 - f). The counterfactual is
# to NOT sell the partial and trail the FULL position instead.
#
# Because the trail fires at the same price P_x whether it covers (1-f)·N or N
# shares, the EV difference between the two strategies collapses to a single
# per-position quantity:
#
#       full_ride_return - partial_sell_return = f · (r_exit - r_act)
#
#   where  r_act  = (partial_fill_price / entry_price) - 1   (return at the partial sell)
#          r_exit = (exit_price        / entry_price) - 1    (return at final exit of remnant)
#          f      = realized partial-sell fraction (partial_sell_pct_actual)
#
# Positive delta  => full ride would have done better (name kept climbing past
#                    activation).
# Negative delta  => the partial sell was the right call (name gave back gains).
#
# This module does NOT fork execution. It only OBSERVES the prices the live
# system already transacts at. Two append-only event types are written to a
# JSONL ledger and joined at analysis time by the entry order_id:
#
#   {"event": "activation", "position_key": <entry order_id>, ...}
#   {"event": "exit",       "position_key": <entry order_id>, ...}
#
# Append-only + join-at-read is deliberate: it never mutates a shared record,
# so it merges cleanly under the workflows' `-X theirs` JSONL conflict policy
# even when the monitor and the Friday exit commit within seconds.
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import datetime
from pathlib import Path
from statistics import mean, median, pstdev

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
LEDGER   = DATA_DIR / "shadow_phase2_fullride.jsonl"


# ── WRITE PATH (called from live code, must never raise) ─────────────────────

def _append(record: dict) -> None:
    """Append one event to the ledger. Best-effort: never raises into caller."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        record.setdefault("ts", datetime.datetime.now().isoformat())
        with open(LEDGER, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:  # pragma: no cover - logging only
        log.warning(f"shadow_phase2: failed to append {record.get('event')} "
                    f"for {record.get('symbol')}: {e}")


def record_activation(*, symbol: str, position_key: str, entry_price: float,
                      activation_price: float, partial_fill_price: float,
                      partial_filled: bool, partial_sell_pct: float,
                      total_qty: int, sell_qty: int, trail_qty: int,
                      trail_pct: float, alpha_score: float,
                      weekly_vol: float) -> None:
    """Log a Phase 2 activation. Call once, right after the upgrade succeeds.

    `position_key` should be the ENTRY order_id (state["positions"][sym]["order_id"]),
    which uniquely identifies the position even if the same symbol recurs in a
    later week under the cross-week carry model.

    `partial_fill_price` is the actual fill of the partial sell when available;
    pass the activation/trigger price as a fallback. When partial_filled is
    False, f is treated as 0 (full ride == partial, delta == 0) and the fill
    price is irrelevant.
    """
    _append({
        "event":            "activation",
        "symbol":           symbol,
        "position_key":     str(position_key),
        "entry_price":      round(float(entry_price), 4),
        "activation_price": round(float(activation_price), 4),
        "partial_fill_price": round(float(partial_fill_price), 4),
        "partial_filled":   bool(partial_filled),
        "partial_sell_pct": round(float(partial_sell_pct), 4) if partial_filled else 0.0,
        "total_qty":        int(total_qty),
        "sell_qty":         int(sell_qty) if partial_filled else 0,
        "trail_qty":        int(trail_qty),
        "trail_pct":        float(trail_pct),
        "alpha_score":      round(float(alpha_score), 4),
        "weekly_vol":       round(float(weekly_vol), 4),
    })


def record_exit(*, symbol: str, position_key: str, exit_price: float,
                exit_reason: str) -> None:
    """Log the final exit of the trailed remnant. Call once when the position
    fully leaves the book.

    exit_reason: 'trail' (trailing stop fired mid-week) or 'friday_moc'
    (survived to the Friday close-out).
    """
    _append({
        "event":        "exit",
        "symbol":       symbol,
        "position_key": str(position_key),
        "exit_price":   round(float(exit_price), 4),
        "exit_reason":  exit_reason,
    })


# ── READ / ANALYSIS PATH ─────────────────────────────────────────────────────

def _load_events() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning(f"shadow_phase2: skipping malformed ledger line")
    return out


def build_pairs(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Join activation and exit events by position_key.

    Returns (matched, unmatched_activations). Defensive against duplicate
    exits (takes the earliest exit per key) and exits with no activation
    (ignored — can't compute a delta without an entry/activation basis).
    """
    activations: dict[str, dict] = {}
    exits: dict[str, dict] = {}
    for e in events:
        key = e.get("position_key")
        if not key:
            continue
        if e.get("event") == "activation":
            activations.setdefault(key, e)          # first activation wins
        elif e.get("event") == "exit":
            exits.setdefault(key, e)                # first (earliest) exit wins

    matched, unmatched = [], []
    for key, act in activations.items():
        ex = exits.get(key)
        if ex is None:
            unmatched.append(act)
            continue

        entry = act["entry_price"]
        if not entry or entry <= 0:
            continue

        f      = float(act.get("partial_sell_pct", 0.0))
        p_act  = float(act.get("partial_fill_price") or act.get("activation_price") or entry)
        p_exit = float(ex["exit_price"])

        r_act  = (p_act / entry) - 1.0
        r_exit = (p_exit / entry) - 1.0
        delta  = f * (r_exit - r_act)            # full_ride - partial_sell

        notional = float(act.get("total_qty", 0)) * entry
        matched.append({
            "symbol":       act["symbol"],
            "position_key": key,
            "f":            f,
            "r_act":        r_act,
            "r_exit":       r_exit,
            "delta":        delta,             # >0: full ride wins
            "partial_ret":  f * r_act + (1 - f) * r_exit,
            "full_ret":     r_exit,
            "notional":     notional,
            "dollar_delta": notional * delta,  # $ full ride would have added/lost
            "exit_reason":  ex.get("exit_reason", "?"),
            "partial_filled": act.get("partial_filled", True),
            "alpha_score":  act.get("alpha_score"),
            "weekly_vol":   act.get("weekly_vol"),
        })

    return matched, unmatched


def summarize(matched: list[dict], unmatched: list[dict]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 2 SHADOW — full ride vs partial sell")
    lines.append("  delta = f·(r_exit − r_act);  positive => FULL RIDE would win")
    lines.append("=" * 72)

    n = len(matched)
    if n == 0:
        lines.append(f"No matched activation/exit pairs yet "
                     f"({len(unmatched)} activation(s) still open).")
        return "\n".join(lines)

    deltas   = [m["delta"] for m in matched]
    partials = [m["partial_ret"] for m in matched]
    fulls    = [m["full_ret"] for m in matched]
    tot_notional = sum(m["notional"] for m in matched) or 1.0
    wins = sum(1 for d in deltas if d > 0)

    lines.append(f"\nSample: {n} matched activation(s), "
                 f"{len(unmatched)} still open.")
    lines.append("")
    lines.append(f"  Mean delta (equal-weight): {mean(deltas)*100:+.3f}%")
    lines.append(f"  Median delta:              {median(deltas)*100:+.3f}%")
    lines.append(f"  Notional-weighted delta:   "
                 f"{sum(m['dollar_delta'] for m in matched)/tot_notional*100:+.3f}%")
    lines.append(f"  Total $ impact of full ride: "
                 f"${sum(m['dollar_delta'] for m in matched):+,.2f}")
    lines.append(f"  Full ride better in:       {wins}/{n} "
                 f"({wins/n*100:.0f}% of activations)")
    lines.append("")
    lines.append(f"  Mean position return  — partial: {mean(partials)*100:+.2f}%   "
                 f"full: {mean(fulls)*100:+.2f}%")
    if n > 1:
        lines.append(f"  Std of position return — partial: {pstdev(partials)*100:.2f}%   "
                     f"full: {pstdev(fulls)*100:.2f}%")
    lines.append("")
    lines.append(f"  {'symbol':<8} {'f':>5} {'r_act':>8} {'r_exit':>8} "
                 f"{'delta':>8}  exit")
    lines.append(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
    for m in sorted(matched, key=lambda x: x["delta"]):
        lines.append(f"  {m['symbol']:<8} {m['f']:>5.2f} "
                     f"{m['r_act']*100:>+7.1f}% {m['r_exit']*100:>+7.1f}% "
                     f"{m['delta']*100:>+7.2f}%  {m['exit_reason']}")

    if unmatched:
        lines.append("")
        lines.append(f"  Open (no exit recorded yet): "
                     f"{', '.join(sorted(u['symbol'] for u in unmatched))}")

    lines.append("")
    lines.append("  Caveats: counterfactual assumes the trail fires at the same")
    lines.append("  price on a full vs reduced share count (true for paper / liquid")
    lines.append("  names). Earnings-gap exits predate the exclusion patch and")
    lines.append("  should be filtered before drawing conclusions.")
    return "\n".join(lines)


def main() -> None:
    events = _load_events()
    matched, unmatched = build_pairs(events)
    print(summarize(matched, unmatched))


if __name__ == "__main__":
    main()
