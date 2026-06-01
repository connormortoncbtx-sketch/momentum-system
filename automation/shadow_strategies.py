"""
automation/shadow_strategies.py
================================
Runs "shadow" (challenger) strategy variants in parallel with the live
(champion) strategy. Each shadow proposes parameter overrides; the script
computes what that shadow would have selected/held/exited each week, scores
it against the same forward returns, and accumulates a divergence record.

The goal is rigorous A/B testing: a proposed change to the strategy
(longer hold periods, different stop widths, re-weighted signals) gets a
hypothetical track record over many weeks BEFORE being promoted to
production. This protects against acting on small samples or post-hoc
hypotheses ("the cadence analyzer suggests holding longer, so let's just
do it") that lead to overfitting.

How shadows are configured:
    config/shadows.json -- registry of active shadow variants. Each entry
    has a name, description, activation date, parameter overrides, and a
    pre-registered "promotion criteria" markdown reference. Max 3 active.

How shadows are evaluated:
    Each weekly run computes the shadow's hypothetical basket for each
    historical week (replaying perf_log data with the override applied).
    The output is appended to data/shadow_performance.csv with one row
    per (shadow, week) including return + bootstrap CI.

How shadows graduate:
    The script does NOT auto-promote. After a shadow has accumulated
    enough weeks (suggested >= 12), human review against pre-registered
    criteria decides whether to retire the shadow as inferior, keep it
    accumulating evidence, or promote it to production.

Currently supported shadow types:
    - "hold_through_rerank": hold positions through week N+1 if they
      re-rank in the top-N. Tests the cadence hypothesis from
      cadence_analyzer.

Future types (when supporting data exists):
    - "stop_width": different stop loss percentages (needs intraday)
    - "weight_override": different scoring weights (needs signal replay)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation.system_logger import log_event, LogStatus

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")

DATA_DIR        = Path("data")
CONFIG_DIR      = Path("config")
PERF_LOG        = DATA_DIR / "performance_log.csv"
SHADOWS_CONFIG  = CONFIG_DIR / "shadows.json"
SHADOW_OUTPUT   = DATA_DIR / "shadow_performance.csv"
SHADOW_SUMMARY  = DATA_DIR / "shadow_summary.csv"

MAX_ACTIVE_SHADOWS  = 3
TOP_N               = 10
BOOTSTRAP_ITERS     = 2000   # for 95% CI on aggregate metrics
CI_CONFIDENCE       = 0.95
MIN_WEEKS_FOR_STATS = 6      # below this, skip the t-test / CI computation


# ─── Shadow strategy implementations ──────────────────────────────────────────
#
# Each strategy function takes the perf_log + ordered_weeks list + entry_week +
# its param overrides and returns a list of (symbol, hypothetical_return) for
# that week's basket. The "champion" baseline is the same logic with default
# behavior (single-week hold, exit Friday close).


def champion_basket(perf: pd.DataFrame, entry_week: str, ordered_weeks: list,
                    top_n: int = TOP_N, rank_col: str = "composite_rank") -> list:
    """
    Champion behavior: top-N by composite_rank, hold Tue open -> Fri close.
    Returns [(symbol, return_pct), ...] for this week's basket.
    """
    week_df = perf[perf["week_of"] == entry_week].copy()
    week_df[rank_col] = pd.to_numeric(week_df[rank_col], errors="coerce")
    ranked = week_df.dropna(subset=[rank_col])
    if ranked.empty:
        return []
    top = ranked.nsmallest(top_n, rank_col)
    results = []
    for _, row in top.iterrows():
        sym = row["symbol"]
        ret = row.get("forward_return_1w")
        if pd.notna(ret):
            results.append((sym, float(ret)))
    return results


def shadow_hold_through_rerank(perf: pd.DataFrame, entry_week: str,
                                ordered_weeks: list, params: dict) -> list:
    """
    Hold-through-rerank shadow.

    Behavior: for each pick in week N's top-N, hold across weeks N, N+1, N+2,...
    as long as the same name still ranks in the top-N in each subsequent
    week. When it drops out, exit at that week's Friday close. Compounds
    the per-week returns.

    Comparison: against champion which exits every Friday and re-enters
    Monday only if still top-N.

    The point is to measure: if you'd just held through re-ranks instead
    of selling-and-rebuying, would you have ended up with more money?
    """
    top_n = params.get("top_n", TOP_N)
    rank_col = params.get("rank_col", "composite_rank")
    max_extension_weeks = params.get("max_extension_weeks", 4)

    # Get this week's picks
    week_df = perf[perf["week_of"] == entry_week].copy()
    week_df[rank_col] = pd.to_numeric(week_df[rank_col], errors="coerce")
    ranked = week_df.dropna(subset=[rank_col])
    if ranked.empty:
        return []
    top = ranked.nsmallest(top_n, rank_col)
    symbols = top["symbol"].tolist()

    try:
        entry_idx = ordered_weeks.index(entry_week)
    except ValueError:
        return []

    results = []
    for sym in symbols:
        compounded = 1.0
        last_week_for_sym = entry_week
        for offset in range(max_extension_weeks + 1):
            current_idx = entry_idx + offset
            if current_idx >= len(ordered_weeks):
                break
            current_week = ordered_weeks[current_idx]

            # Get this week's return for this symbol
            sym_row = perf[(perf["week_of"] == current_week) &
                           (perf["symbol"] == sym)]
            if sym_row.empty:
                break
            r = sym_row["forward_return_1w"].iloc[0]
            if pd.isna(r):
                break
            compounded *= (1 + float(r))
            last_week_for_sym = current_week

            # Check if symbol still in top-N for next week to decide whether
            # to keep holding. If no next-week data exists, exit.
            next_idx = current_idx + 1
            if next_idx >= len(ordered_weeks):
                break
            next_week = ordered_weeks[next_idx]
            next_df = perf[perf["week_of"] == next_week].copy()
            next_df[rank_col] = pd.to_numeric(next_df[rank_col], errors="coerce")
            next_ranked = next_df.dropna(subset=[rank_col])
            if next_ranked.empty:
                break
            next_top = next_ranked.nsmallest(top_n, rank_col)
            if sym not in next_top["symbol"].values:
                break  # dropped out -- exit at this week's close

        total_return = compounded - 1
        results.append((sym, float(total_return)))

    return results


# Registry of available shadow types -> implementation
SHADOW_IMPLEMENTATIONS = {
    "hold_through_rerank": shadow_hold_through_rerank,
}


# ─── Config loading and validation ────────────────────────────────────────────


def load_shadows_config() -> dict:
    """
    Load config/shadows.json. Returns empty dict if file is missing
    (zero active shadows -- the plumbing still runs cleanly).
    """
    if not SHADOWS_CONFIG.exists():
        log.info(f"No shadows config at {SHADOWS_CONFIG} -- zero active shadows")
        return {}
    try:
        with open(SHADOWS_CONFIG) as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        log.error(f"Could not parse {SHADOWS_CONFIG}: {e}")
        return {}


def validate_shadows_config(config: dict) -> dict:
    """
    Filter the config to only active, valid shadows.

    Validation:
      - "active": true (skip otherwise)
      - "type" is in SHADOW_IMPLEMENTATIONS
      - At most MAX_ACTIVE_SHADOWS pass validation
    """
    valid = {}
    for name, spec in config.items():
        if not spec.get("active", False):
            log.info(f"Shadow '{name}': inactive, skipping")
            continue
        if spec.get("type") not in SHADOW_IMPLEMENTATIONS:
            log.warning(f"Shadow '{name}': unknown type '{spec.get('type')}' -- skipping")
            continue
        valid[name] = spec
        if len(valid) >= MAX_ACTIVE_SHADOWS:
            log.warning(f"Hit MAX_ACTIVE_SHADOWS={MAX_ACTIVE_SHADOWS}; "
                        f"ignoring any further shadows in config")
            break
    return valid


# ─── Statistics helpers ───────────────────────────────────────────────────────


def bootstrap_mean_ci(values: list, iters: int = BOOTSTRAP_ITERS,
                      confidence: float = CI_CONFIDENCE) -> tuple:
    """
    Bootstrap CI for the mean of a list of values.

    Returns (mean, ci_low, ci_high) in raw units (not percent).
    Returns (mean, None, None) if too few samples for stable CI.
    """
    arr = np.array([v for v in values if pd.notna(v)])
    if len(arr) < 2:
        return (float(arr.mean()) if len(arr) else None, None, None)

    rng = np.random.default_rng(seed=42)
    boot_means = []
    n = len(arr)
    for _ in range(iters):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)

    alpha = 1 - confidence
    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return (float(arr.mean()), float(lo), float(hi))


def paired_t_pvalue(deltas: list) -> float | None:
    """
    Simple paired t-test on the per-week delta series (shadow - champion).
    Returns p-value (two-sided) or None if too few samples.

    We use a basic implementation rather than scipy.stats to avoid an
    extra dependency. The result is "approximately t-test" -- close enough
    for the precision we need at this stage.
    """
    arr = np.array([v for v in deltas if pd.notna(v)])
    n = len(arr)
    if n < MIN_WEEKS_FOR_STATS:
        return None
    mean = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0 if mean != 0 else 1.0
    t = mean / (sd / np.sqrt(n))
    # Survival function of the t-distribution -- normal approximation OK
    # for n >= 6 and not pursuing tight precision here.
    from math import erf, sqrt
    # Use normal approximation: p ~ 2 * (1 - Phi(|t|))
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return float(p)


# ─── Per-week scoring ─────────────────────────────────────────────────────────


def score_basket(picks: list) -> float | None:
    """
    Mean return across picks. picks is [(symbol, return_pct), ...].
    Returns None if empty.
    """
    if not picks:
        return None
    returns = [r for _, r in picks if pd.notna(r)]
    if not returns:
        return None
    return float(np.mean(returns))


def evaluate_shadow_for_week(perf: pd.DataFrame, shadow_name: str, spec: dict,
                              entry_week: str, ordered_weeks: list) -> dict:
    """
    Compute champion and shadow performance for one entry week.
    """
    impl = SHADOW_IMPLEMENTATIONS[spec["type"]]
    params = spec.get("params", {})

    champ_picks  = champion_basket(perf, entry_week, ordered_weeks,
                                    top_n=params.get("top_n", TOP_N),
                                    rank_col=params.get("rank_col", "composite_rank"))
    shadow_picks = impl(perf, entry_week, ordered_weeks, params)

    champ_return  = score_basket(champ_picks)
    shadow_return = score_basket(shadow_picks)

    delta = None
    if champ_return is not None and shadow_return is not None:
        delta = shadow_return - champ_return

    return {
        "shadow":           shadow_name,
        "week_of":          entry_week,
        "n_picks_champion": len(champ_picks),
        "n_picks_shadow":   len(shadow_picks),
        "champion_return":  round(champ_return, 5)  if champ_return  is not None else None,
        "shadow_return":    round(shadow_return, 5) if shadow_return is not None else None,
        "delta":            round(delta, 5)         if delta         is not None else None,
    }


# ─── Main run ─────────────────────────────────────────────────────────────────


def load_perf_log() -> pd.DataFrame:
    """Load the perf log and return ordered by week."""
    if not PERF_LOG.exists():
        raise FileNotFoundError(f"{PERF_LOG} does not exist -- run collect_returns first")
    df = pd.read_csv(PERF_LOG, low_memory=False)
    df["composite_rank"] = pd.to_numeric(df["composite_rank"], errors="coerce")
    df = df.sort_values("week_of").reset_index(drop=True)
    return df


def get_analyzable_weeks(perf: pd.DataFrame, shadow_spec: dict) -> list:
    """
    Determine which weeks this shadow can be evaluated for.

    A week is analyzable if:
      - It has rows in perf_log with non-null rank_col
      - It's on or after the shadow's "active_since" date

    Note: we deliberately do NOT require max_extension_weeks of forward
    data. The shadow's implementation handles running out of forward
    data gracefully (it exits at whatever the last available week is,
    same as the champion). Excluding weeks for lack of forward data
    would prevent the shadow from being evaluated on its earliest weeks
    until the data fully catches up -- which never happens for a
    recently-added shadow.

    The tradeoff: shadows evaluated near the edge of available data will
    have fewer "extension" opportunities, so the comparison is closer to
    1-week vs 1-week (champion vs champion) for those weeks. That's
    accurate behavior, not a bug -- it just means the shadow's distinct
    behavior only becomes visible once enough forward weeks accumulate.
    """
    rank_col = shadow_spec.get("params", {}).get("rank_col", "composite_rank")
    active_since = shadow_spec.get("active_since")

    weeks = sorted(perf["week_of"].unique().tolist())

    # Filter to weeks with rank data
    weeks_with_ranks = []
    for w in weeks:
        wdf = perf[perf["week_of"] == w]
        if wdf[rank_col].notna().any():
            weeks_with_ranks.append(w)

    # Filter to weeks on/after active_since
    if active_since:
        weeks_with_ranks = [w for w in weeks_with_ranks if w >= active_since]

    return weeks_with_ranks


def run():
    log.info("=" * 60)
    log.info("  SHADOW STRATEGIES")
    log.info("=" * 60)
    log_event("shadow_strategies", LogStatus.INFO, "Starting shadow evaluation")

    config = load_shadows_config()
    active = validate_shadows_config(config)

    if not active:
        log.info("No active shadows configured -- nothing to evaluate")
        log_event("shadow_strategies", LogStatus.INFO,
                  "No active shadows -- skipping",
                  metrics={"config_entries": len(config), "active": 0})
        # Still write empty outputs to confirm plumbing ran
        return

    log.info(f"Active shadows: {list(active.keys())}")

    try:
        perf = load_perf_log()
    except FileNotFoundError as e:
        log.warning(str(e))
        log_event("shadow_strategies", LogStatus.WARNING,
                  "Skipped: perf log missing")
        return

    ordered_weeks = sorted(perf["week_of"].unique().tolist())

    # ── Evaluate each shadow across all eligible weeks ──────────────────
    all_rows = []
    for shadow_name, spec in active.items():
        log.info(f"\n--- Evaluating shadow: {shadow_name} ---")
        log.info(f"    Type:         {spec.get('type')}")
        log.info(f"    Active since: {spec.get('active_since', '(not set)')}")
        log.info(f"    Description:  {spec.get('description', '(no description)')}")

        analyzable = get_analyzable_weeks(perf, spec)
        log.info(f"    Analyzable weeks: {len(analyzable)}")
        if not analyzable:
            log.info(f"    No analyzable weeks yet -- need more forward data")
            continue

        for entry_week in analyzable:
            row = evaluate_shadow_for_week(perf, shadow_name, spec,
                                            entry_week, ordered_weeks)
            all_rows.append(row)

    # ── Write raw per-week results ──────────────────────────────────────
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(SHADOW_OUTPUT, index=False)
        log.info(f"\nWrote {len(df)} rows to {SHADOW_OUTPUT}")
    else:
        log.info("\nNo evaluatable rows produced")
        log_event("shadow_strategies", LogStatus.INFO,
                  "No rows produced (likely no analyzable weeks yet)")
        return

    # ── Compute per-shadow summary with bootstrap CIs ───────────────────
    summary_rows = []
    for shadow_name in active.keys():
        shadow_rows = df[df["shadow"] == shadow_name]
        if shadow_rows.empty:
            continue

        n_weeks = len(shadow_rows)
        champion_returns = shadow_rows["champion_return"].dropna().tolist()
        shadow_returns   = shadow_rows["shadow_return"].dropna().tolist()
        deltas           = shadow_rows["delta"].dropna().tolist()

        champ_mean, champ_lo, champ_hi = bootstrap_mean_ci(champion_returns)
        shadow_mean, shadow_lo, shadow_hi = bootstrap_mean_ci(shadow_returns)
        delta_mean, delta_lo, delta_hi   = bootstrap_mean_ci(deltas)
        p_value = paired_t_pvalue(deltas)

        # Win rate (shadow > champion in how many weeks)
        wins = sum(1 for d in deltas if d > 0)
        win_rate = wins / len(deltas) if deltas else None

        summary_rows.append({
            "shadow":              shadow_name,
            "type":                active[shadow_name].get("type"),
            "n_weeks":             n_weeks,

            "champion_mean":       round(champ_mean, 5)  if champ_mean  is not None else None,
            "champion_ci_low":     round(champ_lo, 5)    if champ_lo    is not None else None,
            "champion_ci_high":    round(champ_hi, 5)    if champ_hi    is not None else None,

            "shadow_mean":         round(shadow_mean, 5) if shadow_mean is not None else None,
            "shadow_ci_low":       round(shadow_lo, 5)   if shadow_lo   is not None else None,
            "shadow_ci_high":      round(shadow_hi, 5)   if shadow_hi   is not None else None,

            "delta_mean":          round(delta_mean, 5)  if delta_mean  is not None else None,
            "delta_ci_low":        round(delta_lo, 5)    if delta_lo    is not None else None,
            "delta_ci_high":       round(delta_hi, 5)    if delta_hi    is not None else None,

            "win_rate":            round(win_rate, 3)    if win_rate    is not None else None,
            "p_value_paired":      round(p_value, 4)     if p_value     is not None else None,
        })

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(SHADOW_SUMMARY, index=False)
        log.info(f"Wrote summary for {len(sdf)} shadows to {SHADOW_SUMMARY}")

        log.info("\n" + "─" * 60)
        log.info("SHADOW PERFORMANCE SUMMARY")
        log.info("─" * 60)
        for _, r in sdf.iterrows():
            log.info(f"\n  {r['shadow']} ({r['type']}, n={r['n_weeks']} weeks):")
            ch_m  = r['champion_mean']
            sh_m  = r['shadow_mean']
            d_m   = r['delta_mean']
            d_lo  = r['delta_ci_low']
            d_hi  = r['delta_ci_high']
            pv    = r['p_value_paired']
            wr    = r['win_rate']

            log.info(f"    Champion mean: {ch_m*100:+.2f}%" if ch_m is not None else "    Champion mean: N/A")
            log.info(f"    Shadow mean:   {sh_m*100:+.2f}%" if sh_m is not None else "    Shadow mean:   N/A")
            if d_m is not None:
                log.info(f"    Delta:         {d_m*100:+.2f}% per week")
                if d_lo is not None and d_hi is not None:
                    log.info(f"    95% CI:        [{d_lo*100:+.2f}%, {d_hi*100:+.2f}%]")
            if wr is not None:
                log.info(f"    Win rate:      {wr*100:.1f}% ({int(wr*r['n_weeks'])}/{r['n_weeks']} weeks)")
            if pv is not None:
                significant = "yes" if pv < 0.05 else "no"
                log.info(f"    p-value:       {pv:.4f} (sig at 0.05: {significant})")
            else:
                log.info(f"    p-value:       N/A (need >= {MIN_WEEKS_FOR_STATS} weeks)")
            log.info(f"    NOTE: Not a recommendation. Pre-registered criteria must be met "
                     f"before promoting to production.")

    log_event("shadow_strategies", LogStatus.SUCCESS,
              f"Evaluated {len(active)} shadows across {len(all_rows)} (shadow, week) pairs",
              metrics={
                  "active_shadows": len(active),
                  "rows_written":   len(all_rows),
              })


if __name__ == "__main__":
    run()
