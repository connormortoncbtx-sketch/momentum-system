"""
automation/cadence_analyzer.py
==============================
Studies whether the weekly cadence is the right cadence by computing what
returns would have looked like with longer hold periods.

Run weekly as part of the learning loop. Each run produces one row per
historical week (where at least 3 weeks of subsequent data exist) and
writes to data/cadence_analysis.csv.

For each week N's top-10 picks (by composite_rank), the script computes:

  Actual capture:
    Tuesday open of week N -> Friday close of week N (forward_return_1w)
    This is the strategy's actual return contribution.

  Counterfactual captures (held the same names longer):
    Tuesday open of week N -> Friday close of week N+1 (2-week hold)
    Tuesday open of week N -> Friday close of week N+2 (3-week hold)
    Tuesday open of week N -> Friday close of week N+3 (4-week hold)

  Peak-capture ceilings:
    Tuesday open of week N -> max(weekly_high) over weeks N..N+2

  Persistence:
    What fraction of week N's top-10 are still in top-10 in N+1, N+2, N+3?

  Roundtrip cost (when re-bought):
    For names that re-rank in week N+1, the gap between week N Friday close
    and week N+1 Tuesday open. This is the friction you pay to exit-and-rebuy
    vs. holding through.

The script does NOT recommend a cadence change. It just produces numbers.
After 8-10 weeks of accumulation, the numbers will tell you whether the
hypothesis ("the 5-day cadence is too short for the signal lifespan") is
supported.

Methodology notes:
  - Top-10 by composite_rank (reflects operational reality, not pure model)
  - All metrics computed counterfactually on the SAME set of names per week
    (no survivorship: a name picked in week N is followed through N+3
    regardless of whether it later re-ranks)
  - Idempotent recompute: each run rebuilds the full cadence_analysis.csv
    from scratch rather than appending. Cheap (the perf log is small) and
    means tweaks to the analysis logic propagate to historical rows.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation.system_logger import log_event, LogStatus
from automation.notifier import notify_alert

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")

DATA_DIR    = Path("data")
PERF_LOG    = DATA_DIR / "performance_log.csv"
OUTPUT_CSV  = DATA_DIR / "cadence_analysis.csv"

# How many weeks of forward data are required before we'll compute metrics
# for a given week N. Setting this to 3 means N+1, N+2, N+3 all need to exist.
MIN_FORWARD_WEEKS = 3

# Top-N picks to analyze. 10 matches the operational basket size.
TOP_N = 10

# Which rank column to use for top-N selection.
#   "composite_rank" -- reflects operational reality (what you actually
#                       would have traded under the current ranking).
#                       Only populated from 2026-04-17 onward in the
#                       current dataset.
#   "alpha_rank"     -- the pure model score, populated throughout history.
#                       Use to study the signal itself rather than the
#                       traded basket.
#
# Default is composite_rank (operational). Override per-run via --rank-col.
DEFAULT_RANK_COL = "composite_rank"


def load_perf_log() -> pd.DataFrame:
    """Load and lightly clean the perf log."""
    if not PERF_LOG.exists():
        raise FileNotFoundError(f"{PERF_LOG} does not exist -- run collect_returns first")

    df = pd.read_csv(PERF_LOG, low_memory=False)

    # Coerce rank columns to numeric (mixed dtypes from pre-fix rows can cause
    # object dtype, which breaks sorting/ranking downstream).
    for col in ["composite_rank", "alpha_rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort weeks ascending for predictable iteration.
    df = df.sort_values(["week_of"]).reset_index(drop=True)
    return df


def get_week_data(perf: pd.DataFrame, week_of: str) -> pd.DataFrame:
    """Slice the perf log to one week's rows."""
    return perf[perf["week_of"] == week_of].copy()


def get_top_n_for_week(perf: pd.DataFrame, week_of: str, rank_col: str,
                        n: int = TOP_N) -> pd.DataFrame:
    """Return the top-N rows for a given week, sorted by the rank column ascending."""
    week_df = get_week_data(perf, week_of)
    ranked = week_df.dropna(subset=[rank_col])
    return ranked.nsmallest(n, rank_col)


def compute_held_longer_return(perf: pd.DataFrame, symbol: str,
                                entry_week: str, n_weeks_forward: int,
                                ordered_weeks: list) -> float | None:
    """
    Compute return from entry_week's Tuesday open through the close of the
    week N+n_weeks_forward (inclusive of intermediate weeks).

    Returns None if data is missing.
    """
    # Find entry price: Tuesday open from entry_week
    entry_rows = perf[(perf["week_of"] == entry_week) & (perf["symbol"] == symbol)]
    if entry_rows.empty:
        return None
    tue_open = entry_rows["tue_open"].iloc[0]
    if pd.isna(tue_open) or tue_open <= 0:
        return None

    # Find the target week (entry_week + n_weeks_forward in the ordered list)
    try:
        entry_idx = ordered_weeks.index(entry_week)
    except ValueError:
        return None
    target_idx = entry_idx + n_weeks_forward
    if target_idx >= len(ordered_weeks):
        return None
    target_week = ordered_weeks[target_idx]

    # Find exit price: Friday close of target week
    exit_rows = perf[(perf["week_of"] == target_week) & (perf["symbol"] == symbol)]
    if exit_rows.empty:
        return None
    fri_close = exit_rows["fri_close"].iloc[0]
    if pd.isna(fri_close) or fri_close <= 0:
        return None

    return float(fri_close / tue_open - 1)


def compute_peak_over_window(perf: pd.DataFrame, symbol: str,
                              entry_week: str, n_weeks_forward: int,
                              ordered_weeks: list) -> float | None:
    """
    Best possible exit from entry_week's Tuesday open through the highest
    weekly_high observed in any of weeks N..N+n_weeks_forward inclusive.

    Returns None if data is missing.
    """
    entry_rows = perf[(perf["week_of"] == entry_week) & (perf["symbol"] == symbol)]
    if entry_rows.empty:
        return None
    tue_open = entry_rows["tue_open"].iloc[0]
    if pd.isna(tue_open) or tue_open <= 0:
        return None

    try:
        entry_idx = ordered_weeks.index(entry_week)
    except ValueError:
        return None

    highs = []
    for offset in range(n_weeks_forward + 1):
        target_idx = entry_idx + offset
        if target_idx >= len(ordered_weeks):
            return None
        week = ordered_weeks[target_idx]
        rows = perf[(perf["week_of"] == week) & (perf["symbol"] == symbol)]
        if rows.empty:
            continue
        h = rows["weekly_high"].iloc[0]
        if pd.notna(h) and h > 0:
            highs.append(float(h))

    if not highs:
        return None
    peak = max(highs)
    return float(peak / tue_open - 1)


def compute_roundtrip_cost(perf: pd.DataFrame, symbol: str,
                            entry_week: str, ordered_weeks: list) -> float | None:
    """
    If a name was picked in entry_week AND re-ranks in entry_week+1, compute
    the round-trip cost: the gap between exit_week's Friday close and the
    next week's Tuesday open. A negative number means you paid a gap to
    re-enter (price rose over the weekend).
    """
    try:
        entry_idx = ordered_weeks.index(entry_week)
    except ValueError:
        return None
    next_idx = entry_idx + 1
    if next_idx >= len(ordered_weeks):
        return None
    next_week = ordered_weeks[next_idx]

    exit_rows = perf[(perf["week_of"] == entry_week) & (perf["symbol"] == symbol)]
    next_rows = perf[(perf["week_of"] == next_week) & (perf["symbol"] == symbol)]
    if exit_rows.empty or next_rows.empty:
        return None

    fri_close   = exit_rows["fri_close"].iloc[0]
    next_tue    = next_rows["tue_open"].iloc[0]
    if pd.isna(fri_close) or pd.isna(next_tue) or fri_close <= 0:
        return None

    # Negative means re-entry costs you (price rose between exit and re-entry).
    # Positive means re-entry was cheaper.
    return float(fri_close / next_tue - 1)


def analyze_week(perf: pd.DataFrame, entry_week: str, ordered_weeks: list,
                 rank_col: str) -> dict:
    """
    Compute all cadence metrics for one entry week.

    Returns a dict of metrics suitable for appending as a row in the output.
    """
    top_picks = get_top_n_for_week(perf, entry_week, rank_col, n=TOP_N)
    if top_picks.empty:
        return None

    symbols = top_picks["symbol"].tolist()

    # ── Capture metrics: actual + counterfactual ────────────────────────
    captures_1wk = []   # actual: Tue open week N -> Fri close week N
    captures_2wk = []   # held through week N+1
    captures_3wk = []   # held through week N+2
    captures_4wk = []   # held through week N+3
    peak_3wk     = []   # best peak in weeks N..N+2

    for sym in symbols:
        c1 = compute_held_longer_return(perf, sym, entry_week, 0, ordered_weeks)
        c2 = compute_held_longer_return(perf, sym, entry_week, 1, ordered_weeks)
        c3 = compute_held_longer_return(perf, sym, entry_week, 2, ordered_weeks)
        c4 = compute_held_longer_return(perf, sym, entry_week, 3, ordered_weeks)
        p3 = compute_peak_over_window(perf, sym, entry_week, 2, ordered_weeks)
        if c1 is not None: captures_1wk.append(c1)
        if c2 is not None: captures_2wk.append(c2)
        if c3 is not None: captures_3wk.append(c3)
        if c4 is not None: captures_4wk.append(c4)
        if p3 is not None: peak_3wk.append(p3)

    # ── Persistence metrics ─────────────────────────────────────────────
    entry_idx = ordered_weeks.index(entry_week)
    n_in_top10_next  = 0
    n_in_top10_two   = 0
    n_in_top10_three = 0

    for offset, counter in [(1, "next"), (2, "two"), (3, "three")]:
        target_idx = entry_idx + offset
        if target_idx >= len(ordered_weeks):
            continue
        target_week = ordered_weeks[target_idx]
        target_top = get_top_n_for_week(perf, target_week, rank_col, n=TOP_N)
        if target_top.empty:
            continue
        target_symbols = set(target_top["symbol"].tolist())
        overlap = len(set(symbols) & target_symbols)
        if counter == "next":  n_in_top10_next  = overlap
        if counter == "two":   n_in_top10_two   = overlap
        if counter == "three": n_in_top10_three = overlap

    # ── Roundtrip cost (for re-ranked names) ────────────────────────────
    next_idx = entry_idx + 1
    roundtrip_costs = []
    if next_idx < len(ordered_weeks):
        next_week_top = get_top_n_for_week(perf, ordered_weeks[next_idx], rank_col, n=TOP_N)
        rerank_symbols = set(symbols) & set(next_week_top["symbol"].tolist())
        for sym in rerank_symbols:
            rc = compute_roundtrip_cost(perf, sym, entry_week, ordered_weeks)
            if rc is not None:
                roundtrip_costs.append(rc)

    def safe_mean(xs):
        return round(float(np.mean(xs)), 5) if xs else None

    return {
        "week_of":                   entry_week,
        "rank_col":                  rank_col,
        "n_picks":                   len(symbols),
        "n_with_actual_data":        len(captures_1wk),

        # Persistence
        "n_top10_in_next_week":      n_in_top10_next,
        "n_top10_in_2_weeks":        n_in_top10_two,
        "n_top10_in_3_weeks":        n_in_top10_three,

        # Mean returns by hold length
        "mean_capture_1wk":          safe_mean(captures_1wk),
        "mean_capture_2wk":          safe_mean(captures_2wk),
        "mean_capture_3wk":          safe_mean(captures_3wk),
        "mean_capture_4wk":          safe_mean(captures_4wk),

        # Ceiling: peak over 3 weeks
        "mean_peak_capture_3wk":     safe_mean(peak_3wk),

        # Friction
        "n_rerank_next_week":        len(roundtrip_costs),
        "mean_roundtrip_cost":       safe_mean(roundtrip_costs),
    }


def run(rank_col: str = DEFAULT_RANK_COL):
    log.info("=" * 60)
    log.info("  CADENCE ANALYZER")
    log.info("=" * 60)
    log.info(f"Rank column: {rank_col}")
    log_event("cadence_analyzer", LogStatus.INFO,
              f"Starting cadence analysis (rank_col={rank_col})")

    try:
        perf = load_perf_log()
    except FileNotFoundError as e:
        log.warning(str(e))
        log_event("cadence_analyzer", LogStatus.WARNING,
                  "Skipped: perf log missing", errors=[str(e)])
        return

    if rank_col not in perf.columns:
        log.error(f"Rank column '{rank_col}' not in perf log -- available: "
                  f"{[c for c in perf.columns if 'rank' in c.lower()]}")
        log_event("cadence_analyzer", LogStatus.ERROR,
                  f"Rank column missing: {rank_col}")
        return

    n_ranked_rows = perf[rank_col].notna().sum()
    if n_ranked_rows == 0:
        log.error(f"Rank column '{rank_col}' has zero non-null values")
        log_event("cadence_analyzer", LogStatus.ERROR,
                  f"Rank column empty: {rank_col}")
        return

    ordered_weeks = sorted(perf["week_of"].unique().tolist())
    log.info(f"Perf log has {len(ordered_weeks)} weeks: "
             f"{ordered_weeks[0]} -> {ordered_weeks[-1]}")
    log.info(f"Rows with {rank_col} populated: {n_ranked_rows:,} / {len(perf):,}")

    # We can only analyze weeks where at least MIN_FORWARD_WEEKS of subsequent
    # data exist. So we cap the analyzable range.
    if len(ordered_weeks) <= MIN_FORWARD_WEEKS:
        log.warning(f"Not enough history yet: need at least {MIN_FORWARD_WEEKS + 1} "
                    f"weeks, have {len(ordered_weeks)}")
        log_event("cadence_analyzer", LogStatus.INFO,
                  "Skipped: insufficient history",
                  metrics={"weeks_available": len(ordered_weeks),
                           "weeks_required": MIN_FORWARD_WEEKS + 1})
        return

    analyzable = ordered_weeks[:-MIN_FORWARD_WEEKS]
    log.info(f"Analyzing {len(analyzable)} weeks "
             f"({analyzable[0]} -> {analyzable[-1]})")

    rows = []
    for entry_week in analyzable:
        result = analyze_week(perf, entry_week, ordered_weeks, rank_col)
        if result is not None:
            rows.append(result)

    if not rows:
        log.warning(f"No analyzable weeks produced metrics with rank_col={rank_col}. "
                    f"If using composite_rank, history may not extend back far enough -- "
                    f"try --rank-col alpha_rank for deeper history.")
        log_event("cadence_analyzer", LogStatus.WARNING,
                  "No metrics produced",
                  metrics={"rank_col": rank_col,
                           "ranked_rows": int(n_ranked_rows)})
        return

    df = pd.DataFrame(rows)

    # Append (don't overwrite) so we accumulate rows from both rank_col runs.
    # Keep last per (week_of, rank_col) so reruns of the same combination
    # update rather than duplicate.
    if OUTPUT_CSV.exists():
        prior = pd.read_csv(OUTPUT_CSV)
        # Filter prior rows that this run is recomputing for the same rank_col
        prior = prior[~((prior["week_of"].isin(df["week_of"])) &
                        (prior["rank_col"] == rank_col))]
        df = pd.concat([prior, df], ignore_index=True).sort_values(["rank_col", "week_of"])

    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Wrote {len(df)} total rows to {OUTPUT_CSV}")

    # Aggregate summary -- show only this run's rank_col
    this_run = df[df["rank_col"] == rank_col]
    recent = this_run.tail(12) if len(this_run) >= 12 else this_run
    log.info(f"\nAggregate over {len(recent)} most recent weeks "
             f"(rank_col={rank_col}):")
    log.info(f"  Mean 1-week capture:   {recent['mean_capture_1wk'].mean()*100:+.2f}%")
    log.info(f"  Mean 2-week capture:   {recent['mean_capture_2wk'].mean()*100:+.2f}%")
    log.info(f"  Mean 3-week capture:   {recent['mean_capture_3wk'].mean()*100:+.2f}%")
    log.info(f"  Mean 4-week capture:   {recent['mean_capture_4wk'].mean()*100:+.2f}%")
    log.info(f"  Mean peak over 3 wks:  {recent['mean_peak_capture_3wk'].mean()*100:+.2f}%")
    log.info(f"")
    log.info(f"  Top-10 persistence:")
    log.info(f"    -> next week:        "
             f"{recent['n_top10_in_next_week'].mean():.1f}/10 picks")
    log.info(f"    -> 2 weeks out:      "
             f"{recent['n_top10_in_2_weeks'].mean():.1f}/10 picks")
    log.info(f"    -> 3 weeks out:      "
             f"{recent['n_top10_in_3_weeks'].mean():.1f}/10 picks")
    log.info(f"")
    rt_mean = recent['mean_roundtrip_cost'].mean()
    if pd.notna(rt_mean):
        n_rerank = recent['n_rerank_next_week'].sum()
        log.info(f"  Mean roundtrip cost:   "
                 f"{rt_mean*100:+.3f}% per re-ranked pick "
                 f"(n={int(n_rerank)} re-ranks observed)")
    else:
        log.info(f"  Mean roundtrip cost:   N/A (no re-rankings observed in window)")

    log_event("cadence_analyzer", LogStatus.SUCCESS,
              f"Analyzed {len(this_run)} weeks (rank_col={rank_col})",
              metrics={
                  "rank_col":                rank_col,
                  "weeks_analyzed":          len(this_run),
                  "mean_capture_1wk_recent": round(float(recent['mean_capture_1wk'].mean() or 0), 5),
                  "mean_capture_2wk_recent": round(float(recent['mean_capture_2wk'].mean() or 0), 5),
                  "mean_capture_3wk_recent": round(float(recent['mean_capture_3wk'].mean() or 0), 5),
              })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rank-col",
        dest="rank_col",
        default=DEFAULT_RANK_COL,
        choices=["composite_rank", "alpha_rank"],
        help=f"Which rank column to use for top-N selection. Default: {DEFAULT_RANK_COL}. "
             f"Use alpha_rank for deeper history (composite_rank only available "
             f"from 2026-04-17 onward).",
    )
    args = parser.parse_args()
    run(rank_col=args.rank_col)
