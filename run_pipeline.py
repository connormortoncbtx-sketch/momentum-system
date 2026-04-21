"""
run_pipeline.py — Master orchestrator
======================================
Runs all pipeline stages in sequence.
Each stage is independently importable for testing.

Usage:
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --from 03   # resume from stage 3
    python run_pipeline.py --only 01   # run one stage only
"""

import sys
import time
import logging
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from automation.system_logger import log_event, LogStatus
from automation.notifier import notify_alert, notify_error, notify_success

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STAGES = [
    ("01", "Universe builder",    "pipeline.01_universe"),
    ("02", "Regime classifier",   "pipeline.02_regime"),
    ("03", "Signal engine",       "pipeline.03_signals"),
    ("04", "Scoring model",       "pipeline.04_model"),
    ("05", "LLM synthesis",       "pipeline.05_llm_synthesis"),
    ("06", "Report generator",    "pipeline.06_report"),
]


def run_stage(module_path: str, label: str, stage_num: str) -> bool:
    import importlib
    log.info(f"▶  {label}")
    log_event("weekly_pipeline", LogStatus.INFO, f"Starting {label}")
    t0 = time.time()
    try:
        mod = importlib.import_module(module_path)
        mod.run()
        elapsed = time.time() - t0
        log.info(f"✓  {label}  ({elapsed:.1f}s)\n")
        log_event("weekly_pipeline", LogStatus.SUCCESS,
                  f"{label} complete",
                  metrics={"stage": stage_num, "elapsed_sec": round(elapsed, 1)})
        return True
    except Exception as e:
        # Preserve the traceback so the GitHub Actions log shows what broke.
        tb = traceback.format_exc()
        log.error(f"✗  {label} FAILED: {e}")
        log.error(tb)
        log_event("weekly_pipeline", LogStatus.ERROR,
                  f"{label} FAILED: {e}",
                  errors=[str(e), tb[:2000]])
        # Do NOT notify per-stage -- main() sends one consolidated notification
        # to avoid spamming the phone on a cascading failure.
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",     dest="from_stage", default="01")
    parser.add_argument("--only",     dest="only_stage", default=None)
    parser.add_argument("--no-holiday-check", dest="skip_holiday", action="store_true",
                        help="Skip the holiday week check (for manual runs)")
    parser.add_argument("--force", dest="force", action="store_true",
                        help="Force run even if scores_final.csv was updated recently "
                             "(bypasses ghost-run prevention).")
    args = parser.parse_args()

    log_event("weekly_pipeline", LogStatus.INFO,
              f"Pipeline invoked (from={args.from_stage}, only={args.only_stage}, "
              f"force={args.force})")

    # ── GHOST RUN PREVENTION ──────────────────────────────────────────────
    # GitHub cron is "best effort" and can fire hours late during high load.
    # If a manual workflow_dispatch ran Friday night and the scheduled cron
    # then fires Saturday (or later), concurrency groups serialize them but
    # both still execute -- the second run overwrites scores_final.csv with
    # a fresh compute on top of the earlier one. This is how the 2026-04-18
    # "ghost pipeline" produced a second set of rankings that silently
    # replaced the manual run's output.
    #
    # Check: if scores_final.csv was updated within MIN_RERUN_HOURS, treat
    # this as a redundant invocation and exit cleanly. --force overrides.
    # --only and --from arguments also override (partial runs are legitimate
    # mid-week recovery operations, not full-pipeline re-executions).
    #
    # Why 18 hours: covers Friday-night → Saturday-morning window comfortably
    # while still allowing a legitimate re-run after a full trading day if
    # needed. A normal weekly cycle (~144 hours apart) is well above this.
    MIN_RERUN_HOURS = 18
    if not args.force and not args.only_stage and args.from_stage == "01":
        scores_path = Path("data/scores_final.csv")
        if scores_path.exists():
            age_hours = (time.time() - scores_path.stat().st_mtime) / 3600
            if age_hours < MIN_RERUN_HOURS:
                log.warning(
                    f"scores_final.csv was modified {age_hours:.1f}h ago "
                    f"(< {MIN_RERUN_HOURS}h threshold)"
                )
                log.warning(
                    "Treating this invocation as a ghost run "
                    "(GitHub cron firing late after a prior run). Exiting cleanly."
                )
                log.warning(
                    "To force re-run, dispatch with --force or wait "
                    f"{MIN_RERUN_HOURS - age_hours:.1f}h."
                )
                log_event("weekly_pipeline", LogStatus.INFO,
                          f"Skipped: ghost run prevention (last run {age_hours:.1f}h ago)",
                          metrics={"last_run_hours_ago": round(age_hours, 2),
                                   "threshold_hours":    MIN_RERUN_HOURS})
                sys.exit(0)

    # Holiday check — skip if not a full 5-day trading week
    # Bypass with --no-holiday-check for manual mid-week runs
    if not args.skip_holiday:
        from automation.tz_utils import assert_normal_week
        if not assert_normal_week("weekly_pipeline"):
            log_event("weekly_pipeline", LogStatus.INFO,
                      "Skipped: not a normal trading week")
            sys.exit(0)

    log.info("=" * 60)
    log.info("  MOMENTUM SYSTEM — Weekly Pipeline")
    log.info("=" * 60)

    overall_start = time.time()
    failed_stage = None

    for num, label, module in STAGES:
        if args.only_stage and num != args.only_stage:
            continue
        if not args.only_stage and num < args.from_stage:
            log.info(f"   skipping {label}")
            continue

        ok = run_stage(module, f"Stage {num}: {label}", num)
        if not ok:
            failed_stage = f"{num}: {label}"
            log.error(f"Pipeline halted at stage {num}")
            log_event("weekly_pipeline", LogStatus.ERROR,
                      f"Pipeline halted at stage {failed_stage}")
            # Single consolidated failure notification with the stage that broke.
            notify_error("weekly_pipeline",
                         f"Pipeline halted at stage {failed_stage}. "
                         f"Check workflow logs for traceback.")
            sys.exit(1)

    elapsed = time.time() - overall_start
    log.info("=" * 60)
    log.info(f"  Pipeline complete  ({elapsed/60:.1f} min)")
    log.info("=" * 60)

    log_event("weekly_pipeline", LogStatus.SUCCESS,
              f"Pipeline complete ({elapsed/60:.1f} min)",
              metrics={"elapsed_min": round(elapsed/60, 1),
                       "stages_run":  len(STAGES) if not args.only_stage else 1})
    notify_success("weekly_pipeline",
                   f"Weekly pipeline complete in {elapsed/60:.1f} min. "
                   f"Check docs/index.html for top picks.")


if __name__ == "__main__":
    main()
