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
from pathlib import Path

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


def run_stage(module_path: str, label: str) -> bool:
    import importlib
    log.info(f"▶  {label}")
    t0 = time.time()
    try:
        mod = importlib.import_module(module_path)
        mod.run()
        log.info(f"✓  {label}  ({time.time()-t0:.1f}s)\n")
        return True
    except Exception as e:
        log.error(f"✗  {label} FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",  dest="from_stage", default="01")
    parser.add_argument("--only",  dest="only_stage", default=None)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  MOMENTUM SYSTEM — Weekly Pipeline")
    log.info("=" * 60)

    overall_start = time.time()

    for num, label, module in STAGES:
        if args.only_stage and num != args.only_stage:
            continue
        if not args.only_stage and num < args.from_stage:
            log.info(f"   skipping {label}")
            continue

        ok = run_stage(module, f"Stage {num}: {label}")
        if not ok:
            log.error(f"Pipeline halted at stage {num}")
            sys.exit(1)

    elapsed = time.time() - overall_start
    log.info("=" * 60)
    log.info(f"  Pipeline complete  ({elapsed/60:.1f} min)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
