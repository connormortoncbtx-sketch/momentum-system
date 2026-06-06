"""
automation/verify_collect_returns.py
=====================================
Post-step verifier called by the friday_learning workflow after the
collect_returns step. Replaces the bash mtime-only check.

The old guard compared performance_log.csv's mtime to the run-start time
and failed the workflow if it hadn't advanced. That caught silent
crashes correctly but ALSO caught legitimate skips:
  - holiday weeks (script returns early)
  - week already in perf_log (idempotency skip)
  - not enough trading days elapsed (timing guard)
  - scores_final.csv missing
  - weekend_refresh / pipeline failed earlier so no scores to consume

This script reads system_log.jsonl, finds the most recent
collect_returns event from this run window, and decides:
  - SUCCESS or legitimate skip (warning/info)  → exit 0
  - ERROR or no event found at all             → exit 1

If the system_log isn't available, falls back to the mtime check (so we
never lose the original safety net entirely).

Usage:
    python automation/verify_collect_returns.py <run_start_epoch>
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

SYSTEM_LOG = Path("data/system_log.jsonl")
PERF_LOG   = Path("data/performance_log.csv")


def parse_iso(ts: str) -> datetime:
    """Parse the ISO8601 timestamps used by system_logger.log_event."""
    # log_event writes UTC ISO with explicit +00:00 offset
    return datetime.fromisoformat(ts)


def find_recent_collect_event(run_start_epoch: int):
    """
    Scan system_log.jsonl from the end backwards looking for the most
    recent collect_returns event whose timestamp is >= run_start_epoch.
    Returns the event dict, or None if no matching event is found.
    """
    if not SYSTEM_LOG.exists():
        return None

    run_start_dt = datetime.fromtimestamp(run_start_epoch, tz=timezone.utc)

    # Read all lines, reverse for newest-first scan. The log is small
    # enough (a few thousand lines after months of operation) that this
    # is fine; if it ever grows huge, switch to a tail-and-iterate.
    try:
        with open(SYSTEM_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"::warning::Could not read {SYSTEM_LOG}: {e}", file=sys.stderr)
        return None

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("workflow") != "collect_returns":
            continue
        # Only consider events from this run window. Skip earlier
        # historical events (e.g. the prior week's success).
        try:
            evt_dt = parse_iso(evt["timestamp"])
        except (KeyError, ValueError):
            continue
        if evt_dt < run_start_dt:
            # Reached older events; nothing newer matched.
            return None
        # Skip the bare "Starting collect_returns" info event -- we want
        # the terminal event (success/warning/error/skip).
        msg = evt.get("message", "")
        if msg.startswith("Starting collect_returns"):
            continue
        return evt

    return None


def mtime_fallback(run_start_epoch: int) -> int:
    """
    Original mtime-based check, kept as a fallback when system_log is
    unavailable. Returns 0 (pass) or 1 (fail).
    """
    if not PERF_LOG.exists():
        print(f"::error::{PERF_LOG} missing -- collect_returns.py never wrote output",
              file=sys.stderr)
        return 1
    file_mtime = int(PERF_LOG.stat().st_mtime)
    if file_mtime < run_start_epoch:
        ms = datetime.fromtimestamp(file_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        rs = datetime.fromtimestamp(run_start_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"::error::{PERF_LOG} was not updated this run.", file=sys.stderr)
        print(f"::error::collect_returns.py likely failed silently "
              f"(continue-on-error masking an exception).", file=sys.stderr)
        print(f"File mtime: {ms}", file=sys.stderr)
        print(f"Run start:  {rs}", file=sys.stderr)
        return 1
    print(f"{PERF_LOG} updated this run -- collect_returns.py succeeded "
          f"(mtime fallback)")
    return 0


def main():
    if len(sys.argv) < 2:
        print("::error::Run start epoch missing -- earlier step likely failed",
              file=sys.stderr)
        return 1
    try:
        run_start_epoch = int(sys.argv[1])
    except ValueError:
        print(f"::error::Invalid run_start_epoch: {sys.argv[1]!r}", file=sys.stderr)
        return 1

    evt = find_recent_collect_event(run_start_epoch)

    if evt is None:
        # No collect_returns event from this run found. That's suspicious --
        # at minimum a "Starting" event should have been logged. Fall back
        # to the mtime check to preserve the original safety net.
        print("No collect_returns terminal event found in system_log; "
              "falling back to mtime check")
        return mtime_fallback(run_start_epoch)

    status  = evt.get("status", "").lower()
    message = evt.get("message", "")

    if status == "success":
        print(f"collect_returns SUCCESS: {message}")
        return 0

    if status in ("info", "warning") and "Skipped" in message:
        # Legitimate skip path: holiday week, already-ran lock, already-in-
        # perf-log, not enough days elapsed, missing scores, etc.
        print(f"collect_returns legitimately skipped ({status}): {message}")
        return 0

    if status == "error":
        print(f"::error::collect_returns reported an error: {message}",
              file=sys.stderr)
        errors = evt.get("errors", [])
        for err in errors:
            print(f"::error::  {err}", file=sys.stderr)
        return 1

    if status == "warning":
        # Warning that isn't a skip -- e.g. "scores_final.csv missing".
        # This is a real condition the operator should see, but it's not
        # a silent failure; the script reported it explicitly. Pass with
        # a workflow-level warning annotation so it's visible.
        print(f"::warning::collect_returns reported warning: {message}")
        return 0

    # Unrecognized status -- be conservative and fail loud so we look at it.
    print(f"::error::collect_returns event had unexpected status {status!r}: {message}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
