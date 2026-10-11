# automation/health_check.py
# Weekly system health check -- runs Fridays at 4:30 PM CT.
# Collects the past 7 days of system logs, sends to Claude for analysis,
# and fires a push notification only if something actionable is found.
#
# Philosophy: if everything is fine, do nothing and stay silent.
# Only alert when human attention is genuinely warranted.

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automation.system_logger import read_logs, format_logs_for_review, log_event, LogStatus
from automation.notifier import notify, NotifyPriority

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR      = Path("data")
HEALTH_LOG    = DATA_DIR / "health_check_log.jsonl"

SYSTEM_PROMPT = """You are the health monitoring system for Momentum Alpha, an autonomous
weekly stock ranking and trading system. You will receive structured logs from the past 7 days
covering all workflows: weekly pipeline, learning loop, Alpaca entry/exit, premarket monitor,
weekend catalyst refresh, and execution tracking.

Your job is to identify anything that warrants human attention. Be specific and concise.

Flag these categories if present:
- CRITICAL: failures that likely caused incorrect trades, missing entries/exits, corrupted data
- WARNING: degraded performance, unusual patterns, metrics outside normal range
- INFO: notable observations that don't require immediate action but are worth knowing

How the system runs now (updated Oct 2026) -- do NOT flag these as problems:
- Times are America/Chicago. A Netlify dispatcher triggers trading jobs on time:
  premarket 06:00 Mon/Tue; position monitor 08:30, 11:15, 14:00 Mon-Fri;
  entry 14:45 Mon/Tue; stops 15:10 Mon/Tue; exit 14:45 Thu/Fri.
  GitHub cron backups fire the same jobs again (two UTC times per job for DST),
  so repeat runs are expected. "Skipped: ...", "not exit day", "already ran",
  "already in exec log" and "ghost run prevention" lines are guards working
  correctly -- normal no-ops, never issues.
- The scoring weights and the ML model are FROZEN on purpose (report-only).
  "Proposal logged, not applied" and "Candidate trained (not deployed)" are expected.
  The retrain "in-sample IC" belongs to an undeployed candidate; normal is 0.15-0.30.
- Negative cadence capture (top-ranked names under-performing) is a known,
  documented weakness under research review. Mention it at most as one INFO line.
- Three shadow portfolios (no orders) update Monday/Tuesday evenings.

Normal ranges:
- Universe size: 1,700-2,300 tickers. Flag below 1,500, or a >15% change vs last week.
- Alpaca entry: 8-10 filled, 0-2 failed. Exit closes every open position on Thu/Fri.
- Stops: every entry week should log a successful place_stops run.

Resolution rule: if a failure was followed later in the window by a successful run of
the same workflow (and mode), report it as a single INFO line marked "resolved", not as
CRITICAL/WARNING. Only flag what is still broken or what a human must still do.

If everything looks normal across all workflows, respond with exactly:
CLEAR: System operating normally. No issues detected.

If issues exist, respond with:
ISSUES FOUND:
[bullet list of specific issues with workflow name, timestamp if relevant, and recommended action]

Be direct. Do not pad with reassurances. If it is CLEAR, say CLEAR. If there are issues, say exactly what they are."""


def run_health_check() -> dict:
    """
    Read weekly logs, send to Claude, return analysis result.
    """
    log.info("=" * 60)
    log.info("WEEKLY HEALTH CHECK")
    log.info("=" * 60)

    # One check per week: the cron fires twice (DST pair), so skip a repeat run
    # within 18 hours of the last completed check -- avoids duplicate alerts.
    if HEALTH_LOG.exists():
        try:
            last = json.loads(HEALTH_LOG.read_text().strip().splitlines()[-1])
            age_h = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(last["timestamp"])).total_seconds() / 3600
            if age_h < 18:
                log.info(f"Skipped: health check already ran {age_h:.1f}h ago")
                log_event("health_check", LogStatus.INFO,
                          f"Skipped: already ran {age_h:.1f}h ago")
                return {"status": "skipped", "message": "already ran"}
        except Exception as e:
            log.warning(f"Could not read last health check time: {e}")

    # Read last 7 days of logs
    entries = read_logs(days=7)
    log.info(f"Log entries reviewed: {len(entries)}")

    if not entries:
        log.info("No log entries found -- system may be newly deployed")
        # H1 fix: this was the most common failure mode -- for weeks the system_log
        # was empty (no module wrote to it) and the health check quietly logged a
        # warning to itself and returned. The user never saw alerts. Now this path
        # escalates via push notification so the "nothing is being observed"
        # condition can't hide.
        log_event("health_check", LogStatus.WARNING,
                  "No log entries found for past 7 days")
        notify(
            title    = "Momentum Alpha — OBSERVABILITY GAP",
            message  = ("Health check found zero log entries in the last 7 days. "
                        "Either the system hasn't run, or no module is writing to "
                        "system_log.jsonl. Investigate before the next pipeline run."),
            priority = NotifyPriority.HIGH,
            tags     = ["warning", "eyes"],
        )
        return {"status": "warning", "message": "No log entries found"}

    # Format for Claude
    log_text = format_logs_for_review(entries)

    # Count by status for summary
    status_counts = {}
    workflow_counts = {}
    for e in entries:
        st = e.get("status", "info")
        wf = e.get("workflow", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1
        workflow_counts[wf] = workflow_counts.get(wf, 0) + 1

    log.info(f"Status breakdown: {status_counts}")
    log.info(f"Workflows covered: {list(workflow_counts.keys())}")

    # Send to Claude
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set -- skipping Claude analysis")
        log_event("health_check", LogStatus.WARNING,
                  "ANTHROPIC_API_KEY not set -- health check skipped")
        return {"status": "warning", "message": "API key not set"}

    log.info("Sending logs to Claude for analysis...")
    try:
        from anthropic import Anthropic
        client   = Anthropic()
        response = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 1000,
            system     = SYSTEM_PROMPT,
            messages   = [{
                "role": "user",
                "content": f"Weekly system logs ({len(entries)} entries, "
                           f"{len(entries)} events across {len(workflow_counts)} workflows):\n\n"
                           f"{log_text}"
            }],
        )
        analysis = response.content[0].text.strip()
    except Exception as e:
        log.error(f"Claude API call failed: {e}")
        log_event("health_check", LogStatus.ERROR,
                  "Claude API call failed", errors=[str(e)])
        notify(
            "Momentum Alpha — Health Check Failed",
            f"Could not complete weekly health check: {e}",
            priority=NotifyPriority.HIGH,
        )
        return {"status": "error", "message": str(e)}

    log.info(f"Claude analysis:\n{analysis}")

    is_clear = analysis.upper().startswith("CLEAR")

    # Log the health check result
    log_event(
        "health_check",
        LogStatus.SUCCESS if is_clear else LogStatus.WARNING,
        "Weekly health check complete",
        metrics={
            "entries_reviewed": len(entries),
            "workflows_covered": len(workflow_counts),
            "errors_in_period": status_counts.get("error", 0),
            "warnings_in_period": status_counts.get("warning", 0),
            "result": "clear" if is_clear else "issues_found",
        }
    )

    # Save health check to its own log
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    health_entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "is_clear":   is_clear,
        "analysis":   analysis,
        "entries_reviewed": len(entries),
        "status_counts": status_counts,
    }
    with open(HEALTH_LOG, "a") as f:
        f.write(json.dumps(health_entry) + "\n")

    # Notify only if issues found
    if is_clear:
        log.info("Health check CLEAR -- no notification sent")
    else:
        log.warning("Health check found issues -- sending notification")
        # Extract first 500 chars for notification
        summary = analysis[:500] + ("..." if len(analysis) > 500 else "")
        notify(
            title    = "Momentum Alpha — Weekly Health Check",
            message  = summary,
            priority = NotifyPriority.HIGH,
            tags     = ["warning"],
        )

    return {
        "status":   "clear" if is_clear else "issues",
        "analysis": analysis,
        "entries":  len(entries),
    }


def run():
    result = run_health_check()
    if result["status"] in ("clear", "skipped"):
        log.info("System healthy -- no action required")
    elif result["status"] == "issues":
        log.warning("Issues detected -- notification sent")
    else:
        log.error(f"Health check failed: {result.get('message')}")


if __name__ == "__main__":
    run()
