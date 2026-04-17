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

Normal ranges for reference:
- Pipeline IC score: 0.40 to 0.75 (below 0.40 is concerning)
- Universe size: 4800 to 5500 tickers (significant change warrants review)
- Alpaca fills: expect 8-10 filled, 0-2 failed per week
- Weekend refresh: should run Sunday evening and Monday morning
- Premarket monitor: should run 5 checks on Monday morning

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

    # Read last 7 days of logs
    entries = read_logs(days=7)
    log.info(f"Log entries reviewed: {len(entries)}")

    if not entries:
        log.info("No log entries found -- system may be newly deployed")
        log_event("health_check", LogStatus.WARNING,
                  "No log entries found for past 7 days")
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
            model      = "claude-sonnet-4-20250514",
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
    if result["status"] == "clear":
        log.info("System healthy -- no action required")
    elif result["status"] == "issues":
        log.warning("Issues detected -- notification sent")
    else:
        log.error(f"Health check failed: {result.get('message')}")


if __name__ == "__main__":
    run()
