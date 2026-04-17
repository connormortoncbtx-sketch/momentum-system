# automation/system_logger.py
# Centralized structured logging for all Momentum Alpha workflows.
# Every workflow calls log_event() to append a JSON line to data/system_log.jsonl
#
# Format: one JSON object per line, each with:
#   timestamp, workflow, status, message, metrics (dict), errors (list)
#
# Usage:
#   from automation.system_logger import log_event, LogStatus
#   log_event("weekly_pipeline", LogStatus.SUCCESS, "Pipeline complete",
#             metrics={"regime": "risk_on", "universe": 5102, "ic": 0.6158})
#
# To retrieve logs for a time window:
#   entries = read_logs(days=7)

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from enum import Enum

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
LOG_FILE = DATA_DIR / "system_log.jsonl"
MAX_LOG_LINES = 10000  # rotate after this many lines


class LogStatus(str, Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR   = "error"
    INFO    = "info"


def log_event(
    workflow:  str,
    status:    LogStatus,
    message:   str,
    metrics:   dict = None,
    errors:    list = None,
):
    """
    Append a structured log entry to data/system_log.jsonl.
    Safe to call from any workflow -- creates the file if it doesn't exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workflow":  workflow,
        # Use .value for proper enum-string serialization. Previously str(status) returned
        # "LogStatus.SUCCESS" not "success" due to Python 3.11+ Enum str() behavior change,
        # which made all downstream consumers (read_logs, format_logs_for_review, health_check)
        # see malformed status values.
        "status":    status.value if isinstance(status, LogStatus) else str(status),
        "message":   message,
        "metrics":   metrics or {},
        "errors":    errors or [],
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning(f"system_logger: failed to write log entry: {e}")

    # Rotate if too large
    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_LOG_LINES:
            # Keep the most recent 8000 lines
            LOG_FILE.write_text(
                "\n".join(lines[-8000:]) + "\n",
                encoding="utf-8"
            )
    except Exception:
        pass


def read_logs(days: int = 7) -> list[dict]:
    """
    Read log entries from the past N days.
    Returns list of dicts sorted oldest to newest.
    """
    if not LOG_FILE.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []

    try:
        for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts >= cutoff:
                    entries.append(entry)
            except Exception:
                continue
    except Exception as e:
        log.warning(f"system_logger: failed to read logs: {e}")

    return sorted(entries, key=lambda e: e["timestamp"])


def format_logs_for_review(entries: list[dict]) -> str:
    """
    Format log entries into a readable text block for Claude review.
    """
    if not entries:
        return "No log entries found."

    lines = []
    for e in entries:
        ts   = e.get("timestamp", "")[:19].replace("T", " ")
        wf   = e.get("workflow", "unknown")
        st   = e.get("status", "info").upper()
        msg  = e.get("message", "")
        metrics = e.get("metrics", {})
        errors  = e.get("errors", [])

        line = f"[{ts}] [{st}] {wf}: {msg}"
        if metrics:
            line += "  |  " + "  ".join(f"{k}={v}" for k, v in metrics.items())
        if errors:
            line += "  |  ERRORS: " + "; ".join(str(e) for e in errors)
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test
    log_event("test", LogStatus.SUCCESS, "Logger working",
              metrics={"test": True})
    entries = read_logs(days=1)
    print(f"Log entries in last 24h: {len(entries)}")
    print(format_logs_for_review(entries[-5:]))
