# automation/notifier.py
# Push notification system via Ntfy.sh -- free, no account required.
#
# Install the Ntfy app on your phone and subscribe to your channel.
# Channel name is stored as GitHub secret NTFY_CHANNEL.
#
# Usage:
#   from automation.notifier import notify, NotifyPriority
#   notify("Pipeline failed", "Stage 3 signals error on ORCL", priority=NotifyPriority.HIGH)

import logging
import os
import sys
import urllib.request
import urllib.error
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

NTFY_BASE    = "https://ntfy.sh"
CHANNEL_KEY  = "NTFY_CHANNEL"


class NotifyPriority(str, Enum):
    LOW      = "low"
    DEFAULT  = "default"
    HIGH     = "high"
    URGENT   = "urgent"


def notify(
    title:    str,
    message:  str,
    priority: NotifyPriority = NotifyPriority.DEFAULT,
    tags:     list[str] = None,
) -> bool:
    """
    Send a push notification via Ntfy.sh.
    Returns True if sent successfully, False otherwise.
    Fails silently -- never raises exceptions that would break a workflow.
    """
    channel = os.environ.get(CHANNEL_KEY, "").strip()
    if not channel:
        log.debug("NTFY_CHANNEL not set -- skipping notification")
        return False

    url = f"{NTFY_BASE}/{channel}"

    headers = {
        "Title":    title.encode("utf-8"),
        "Priority": priority.value.encode("utf-8"),
        "Content-Type": b"text/plain",
    }
    if tags:
        headers["Tags"] = ",".join(tags).encode("utf-8")

    try:
        req  = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                log.info(f"Notification sent: {title}")
                return True
            else:
                log.warning(f"Ntfy returned status {resp.status}")
                return False
    except urllib.error.URLError as e:
        log.warning(f"Notification failed (network): {e}")
        return False
    except Exception as e:
        log.warning(f"Notification failed: {e}")
        return False


def notify_error(workflow: str, error: str):
    """Convenience wrapper for error notifications."""
    notify(
        title    = f"Momentum Alpha — {workflow} ERROR",
        message  = error,
        priority = NotifyPriority.HIGH,
        tags     = ["warning", "chart_with_downwards_trend"],
    )


def notify_alert(workflow: str, message: str):
    """Convenience wrapper for warning notifications."""
    notify(
        title    = f"Momentum Alpha — {workflow} Alert",
        message  = message,
        priority = NotifyPriority.DEFAULT,
        tags     = ["bell"],
    )


def notify_success(workflow: str, message: str):
    """Convenience wrapper for success notifications."""
    notify(
        title    = f"Momentum Alpha — {workflow}",
        message  = message,
        priority = NotifyPriority.LOW,
        tags     = ["white_check_mark"],
    )


if __name__ == "__main__":
    # Test -- requires NTFY_CHANNEL env var
    result = notify(
        "Momentum Alpha Test",
        "Notification system working correctly.",
        priority=NotifyPriority.LOW
    )
    print(f"Notification sent: {result}")
