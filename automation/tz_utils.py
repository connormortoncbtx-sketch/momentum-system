"""
automation/tz_utils.py
=======================
Shared timezone utilities for all automation scripts.
Handles US Central Time (CT) correctly across DST transitions.

US Central Time:
    CDT (Daylight): UTC-5  (2nd Sunday March → 1st Sunday November)
    CST (Standard): UTC-6  (1st Sunday November → 2nd Sunday March)

Usage:
    from automation.tz_utils import now_ct, ct_hour, is_market_open, wait_until_ct_hour
"""

from datetime import datetime, timezone, timedelta
import time
import logging

log = logging.getLogger(__name__)


def ct_offset() -> int:
    """
    Returns current CT UTC offset in hours (-5 for CDT, -6 for CST).
    Computes DST status based on US rules without external libraries.
    """
    now_utc = datetime.now(timezone.utc)
    year    = now_utc.year

    # DST starts: 2nd Sunday in March at 2:00 AM local
    # DST ends:   1st Sunday in November at 2:00 AM local
    def nth_sunday(year, month, n):
        """Return date of nth Sunday in given month/year."""
        d = datetime(year, month, 1)
        days_until_sunday = (6 - d.weekday()) % 7
        first_sunday = d + timedelta(days=days_until_sunday)
        return first_sunday + timedelta(weeks=n-1)

    dst_start = nth_sunday(year, 3, 2).replace(hour=8, tzinfo=timezone.utc)   # 2am CST = 8am UTC
    dst_end   = nth_sunday(year, 11, 1).replace(hour=7, tzinfo=timezone.utc)  # 2am CDT = 7am UTC

    if dst_start <= now_utc < dst_end:
        return -5   # CDT
    else:
        return -6   # CST


def now_ct() -> datetime:
    """Return current datetime in US Central Time."""
    offset = ct_offset()
    return datetime.now(timezone.utc) + timedelta(hours=offset)


def ct_hour() -> float:
    """Return current hour in CT as float (e.g. 6.5 = 6:30 AM)."""
    ct = now_ct()
    return ct.hour + ct.minute / 60


def is_dst() -> bool:
    """Return True if currently observing CDT (Daylight Saving Time)."""
    return ct_offset() == -5


def format_ct(dt: datetime = None) -> str:
    """Format a datetime as CT string with DST label."""
    if dt is None:
        dt = now_ct()
    label = "CDT" if is_dst() else "CST"
    return dt.strftime(f"%Y-%m-%d %H:%M {label}")


def wait_until_ct_hour(target_hour: float, label: str = ""):
    """
    Sleep until target_hour in CT. If already past, return immediately.

    Args:
        target_hour: Target hour as float (e.g. 6.0 = 6:00 AM, 6.5 = 6:30 AM)
        label: Optional label for log message
    """
    current = ct_hour()
    if current >= target_hour:
        return

    wait_minutes = (target_hour - current) * 60
    target_str = f"{int(target_hour)}:{int((target_hour % 1) * 60):02d} CT"
    label_str  = f" ({label})" if label else ""

    log.info(f"Waiting {wait_minutes:.0f} min until {target_str}{label_str} "
             f"[currently {now_ct().strftime('%H:%M')} {'CDT' if is_dst() else 'CST'}]")
    time.sleep(wait_minutes * 60)


def market_opens_ct() -> float:
    """Return market open time in CT hours (8.5 = 8:30 AM)."""
    return 8.5   # 9:30 AM ET = 8:30 AM CT always


def premarket_opens_ct() -> float:
    """Return pre-market open time in CT hours (3.0 = 3:00 AM)."""
    return 3.0   # 4:00 AM ET = 3:00 AM CT


if __name__ == "__main__":
    print(f"Current CT:  {format_ct()}")
    print(f"CT offset:   UTC{ct_offset()}")
    print(f"DST active:  {is_dst()}")
    print(f"CT hour:     {ct_hour():.2f}")
    print(f"Market open: {market_opens_ct()} CT")
