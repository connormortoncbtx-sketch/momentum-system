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

from datetime import datetime, timezone, timedelta, date
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

    def nth_sunday(year, month, n):
        d = datetime(year, month, 1)
        days_until_sunday = (6 - d.weekday()) % 7
        first_sunday = d + timedelta(days=days_until_sunday)
        return first_sunday + timedelta(weeks=n-1)

    dst_start = nth_sunday(year, 3, 2).replace(hour=8, tzinfo=timezone.utc)
    dst_end   = nth_sunday(year, 11, 1).replace(hour=7, tzinfo=timezone.utc)

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
    """Sleep until target_hour in CT. If already past, return immediately."""
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
    return 8.5


def premarket_opens_ct() -> float:
    """Return pre-market open time in CT hours (3.0 = 3:00 AM)."""
    return 3.0


# ── HOLIDAY CALENDAR ──────────────────────────────────────────────────────────

def _nyse_holidays(year: int) -> set[date]:
    """
    Return set of NYSE full-closure dates for the given year.
    Based on NYSE holiday schedule rules — no external dependencies.
    """
    holidays = set()

    def nth_weekday(year, month, n, weekday):
        """nth occurrence of weekday (0=Mon) in month. n=1 is first."""
        d = date(year, month, 1)
        delta = (weekday - d.weekday()) % 7
        first = d + timedelta(days=delta)
        return first + timedelta(weeks=n-1)

    def nearest_weekday(d: date) -> date:
        """If holiday falls on weekend, NYSE observes nearest weekday."""
        if d.weekday() == 5:   # Saturday → Friday
            return d - timedelta(days=1)
        if d.weekday() == 6:   # Sunday → Monday
            return d + timedelta(days=1)
        return d

    # New Year's Day
    holidays.add(nearest_weekday(date(year, 1, 1)))
    # MLK Day — 3rd Monday in January
    holidays.add(nth_weekday(year, 1, 3, 0))
    # Presidents Day — 3rd Monday in February
    holidays.add(nth_weekday(year, 2, 3, 0))
    # Good Friday — 2 days before Easter
    easter = _easter(year)
    holidays.add(easter - timedelta(days=2))
    # Memorial Day — last Monday in May
    holidays.add(nth_weekday(year, 5, 5, 0) if nth_weekday(year, 5, 5, 0).month == 5
                 else nth_weekday(year, 5, 4, 0))
    # Juneteenth — June 19
    holidays.add(nearest_weekday(date(year, 6, 19)))
    # Independence Day — July 4
    holidays.add(nearest_weekday(date(year, 7, 4)))
    # Labor Day — 1st Monday in September
    holidays.add(nth_weekday(year, 9, 1, 0))
    # Thanksgiving — 4th Thursday in November
    holidays.add(nth_weekday(year, 11, 4, 3))
    # Christmas — December 25
    holidays.add(nearest_weekday(date(year, 12, 25)))

    return holidays


def _easter(year: int) -> date:
    """Compute Easter date using Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def is_trading_day(d: date = None) -> bool:
    """Return True if d is a full NYSE trading day (not holiday, not weekend)."""
    if d is None:
        d = now_ct().date()
    if d.weekday() >= 5:   # Saturday or Sunday
        return False
    holidays = _nyse_holidays(d.year)
    return d not in holidays


def is_normal_trading_week(ref_date: date = None) -> bool:
    """
    Return True if the current week (Mon-Fri) has all 5 full trading days.
    A week with any holiday is considered disrupted and should be skipped.

    This enforces the system's core principle: consistent repeatable cadence.
    Holiday weeks produce non-comparable data that degrades model quality.
    """
    if ref_date is None:
        ref_date = now_ct().date()

    # Find Monday of this week
    monday = ref_date - timedelta(days=ref_date.weekday())
    week_days = [monday + timedelta(days=i) for i in range(5)]  # Mon-Fri

    holidays = _nyse_holidays(ref_date.year)
    for d in week_days:
        if d in holidays:
            log.info(f"Holiday week detected: {d.strftime('%A %Y-%m-%d')} is a market holiday")
            return False
    return True


def assert_normal_week(script_name: str = "") -> bool:
    """
    Call at the top of any script that should skip on holiday weeks.
    Logs a clear message and returns False if week is disrupted.

    Usage:
        from automation.tz_utils import assert_normal_week
        if not assert_normal_week("weekend_refresh"):
            sys.exit(0)
    """
    if not is_normal_trading_week():
        label = f" [{script_name}]" if script_name else ""
        log.info(f"Holiday week{label} — skipping. System resumes next full trading week.")
        return False
    return True


if __name__ == "__main__":
    print(f"Current CT:       {format_ct()}")
    print(f"CT offset:        UTC{ct_offset()}")
    print(f"DST active:       {is_dst()}")
    print(f"CT hour:          {ct_hour():.2f}")
    print(f"Trading day:      {is_trading_day()}")
    print(f"Normal week:      {is_normal_trading_week()}")
    print(f"\nNYSE holidays {now_ct().year}:")
    for h in sorted(_nyse_holidays(now_ct().year)):
        print(f"  {h.strftime('%A %B %d')}")
