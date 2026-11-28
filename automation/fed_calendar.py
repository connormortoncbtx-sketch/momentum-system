"""
automation/fed_calendar.py -- scheduled 2pm-ET Fed releases (the live condor's entry time).
Source: federalreserve.gov FOMC calendar, revised Nov 12, 2026 (Oct 26-27, 2027 is a strategy meeting with NO policy
decision; the December 2027 meeting moved to Nov 16-17). Decisions are announced at 2:00 pm ET on the meeting's second
day. Minutes are released at 2:00 pm ET three weeks after each decision (computed from that rule; the Fed occasionally
shifts a release around holidays -- check the calendar page when extending). EXTEND BEFORE NOV 2027.
"""
import datetime as dt

DECISION_DAYS = {"2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15",
                 "2027-11-17"}
MINUTES_DAYS = {(dt.date.fromisoformat(d) + dt.timedelta(days=21)).isoformat() for d in DECISION_DAYS}
LAST_KNOWN = max(DECISION_DAYS | MINUTES_DAYS)


def fed_event(day):
    """'Fed decision' / 'Fed minutes' / None for a date or ISO string."""
    d = day if isinstance(day, str) else day.isoformat()
    return "Fed decision" if d in DECISION_DAYS else "Fed minutes" if d in MINUTES_DAYS else None
