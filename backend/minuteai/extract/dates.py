"""Deterministic resolution of relative due-date phrases ("by Friday", "end of next week")."""

from __future__ import annotations

import re
from datetime import date, timedelta

from dateutil import parser as dateparser

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "ten": 10, "fourteen": 14, "couple": 2, "few": 3}


def _next_weekday(ref: date, weekday: int) -> date:
    """The next occurrence of `weekday` strictly after `ref`."""
    days = (weekday - ref.weekday()) % 7 or 7
    return ref + timedelta(days=days)


def _weekday_of_next_week(ref: date, weekday: int) -> date:
    """"next Friday" = the Friday of the following calendar week (weeks start Monday)."""
    next_monday = ref + timedelta(days=7 - ref.weekday())
    return next_monday + timedelta(days=weekday)


def _end_of_month(ref: date) -> date:
    nxt = (ref.replace(day=28) + timedelta(days=4)).replace(day=1)
    return nxt - timedelta(days=1)


def resolve_relative_date(phrase: str | None, ref: date) -> date | None:
    """Resolve `phrase` against meeting date `ref`. Returns None if unresolvable."""
    if not phrase:
        return None
    p = phrase.lower().strip()
    p = re.sub(r"^(by|before|on|until|till|no later than|due)\s+", "", p)
    p = re.sub(r"\s+", " ", p)

    if p in ("today", "tonight", "end of day", "eod", "end of today"):
        return ref
    if p in ("tomorrow", "tomorrow morning", "end of tomorrow"):
        return ref + timedelta(days=1)
    if re.search(r"day after tomorrow", p):
        return ref + timedelta(days=2)
    if re.search(r"end of (the |this )?week|this week|eow", p):
        return ref + timedelta(days=(4 - ref.weekday()) % 7)
    if re.search(r"end of next week|next week", p):
        return _weekday_of_next_week(ref, 4)
    if re.search(r"end of (the |this )?month|this month|eom", p):
        return _end_of_month(ref)
    if re.search(r"next month", p):
        return _end_of_month(_end_of_month(ref) + timedelta(days=1))
    m = re.search(r"in (\w+) (day|week)s?", p)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS.get(m.group(1))
        if n:
            return ref + timedelta(days=n * (7 if m.group(2) == "week" else 1))
    for i, day in enumerate(WEEKDAYS):
        if re.search(rf"\b{day}\b", p):
            if re.search(rf"next {day}", p):
                return _weekday_of_next_week(ref, i)
            return _next_weekday(ref, i)
    if not re.search(r"\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", p):
        return None
    try:
        default = ref.replace(day=1) if not re.search(r"\b\d{1,2}(st|nd|rd|th)?\b", p) else ref
        parsed = dateparser.parse(p, fuzzy=True, default=_dt(default)).date()
    except (ValueError, OverflowError):
        return None
    if parsed < ref and not re.search(r"\b\d{4}\b", p):
        parsed = parsed.replace(year=parsed.year + 1)
    return parsed


def _dt(d: date):
    from datetime import datetime

    return datetime(d.year, d.month, d.day)


RELATIVE_DATE_PATTERN = re.compile(
    r"\b(?:today|tonight|tomorrow|(?:end of )?(?:this |next )?(?:week|month)|"
    r"(?:next )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"in \w+ (?:days?|weeks?))\b",
    re.IGNORECASE,
)


def relative_phrases(text: str) -> list[str]:
    return [m.group(0) for m in RELATIVE_DATE_PATTERN.finditer(text)]
