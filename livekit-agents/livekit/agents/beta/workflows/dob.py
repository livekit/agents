from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time


@dataclass
class GetDOBResult:
    date_of_birth: date
    time_of_birth: time | None = None


def normalize_date_of_birth(year: int, month: int, day: int) -> date:
    """Normalize two-digit birth years and reject invalid or future dates."""
    if 0 <= year < 100:
        year += 2000 if year <= date.today().year % 100 else 1900
    value = date(year, month, day)
    if value > date.today():
        raise ValueError("Date of birth cannot be in the future")
    return value
