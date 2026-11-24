"""Date helpers shared by the reader, job 2, the report and the tools.

Kept free of heavy imports (no OpenCV/Torch) so the report and tests can
use them cheaply.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re


def parse_date(value: str | None) -> dt.date | None:
    """Turn a read date into a real date, or None if it is not one.

    Accepts DDMMYY / DDMMYYYY, with or without "/", "-", "." or spaces
    between the parts ("02/08/11", "020811", "02/08/2011"). Two-digit
    years are 20YY. "88/55/28" -> None (no month 55); "31/02/11" -> None.
    """
    digits = re.sub(r"[\s/.\-]", "", str(value or ""))
    if not digits.isdigit() or len(digits) not in (6, 8):
        return None
    day, month = int(digits[0:2]), int(digits[2:4])
    year = int(digits[4:]) + (2000 if len(digits) == 6 else 0)
    if not (1 <= month <= 12 and 1 <= year <= 9999):
        return None
    if not 1 <= day <= calendar.monthrange(year, month)[1]:
        return None
    return dt.date(year, month, day)


def is_valid_date(value: str | None) -> bool:
    """True when `value` is a real calendar date (see `parse_date`)."""
    return parse_date(value) is not None


def is_flagged(signature_detected: str, valid_date: str) -> str:
    """YES when the cheque needs a person to look at it: no signature found,
    or the date read is not a real calendar date. Both inputs are YES/NO."""
    return "YES" if signature_detected != "YES" or valid_date != "YES" else "NO"
