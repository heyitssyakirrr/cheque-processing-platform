"""Which day (YYYYMMDD) a job works on.

Both jobs take ``--date YYYYMMDD`` for reruns and catch-up. Without it they
use today minus DATE_OFFSET_DAYS. Once the batch date is confirmed (for
example a warehouse function, like get_batch_date_dwh in TransactionODD),
only ``default_batch_date`` needs to change.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

from Cheque_Processing.app.settings import settings


def default_batch_date(today: date | None = None) -> str:
    today = today or date.today()
    return (today - timedelta(days=settings.date_offset_days)).strftime("%Y%m%d")


def parse_batch_date(value: str) -> str:
    """argparse type for --date: a real calendar day in YYYYMMDD form."""
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date in YYYYMMDD form") from None
    if len(value) != 8:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date in YYYYMMDD form")
    return value


def add_date_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--date", type=parse_batch_date, default=None,
        help="day to work on, YYYYMMDD (default: today minus DATE_OFFSET_DAYS)",
    )
