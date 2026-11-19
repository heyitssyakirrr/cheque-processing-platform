"""Delete old day folders: fetched images after INCOMING_RETENTION_DAYS, batch
output after BATCH_RETENTION_DAYS (0 keeps them forever).

Only days whose batch finished (data/batches/<date>/_BATCH_COMPLETE.json,
written by job 2 when no cheque is left to retry) are deleted. An unfinished
day past its retention is kept and logged, so nothing is lost silently.
Only folders named YYYYMMDD are touched.
"""

from __future__ import annotations

import logging
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

from app.settings import settings

BATCH_MARKER = "_BATCH_COMPLETE.json"

logger = logging.getLogger("retention")


def _day(name: str) -> date | None:
    if len(name) != 8 or not name.isdigit():
        return None
    try:
        return datetime.strptime(name, "%Y%m%d").date()
    except ValueError:
        return None


def _old_days(parent: Path, keep_days: int, today: date) -> list[Path]:
    if keep_days <= 0 or not parent.is_dir():
        return []
    cutoff = today - timedelta(days=keep_days)
    return sorted(path for path in parent.iterdir()
                  if path.is_dir() and (day := _day(path.name)) is not None and day < cutoff)


def _finished(batch_id: str) -> bool:
    return (settings.batch_dir(batch_id) / BATCH_MARKER).is_file()


def apply_retention(today: date | None = None) -> dict[str, list[str]]:
    """Returns what was deleted and what was kept because it never finished."""
    today = today or date.today()
    report: dict[str, list[str]] = {"incoming_deleted": [], "batches_deleted": [], "kept_unfinished": []}
    for kind, parent, keep in (("incoming", settings.incoming_dir, settings.incoming_retention_days),
                               ("batches", settings.batches_dir, settings.batch_retention_days)):
        for path in _old_days(parent, keep, today):
            if not _finished(path.name):
                report["kept_unfinished"].append(f"{kind}/{path.name}")
                continue
            shutil.rmtree(path)
            report[f"{kind}_deleted"].append(path.name)
    if report["incoming_deleted"]:
        logger.info("Retention: deleted fetched images of %s", ", ".join(report["incoming_deleted"]))
    if report["batches_deleted"]:
        logger.info("Retention: deleted batch output of %s", ", ".join(report["batches_deleted"]))
    if report["kept_unfinished"]:
        logger.warning("Retention: kept %s (past retention, but the batch never finished)",
                       ", ".join(report["kept_unfinished"]))
    return report
