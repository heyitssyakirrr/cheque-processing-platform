"""One log per job per day: data/logs/<YYYYMMDD>_<job>.log, plus the console."""

from __future__ import annotations

import logging
import sys

from Cheque_Processing.app.settings import settings


def setup_job_log(job: str, batch_id: str) -> logging.Logger:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    path = settings.logs_dir / f"{batch_id}_{job}.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    # paramiko logs every connection at INFO; keep only its warnings.
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logger = logging.getLogger(job)
    logger.info("Log: %s", path)
    return logger
