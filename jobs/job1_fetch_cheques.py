"""Job 1: fetch one day's .IMG files from the source server (SFTP) into data/incoming/<date>/.

    python jobs/job1_fetch_cheques.py                    today minus DATE_OFFSET_DAYS
    python jobs/job1_fetch_cheques.py --date 20261119
    python jobs/job1_fetch_cheques.py --force            fetch a day again (re-checks every file)
    python jobs/job1_fetch_cheques.py --skip-trigger     by hand only: don't wait for the trigger files

Waits for every bank's trigger file (until FETCH_GIVE_UP_AT), copies the
files, then writes data/incoming/<date>/_FETCH_COMPLETE.json, which job 2
waits for. Nothing on the source server is changed.

Exit codes: 0 day fetched (or already fetched), 1 not ready or some files not
copied (run again later), 2 configuration problem (fix .env first).
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Cheque_Processing.app.batch_date import add_date_argument, default_batch_date  # noqa: E402
from Cheque_Processing.app.fetch import ConfigError, fetch_day  # noqa: E402
from Cheque_Processing.app.job_log import setup_job_log  # noqa: E402
from Cheque_Processing.app.settings import settings  # noqa: E402

logger = logging.getLogger("job1")


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()  # a crash inside a C library prints where, instead of dying silently
    parser = argparse.ArgumentParser(description="Job 1: fetch one day's cheque images from the source server.")
    add_date_argument(parser)
    parser.add_argument("--force", action="store_true", help="fetch the day again even if it is complete")
    parser.add_argument("--skip-trigger", action="store_true", help="don't wait for the trigger files (by hand only)")
    args = parser.parse_args(argv)
    batch_id = args.date or default_batch_date()
    setup_job_log("job1", batch_id)
    logger.info("Job 1 for %s from %s %s (banks %s)", batch_id, settings.source_transport,
                settings.sftp_host if settings.source_transport == "sftp" else settings.source_root,
                ",".join(settings.banks))
    try:
        code = fetch_day(batch_id, force=args.force, skip_trigger=args.skip_trigger).code
    except ConfigError as error:
        logger.error("%s", error)
        code = 2
    except Exception:  # network down, server refused, ...: worth another try later
        logger.exception("Job 1 failed")
        code = 1
    logger.info("Exit code %d", code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
