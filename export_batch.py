"""Collect a processed batch's output CSV and original cheque PNGs into one folder.

Usage: ``python export_batch.py --batch-id 20260916`` creates
``data/batches/20260916/batch output/`` containing ``20260916_output.csv`` and
every ``<run_id>/original/cheque.png`` copied as ``<run_id>.png``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime

from Cheque_Processing.app.services.date_extraction import is_valid_date
from Cheque_Processing.app.settings import settings

OUTPUT_FOLDER_NAME = "batch output"


def _copy_with_valid_date(source, target) -> None:
    """Copy the output CSV, adding a valid_date (YES/NO) column right after
    date when the batch was processed before that column existed."""
    with source.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "valid_date" not in fields and "date" in fields:
        fields.insert(fields.index("date") + 1, "valid_date")
        for row in rows:
            row["valid_date"] = "YES" if is_valid_date(row.get("date", "")) else "NO"
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a batch's output CSV and original cheque PNGs.")
    parser.add_argument("--batch-id", default=f"{datetime.now():%Y%m%d}", help="Date in YYYYMMDD form")
    args = parser.parse_args(argv)
    batch_id = args.batch_id
    if not (len(batch_id) == 8 and batch_id.isdigit()):
        parser.error("--batch-id must be YYYYMMDD")

    batch_dir = settings.batch_dir(batch_id)
    output_csv = batch_dir / f"{batch_id}_output.csv"
    if not output_csv.is_file():
        print(f"Output CSV not found: {output_csv}")
        return 1

    export_dir = batch_dir / OUTPUT_FOLDER_NAME
    export_dir.mkdir(parents=True, exist_ok=True)
    _copy_with_valid_date(output_csv, export_dir / output_csv.name)

    copied = 0
    for png in sorted(batch_dir.glob("*/original/cheque.png")):
        run_dir = png.parent.parent
        if run_dir == export_dir:
            continue
        shutil.copy2(png, export_dir / f"{run_dir.name}.png")
        copied += 1

    print(f"Exported {output_csv.name} and {copied} cheque PNG(s) to {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

