"""Collect a processed batch's results into one folder, and score them
against an answer key when there is one.

    python export_batch.py --batch-id 20261113
        data/batches/20261113/batch output/ gets:
          20261113_output.csv   the results, always (older layouts converted to
                                the current columns, see app/report.py)
          <run>.png             each original cheque image
          20261113_scored.csv   only when data/batches/20261113/answer_key.csv
                                exists: model vs answer, 1/0 accuracy per
                                cheque, TOTAL and PERCENTAGE rows

    python export_batch.py --batch-id 20261113 --write-answer-key
        creates data/batches/20261113/answer_key.csv with every file name, for
        you to fill in: signature YES/NO, date (310811, 31/08/2011, ... or
        NO DATE), optional note. Then run the export again.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime

from Cheque_Processing.app import report
from Cheque_Processing.app.settings import settings

OUTPUT_FOLDER_NAME = "batch output"
ANSWER_KEY_NAME = "answer_key.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a batch's results, scored when an answer key exists.")
    parser.add_argument("--batch-id", default=f"{datetime.now():%Y%m%d}", help="Date in YYYYMMDD form")
    parser.add_argument("--write-answer-key", action="store_true",
                        help=f"Create {ANSWER_KEY_NAME} in the batch folder to fill in, then stop")
    args = parser.parse_args(argv)
    batch_id = args.batch_id
    if not (len(batch_id) == 8 and batch_id.isdigit()):
        parser.error("--batch-id must be YYYYMMDD")

    batch_dir = settings.batch_dir(batch_id)
    output_csv = batch_dir / f"{batch_id}_output.csv"
    if not output_csv.is_file():
        print(f"Output CSV not found: {output_csv}")
        return 1
    rows = report.read_output(output_csv)
    answer_key = batch_dir / ANSWER_KEY_NAME

    if args.write_answer_key:
        if answer_key.exists():
            print(f"{answer_key} already exists; not overwritten.")
            return 1
        report.write_answer_template(answer_key, rows)
        print(f"Created {answer_key} with {len(rows)} file name(s). Fill in signature (YES/NO), "
              f"date (e.g. 310811, 31/08/2011, or NO DATE) and an optional note, then run the export again.")
        return 0

    export_dir = batch_dir / OUTPUT_FOLDER_NAME
    export_dir.mkdir(parents=True, exist_ok=True)
    report.write_csv(export_dir / output_csv.name, report.OUTPUT_FIELDS, rows)

    copied = 0
    for png in sorted(batch_dir.glob("*/original/cheque.png")):
        run_dir = png.parent.parent
        if run_dir == export_dir:
            continue
        shutil.copy2(png, export_dir / f"{run_dir.name}.png")
        copied += 1
    print(f"Exported {output_csv.name} ({len(rows)} cheques) and {copied} cheque PNG(s) to {export_dir}")

    scored_csv = export_dir / f"{batch_id}_scored.csv"
    scored_csv.unlink(missing_ok=True)  # never leave a scored report from an earlier answer key
    if answer_key.is_file():
        try:
            answers = report.load_answers(answer_key)
            scored = report.scored_rows(rows, answers)
        except ValueError as error:
            print(f"No scored report: {error}")
            return 1
        report.write_csv(scored_csv, report.SCORED_FIELDS, scored)
        total, percentage = scored[-2], scored[-1]
        scored_count = sum(1 for r in rows if r["file_name"] in answers)
        print(f"Scored {scored_count} cheque(s): signature {total['signature_accuracy']} "
              f"({percentage['signature_accuracy']}), date {total['date_accuracy']} ({percentage['date_accuracy']}), "
              f"flagged {total['flagged']} ({percentage['flagged']}) -> {scored_csv.name}")
        missing = [r["file_name"] for r in rows if r["file_name"] not in answers]
        if missing:
            print(f"Not in the answer key (shown, not scored): {len(missing)} cheque(s), e.g. {missing[0]}")
        exported = {r["file_name"] for r in rows}
        unused = sorted(name for name in answers if name not in exported)
        if unused:
            print(f"In the answer key but not in the output (failed or not processed; see "
                  f"{batch_id}_failed.csv): {len(unused)} cheque(s), e.g. {unused[0]}")
    else:
        print(f"No {ANSWER_KEY_NAME} in the batch folder, so no scored report (normal outside UAT).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
