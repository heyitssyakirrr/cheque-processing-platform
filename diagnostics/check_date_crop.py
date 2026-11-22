"""See where the date crop lands, without running any model.

Re-crops the date box of every cheque in an already processed batch using the
current date-crop settings, and writes one picture per cheque so you can check
by eye that all six digits are inside, with nothing cut at the top, bottom or
right. Nothing is re-read: it uses the preprocessed image and the TARIKH/DATE
label positions saved by job 2, so the batch must have been run with
SAVE_ARTIFACTS=true.

Usage (from the project folder):
    python diagnostics/check_date_crop.py --batch-id 20261113

Writes data/batches/<batch-id>/date crop check/:
    <run>.png         NEW crop (current settings), WHERE it sits on the
                      cheque (red box), and the OLD crop.png saved when the
                      batch ran, when there is one
    _all_new.png      every NEW crop stacked in one picture
    _summary.csv      per cheque: image size, printed TARIKH width (the
                      crop's measuring unit) and that width as a fraction of
                      the image width, crop box, and how it was found
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import date_extraction  # noqa: E402
from app.services.date_extraction import find_crop_box  # noqa: E402
from app.settings import settings  # noqa: E402

OUTPUT_FOLDER_NAME = "date crop check"
SCALE = 2.5  # the same upscale job 2 applies before saving date/crop.png
LABEL_HEIGHT = 34
WHITE, BLACK, BLUE, GREY, RED = (255, 255, 255), (0, 0, 0), (200, 90, 0), (150, 150, 150), (0, 0, 230)


def _labelled(image: np.ndarray, text: str, width: int) -> np.ndarray:
    """Image with a caption bar on top, padded on the right to `width`."""
    bar = np.full((LABEL_HEIGHT, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (6, LABEL_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
    framed = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=BLUE)
    framed = cv2.copyMakeBorder(framed, 0, 0, 0, max(0, width - framed.shape[1]), cv2.BORDER_CONSTANT, value=WHITE)
    return np.vstack([bar, framed[:, :width]])


def _check_run(run_dir: Path, export_dir: Path) -> tuple[dict, np.ndarray | None]:
    row = {
        "run": run_dir.name, "image_size": "", "tarikh_width_px": "", "tarikh_width_vs_page": "",
        "crop_box": "", "found_by": "", "label_text": "", "note": "",
    }
    preprocessed = run_dir / "preprocessed" / "cheque.png"
    if not preprocessed.is_file():
        row["note"] = "no preprocessed/cheque.png (run with SAVE_ARTIFACTS=true)"
        return row, None
    image = cv2.imread(str(preprocessed))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    lines = result.get("text", {}).get("detections", [])
    if not lines:
        row["note"] = "no OCR text saved; fixed upper-right zone used"

    height, width = image.shape[:2]
    row["image_size"] = f"{width}x{height}"
    (x0, y0, x1, y1), found_by, label_text, _ = find_crop_box(image, lines)
    row.update(crop_box=f"{x0},{y0},{x1},{y1}", found_by=found_by, label_text=label_text)
    tarikh, date = date_extraction._find_label_lines(lines, image.shape)
    unit = float(y1 - y0)
    if tarikh is not None:
        tx0, _, tx1, _ = date_extraction._bounds(tarikh["box"])
        row.update(tarikh_width_px=round(tx1 - tx0), tarikh_width_vs_page=round((tx1 - tx0) / width, 4))
    if tarikh is not None or date is not None:
        unit = date_extraction._label_geometry(tarikh, date, image.shape)[0]
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        row["note"] = "empty crop"
        return row, None
    new = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)

    old_path = run_dir / "date" / "crop.png"
    old = cv2.imread(str(old_path)) if old_path.is_file() else None
    # The crop in its surroundings: label on the left, cheque edge on the
    # right, so a crop in the wrong place is obvious even when it looks clean.
    cx0, cy0 = max(0, int(x0 - 1.6 * unit)), max(0, int(y0 - 0.6 * unit))
    cx1, cy1 = min(width, int(x1 + 0.6 * unit)), min(height, int(y1 + 0.6 * unit))
    context = image[cy0:cy1, cx0:cx1].copy()
    cv2.rectangle(context, (x0 - cx0, y0 - cy0), (x1 - cx0 - 1, y1 - cy0 - 1), RED, 2)
    context = cv2.resize(context, None, fx=SCALE * 0.6, fy=SCALE * 0.6, interpolation=cv2.INTER_CUBIC)

    panel_width = max(new.shape[1], context.shape[1], old.shape[1] if old is not None else 0) + 6
    picture = [
        _labelled(new, f"NEW  {run_dir.name}", panel_width),
        _labelled(context, "WHERE  (red box = new crop)", panel_width),
    ]
    if old is not None:
        picture.append(_labelled(old, "OLD  (saved when the batch ran)", panel_width))
    cv2.imwrite(str(export_dir / f"{run_dir.name}.png"), np.vstack(picture))
    return row, _labelled(new, run_dir.name, new.shape[1] + 6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-crop each cheque's date box with the current settings.")
    parser.add_argument("--batch-id", default=f"{datetime.now():%Y%m%d}", help="Date in YYYYMMDD form")
    args = parser.parse_args(argv)
    if not (len(args.batch_id) == 8 and args.batch_id.isdigit()):
        parser.error("--batch-id must be YYYYMMDD")

    batch_dir = settings.batch_dir(args.batch_id)
    run_dirs = sorted(path.parent for path in batch_dir.glob("*/result.json"))
    if not run_dirs:
        print(f"No processed cheques with result.json in {batch_dir} (was SAVE_ARTIFACTS=true?)")
        return 1

    export_dir = batch_dir / OUTPUT_FOLDER_NAME
    export_dir.mkdir(parents=True, exist_ok=True)
    rows, sheet = [], []
    for run_dir in run_dirs:
        row, picture = _check_run(run_dir, export_dir)
        rows.append(row)
        if picture is not None:
            sheet.append(picture)
        print(f"{row['run']}: {row['crop_box'] or '-'} {row['note']}".rstrip())

    if sheet:
        width = max(picture.shape[1] for picture in sheet)
        padded = [
            cv2.copyMakeBorder(p, 0, 8, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=GREY) for p in sheet
        ]
        cv2.imwrite(str(export_dir / "_all_new.png"), np.vstack(padded))
    with (export_dir / "_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(sheet)} of {len(rows)} crop(s) written to {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
