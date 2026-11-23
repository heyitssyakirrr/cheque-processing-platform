"""Show, step by step, how the date area is found on each cheque, and compare
the ways of cropping it. No OCR or signature model is run: it uses the
preprocessed image and the OCR text boxes job 2 saved, so the batch must have
been run with SAVE_ARTIFACTS=true. With --read (default) the small digit
model reads each candidate crop, which takes about a second per cheque.

Usage (from the project folder):
    python diagnostics/explore_date_area.py --batch-id 20261113
    python diagnostics/explore_date_area.py --batch-id 20261113 --only 009 012

Writes data/batches/<batch-id>/date area explore/:
    <run>.png    1. PLAN: everything OCR found near the date (blue, with its
                    text), TARIKH/DATE (green), the guide-letter search band
                    (yellow), the D D M M Y Y letters found (cyan dots), the
                    six boxes fitted to them (cyan lines), the label-only crop
                    (orange) and the crop job 2 will use (red).
                 2. For each candidate crop: the crop, then what the digit
                    reader sees in it (white = ink after its automatic
                    threshold; green lines = the row it picked; red lines =
                    where it splits the six digits), and the reading.
    _summary.csv one row per cheque: image size, TARIKH/DATE text and boxes,
                 TARIKH width (the unit), guide letters found, fitted box
                 start and spacing (in TARIKH widths), both crop boxes, and
                 both readings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import date_extraction, digit_segmentation  # noqa: E402
from app.settings import settings  # noqa: E402

OUTPUT_FOLDER_NAME = "date area explore"
SCALE = 2.5  # the same upscale job 2 applies before reading the crop
CAPTION = 34
WHITE, BLACK, BLUE, GREEN, DARK_GREEN = (255, 255, 255), (0, 0, 0), (200, 90, 0), (0, 170, 0), (0, 100, 0)
YELLOW, CYAN, ORANGE, RED, GREY = (0, 200, 230), (200, 200, 0), (0, 140, 255), (0, 0, 230), (150, 150, 150)


def _caption(image: np.ndarray, text: str, width: int) -> np.ndarray:
    bar = np.full((CAPTION, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (6, CAPTION - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLACK, 2)
    framed = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=GREY)
    pad = max(0, width - framed.shape[1])
    framed = cv2.copyMakeBorder(framed, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=WHITE)
    return np.vstack([bar, framed[:, :width]])


def _rect(canvas, box, offset, scale, colour, thickness=2):
    (x0, y0, x1, y1), (ox, oy) = box, offset
    cv2.rectangle(canvas, (int((x0 - ox) * scale), int((y0 - oy) * scale)),
                  (int((x1 - ox) * scale), int((y1 - oy) * scale)), colour, thickness)


def _plan_panel(image: np.ndarray, plan: dict, lines: list[dict]) -> np.ndarray:
    """The top-right of the cheque with every measurement drawn on it."""
    height, width = image.shape[:2]
    unit = plan["unit"] or 0.05 * width
    final = plan["box"]
    label_lefts = [date_extraction._bounds(l["box"])[0] for l in (plan["tarikh"], plan["date"]) if l]
    left = min([final[0], *label_lefts])  # no labels found: start at the crop
    region = (max(0, int(left - 0.5 * unit)), max(0, int(final[1] - 1.2 * unit)),
              width, min(height, int(final[3] + 1.0 * unit)))
    scale = 1.4
    rx0, ry0, rx1, ry1 = region
    canvas = cv2.resize(image[ry0:ry1, rx0:rx1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    offset = (rx0, ry0)

    for line in lines:
        bx0, by0, bx1, by1 = date_extraction._bounds(line["box"])
        if bx1 < rx0 or by1 < ry0 or by0 > ry1:
            continue
        colour = GREEN if line is plan["tarikh"] else DARK_GREEN if line is plan["date"] else BLUE
        _rect(canvas, (bx0, by0, bx1, by1), offset, scale, colour, 2 if colour != BLUE else 1)
        label = f"{line['text']} {float(line['confidence']):.2f}"
        cv2.putText(canvas, label, (int((bx0 - rx0) * scale), max(12, int((by0 - ry0) * scale) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    grid = plan.get("grid")
    if grid:
        _rect(canvas, grid["band"], offset, scale, YELLOW, 1)
        for cx, top in grid["letters"]:
            cv2.circle(canvas, (int((cx - rx0) * scale), int((top - ry0) * scale)), 3, CYAN, -1)
        if not grid.get("rejected"):
            for k in range(7):
                x = grid["start_x"] + (k - 0.5) * grid["pitch_px"]
                cv2.line(canvas, (int((x - rx0) * scale), int((final[1] - ry0) * scale)),
                         (int((x - rx0) * scale), int((grid["letters_top"] - ry0) * scale)), CYAN, 1)
    if plan.get("label_box"):
        _rect(canvas, plan["label_box"], offset, scale, ORANGE, 2)
    _rect(canvas, final, offset, scale, RED, 3)
    return canvas


def _reader_view(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The crop as job 2 upscales it, and the reader's ink mask with its row
    (green) and digit splits (red) drawn on."""
    big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    seen = digit_segmentation.explain(big, settings.digit_count)
    mask = cv2.cvtColor(seen["binary"], cv2.COLOR_GRAY2BGR)
    w = mask.shape[1]
    cv2.line(mask, (0, seen["top"]), (w, seen["top"]), GREEN, 2)
    cv2.line(mask, (0, seen["bottom"] - 1), (w, seen["bottom"] - 1), GREEN, 2)
    for x0, x1 in seen["lanes"]:
        for x in (x0, x1):
            cv2.line(mask, (seen["left"] + x, seen["top"]), (seen["left"] + x, seen["bottom"]), RED, 2)
    return big, mask


def _read(crop: np.ndarray) -> str:
    from app.services import digit_classifier

    big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    with tempfile.TemporaryDirectory() as folder:
        slices = digit_segmentation.slice_digits(big, Path(folder), digit_count=settings.digit_count)
    if not slices:
        return "no handwriting (blank date)"
    top = digit_classifier.classify_digits(slices, top_k=1)
    digits = "".join(candidates[0][0] for candidates in top)
    pretty = f"{digits[0:2]}/{digits[2:4]}/{digits[4:6]}" if len(digits) == 6 else digits
    return f"{pretty} ({'valid' if date_extraction.is_valid_date(digits) else 'invalid'})"


def _explore(run_dir: Path, export_dir: Path, read: bool) -> dict:
    row: dict = {"run": run_dir.name}
    image = cv2.imread(str(run_dir / "preprocessed" / "cheque.png"))
    if image is None:
        row["note"] = "no preprocessed/cheque.png (run with SAVE_ARTIFACTS=true)"
        return row
    lines = json.loads((run_dir / "result.json").read_text(encoding="utf-8")).get("text", {}).get("detections", [])
    plan = date_extraction.plan_crop(image, lines)
    height, width = image.shape[:2]
    row["image_size"] = f"{width}x{height}"
    for name in ("tarikh", "date"):
        line = plan[name]
        row[f"{name}_text"] = line["text"] if line else ""
        row[f"{name}_box"] = ",".join(str(int(v)) for v in date_extraction._bounds(line["box"])) if line else ""
    unit = plan["unit"]
    row["unit_px"] = round(unit, 1) if unit else ""
    row["unit_vs_page"] = round(unit / width, 4) if unit else ""
    grid = plan.get("grid") or {}
    row.update(
        letters_found=len(grid.get("letters", [])), letters_fitted=grid.get("inliers", ""),
        box_start_units=round(grid["start_units"], 3) if grid else "",
        box_pitch_units=round(grid["pitch_units"], 3) if grid else "",
        letter_threshold=grid.get("threshold", ""),
        label_box=",".join(map(str, plan["label_box"])) if plan.get("label_box") else "",
        final_box=",".join(map(str, plan["box"])), mode=plan["mode"], notes="; ".join(plan["notes"]),
    )

    panels = [("PLAN  " + run_dir.name + "   red = crop used, orange = label-only crop", _plan_panel(image, plan, lines))]
    candidates = [("used (" + plan["mode"] + ")", plan["box"])]
    if plan.get("label_box") and tuple(plan["label_box"]) != tuple(plan["box"]):
        candidates.append(("label-only", plan["label_box"]))
    for name, (x0, y0, x1, y1) in candidates:
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        big, mask = _reader_view(crop)
        reading = _read(crop) if read else ""
        row["reading_" + name.split()[0].replace("-", "_")] = reading
        panels.append((f"{name.upper()} crop   {reading}", big))
        panels.append((f"{name} - what the digit reader sees", mask))

    width_out = max(p.shape[1] for _, p in panels) + 4
    cv2.imwrite(str(export_dir / f"{run_dir.name}.png"),
                np.vstack([_caption(p, text, width_out) for text, p in panels]))
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show how the date area is found on each cheque.")
    parser.add_argument("--batch-id", default=f"{datetime.now():%Y%m%d}", help="Date in YYYYMMDD form")
    parser.add_argument("--only", nargs="*", help="Run folder names or prefixes to include, e.g. 009 012")
    parser.add_argument("--no-read", action="store_true", help="Skip the digit model (crops only)")
    args = parser.parse_args(argv)
    if not (len(args.batch_id) == 8 and args.batch_id.isdigit()):
        parser.error("--batch-id must be YYYYMMDD")

    batch_dir = settings.batch_dir(args.batch_id)
    run_dirs = sorted(path.parent for path in batch_dir.glob("*/result.json"))
    if args.only:
        run_dirs = [d for d in run_dirs if any(d.name.startswith(prefix) for prefix in args.only)]
    if not run_dirs:
        print(f"No processed cheques with result.json in {batch_dir} (was SAVE_ARTIFACTS=true?)")
        return 1

    export_dir = batch_dir / OUTPUT_FOLDER_NAME
    export_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in run_dirs:
        try:
            row = _explore(run_dir, export_dir, read=not args.no_read)
        except Exception as error:  # one odd cheque must not stop the rest
            row = {"run": run_dir.name, "note": f"tool error: {type(error).__name__}: {error}"}
        rows.append(row)
        print(f"{row['run']}: {row.get('mode', '-')} {row.get('reading_used', '')} {row.get('notes', row.get('note', ''))}")

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (export_dir / "_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grid_rows = [r for r in rows if r.get("mode") == "grid"]
    print(f"\n{len(grid_rows)} of {len(rows)} cheque(s) placed by the D D M M Y Y letters; "
          f"the rest by the TARIKH/DATE label alone.")
    if grid_rows:
        pitches = [r["box_pitch_units"] for r in grid_rows]
        starts = [r["box_start_units"] for r in grid_rows]
        print(f"Box spacing {min(pitches)}-{max(pitches)} and first box centre {min(starts)}-{max(starts)} "
              f"TARIKH widths right of TARIKH.")
    print(f"Pictures and _summary.csv in {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
