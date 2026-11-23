"""Experiment: how the two ink settings change what the digit reader reads.

  DIGIT_INK_CEILING       strokes = pixels darker than min(Otsu, ceiling).
                          255 = plain Otsu (no cap). Lower = thinner strokes,
                          less grey background taken as ink.
  DIGIT_INK_MIN_CONTRAST  only decides "blank date": a pixel must also be
                          this much darker than the crop's background.
                          0 = off. Never changes the strokes themselves.

Every combination is run on every cheque of a processed batch (no OCR or
signature model; the crop is placed exactly as job 2 places it). Give it the
real answers and it scores each combination.

Usage (from the project folder):
    1. python diagnostics/compare_ink_settings.py --batch-id 20261113 --write-truth
       writes data/batches/20261113/ink settings/truth.csv with one row per
       cheque. Fill in the "expected" column: the date as six digits
       (310811), "blank" when no date is written, or "skip" (stamped date,
       unreadable) to leave it out of the score.
    2. python diagnostics/compare_ink_settings.py --batch-id 20261113
       (optional: --ceilings 255 190 170 150 --contrasts 0 30 40 50 60)

Writes data/batches/<batch-id>/ink settings/:
    readings.csv   one row per cheque, one column per combination
    scores.csv     per combination: dates right, blanks right, wrong dates
                   that look valid (the dangerous kind), total right
    <run>.png      the crop and the strokes the reader keeps at each ceiling
                   (white = ink), to see whether faint edges are lost
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

OUTPUT_FOLDER_NAME = "ink settings"
SCALE = 2.5
CAPTION = 30


def _caption(image: np.ndarray, text: str, width: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bar = np.full((CAPTION, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (6, CAPTION - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    pad = max(0, width - image.shape[1])
    framed = cv2.copyMakeBorder(image, 0, 4, 0, pad, cv2.BORDER_CONSTANT, value=(150, 150, 150))
    return np.vstack([bar, framed[:, :width]])


def _read(big: np.ndarray, ceiling: float, contrast: float) -> tuple[str, str]:
    """(reading for the table, raw digits or 'blank')."""
    from app.services import digit_classifier

    saved = settings.digit_ink_ceiling, settings.digit_ink_min_contrast
    settings.digit_ink_ceiling, settings.digit_ink_min_contrast = ceiling, contrast
    try:
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        reason = digit_segmentation.blank_reason(gray)
        if reason:
            return f"blank ({reason})", "blank"
        with tempfile.TemporaryDirectory() as folder:
            slices = digit_segmentation.slice_digits(big, Path(folder), digit_count=settings.digit_count)
        if not slices:
            return "blank", "blank"
        digits = "".join(c[0][0] for c in digit_classifier.classify_digits(slices, top_k=1))
        pretty = f"{digits[0:2]}/{digits[2:4]}/{digits[4:6]}" if len(digits) == 6 else digits
        return pretty + ("" if date_extraction.is_valid_date(digits) else " (invalid)"), digits
    finally:
        settings.digit_ink_ceiling, settings.digit_ink_min_contrast = saved


def _strokes(big: np.ndarray, ceiling: float) -> np.ndarray:
    saved = settings.digit_ink_ceiling
    settings.digit_ink_ceiling = ceiling
    try:
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        return 255 * (gray <= digit_segmentation.ink_threshold(gray)).astype(np.uint8)
    finally:
        settings.digit_ink_ceiling = saved


def _load_truth(path: Path) -> dict[str, str]:
    truth = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                expected = (row.get("expected") or "").strip().lower().replace("/", "")
                if expected:
                    truth[row["run"]] = expected
    return truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare digit-reader ink settings on a processed batch.")
    parser.add_argument("--batch-id", default=f"{datetime.now():%Y%m%d}", help="Date in YYYYMMDD form")
    parser.add_argument("--only", nargs="*", help="Run folder names or prefixes to include, e.g. 009 012")
    parser.add_argument("--ceilings", nargs="*", type=float, default=[255, 190, 170, 150])
    parser.add_argument("--contrasts", nargs="*", type=float, default=[0, 30, 40, 50, 60])
    parser.add_argument("--write-truth", action="store_true", help="Write truth.csv to fill in, then stop")
    parser.add_argument("--no-pictures", action="store_true")
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
    out = batch_dir / OUTPUT_FOLDER_NAME
    out.mkdir(parents=True, exist_ok=True)
    truth_path = out / "truth.csv"

    if args.write_truth:
        if truth_path.exists():
            print(f"{truth_path} already exists; not overwritten.")
            return 1
        with truth_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["run", "expected"])
            writer.writerows([d.name, ""] for d in run_dirs)
        print(f"Fill in the expected column of {truth_path}: six digits (310811), blank, or skip.")
        return 0

    truth = _load_truth(truth_path)
    variants = [(c, k) for c in args.ceilings for k in args.contrasts]
    names = [f"ceiling{c:g}_contrast{k:g}" for c, k in variants]
    rows, scores = [], {name: {"dates_right": 0, "blanks_right": 0, "wrong_but_valid": 0, "scored": 0}
                        for name in names}

    for run_dir in run_dirs:
        image = cv2.imread(str(run_dir / "preprocessed" / "cheque.png"))
        if image is None:
            continue
        lines = json.loads((run_dir / "result.json").read_text(encoding="utf-8")).get("text", {}).get("detections", [])
        x0, y0, x1, y1 = date_extraction.plan_crop(image, lines)["box"]
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        expected = truth.get(run_dir.name, "")
        row = {"run": run_dir.name, "expected": expected}
        for (ceiling, contrast), name in zip(variants, names):
            shown, raw = _read(big, ceiling, contrast)
            row[name] = shown
            if expected and expected != "skip":
                score = scores[name]
                score["scored"] += 1
                if expected == "blank":
                    score["blanks_right"] += raw == "blank"
                elif raw == expected:
                    score["dates_right"] += 1
                elif raw != "blank" and date_extraction.is_valid_date(raw):
                    score["wrong_but_valid"] += 1
        rows.append(row)
        print(f"{run_dir.name}: expected {expected or '?'}; "
              + "; ".join(f"{n}={row[n]}" for n in names if n.endswith(f"_contrast{args.contrasts[-1]:g}")))

        if not args.no_pictures:
            panels = [("crop", big)] + [(f"strokes at ceiling {c:g} (255 = plain Otsu)", _strokes(big, c))
                                        for c in args.ceilings]
            width = max(p.shape[1] for _, p in panels)
            cv2.imwrite(str(out / f"{run_dir.name}.png"), np.vstack([_caption(p, t, width) for t, p in panels]))

    with (out / "readings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["run", "expected", *names])
        writer.writeheader()
        writer.writerows(rows)

    if not truth:
        print(f"\nNo answers in {truth_path} yet: run with --write-truth, fill it in, run again to get scores.")
        print(f"Readings in {out / 'readings.csv'}")
        return 0
    ranked = sorted(names, key=lambda n: (-(scores[n]["dates_right"] + scores[n]["blanks_right"]),
                                          scores[n]["wrong_but_valid"]))
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["setting", "dates_right", "blanks_right", "total_right", "wrong_but_valid", "scored"])
        for name in ranked:
            s = scores[name]
            writer.writerow([name, s["dates_right"], s["blanks_right"], s["dates_right"] + s["blanks_right"],
                             s["wrong_but_valid"], s["scored"]])
    print("\nBest settings (most right, then fewest wrong dates that look valid):")
    for name in ranked[:6]:
        s = scores[name]
        print(f"  {name}: {s['dates_right'] + s['blanks_right']}/{s['scored']} right "
              f"({s['dates_right']} dates, {s['blanks_right']} blanks), {s['wrong_but_valid']} wrong-but-valid")
    print(f"Full table: {out / 'scores.csv'}; readings: {out / 'readings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
