"""Find a printed date label with PaddleOCR, then classify each handwritten
digit in the date row with a fine-tuned CNN (grid-sliced, not TrOCR)."""

import calendar
import csv
import datetime as dt
from difflib import SequenceMatcher
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services import digit_classifier, digit_segmentation
from app.settings import settings


def _normalise(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _bounds(box: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def _score_candidates(
    lines: list[dict[str, Any]],
    aliases: tuple[str, ...],
    image_shape: tuple[int, ...],
    allow_weak_wn: bool = False,
) -> dict[str, Any] | None:
    """Find the best-scoring line matching any of `aliases`."""
    normalised_aliases = [_normalise(alias) for alias in aliases]
    height, width = image_shape[:2]
    best: tuple[float, dict[str, Any]] | None = None
    for line in lines:
        text = _normalise(str(line["text"]))
        if len(text) < 2:
            continue
        direct = [alias for alias in normalised_aliases if alias in text or text in alias]
        similarity = max(
            (SequenceMatcher(None, text, alias).ratio() for alias in normalised_aliases if len(alias) >= 3),
            default=0.0,
        )
        x0, y0, x1, y1 = _bounds(line["box"])
        weak_wn = (
            allow_weak_wn
            and text == "WN"
            and (x0 + x1) / 2 >= width * 0.55
            and (y0 + y1) / 2 <= height * 0.50
        )
        if not direct and similarity < 0.66 and not weak_wn:
            continue
        score = float(line["confidence"]) + similarity + (2.0 if text in direct else 0.0)
        if best is None or score > best[0]:
            best = (score, line)
    return best[1] if best else None


def _find_label_lines(
    lines: list[dict[str, Any]], image_shape: tuple[int, ...]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Locate TARIKH and DATE as the two independent, stacked printed lines
    they actually are on a Public Bank cheque (TARIKH above, DATE below)."""
    tarikh = _score_candidates(lines, settings.date_label_aliases_tarikh, image_shape)
    date = _score_candidates(
        lines,
        settings.date_label_aliases_date + settings.date_label_aliases_ambiguous,
        image_shape,
        allow_weak_wn=True,
    )

    if tarikh is not None and date is not None and tarikh is not date:
        tx0, ty0, tx1, ty1 = _bounds(tarikh["box"])
        dx0, dy0, dx1, dy1 = _bounds(date["box"])
        x_overlap = min(tx1, dx1) - max(tx0, dx0)
        vertically_ordered = dy0 >= ty0
        if x_overlap <= 0 or not vertically_ordered:
            if float(tarikh["confidence"]) >= float(date["confidence"]):
                date = None
            else:
                tarikh = None

    return tarikh, date


def _label_geometry(
    tarikh: dict[str, Any] | None,
    date: dict[str, Any] | None,
    image_shape: tuple[int, ...],
) -> tuple[float, float, float, str]:
    """Measure the printed TARIKH/DATE label to place the date boxes.

    Returns (unit, anchor_x, top_y, note):
      unit      the printed width of the word TARIKH, in pixels. Every date
                box measurement is a multiple of it, so the crop scales with
                the cheque however big the image is. (The old unit, the
                median height of every OCR line, also counted the
                handwritten digits and could be 50% off.)
      anchor_x  where the first date box starts (TARIKH's right edge)
      top_y     the top of TARIKH
    """
    width = image_shape[1]
    expected_unit = width * settings.date_label_unit_page_fraction
    note = ""
    if tarikh is not None:
        x0, y0, x1, _ = _bounds(tarikh["box"])
        letters = len(_normalise(str(tarikh["text"])))
        unit = x1 - x0
        if letters > 6:
            # OCR merged TARIKH with a handwritten digit ("TARIKH3"): keep only
            # TARIKH's share of the box, so the first digit is not skipped.
            unit = unit * 6 / letters
            note = "TARIKH box shortened (merged with other text)"
        anchor_x, top_y = x0 + unit, y0
    else:
        x0, y0, x1, _ = _bounds(date["box"])  # date is not None here
        unit = (x1 - x0) / settings.date_label_date_width_units
        anchor_x = x1 + settings.date_label_date_right_gap_units * unit
        top_y = y0 - settings.date_label_tarikh_height_units * unit
        note = "TARIKH not found; placed from DATE"

    low, high = settings.date_label_unit_tolerance
    if not low * expected_unit <= unit <= high * expected_unit:
        note = (note + "; " if note else "") + f"label width {unit:.0f}px implausible, page width used"
        unit = expected_unit
    return unit, anchor_x, top_y, note


def _date_row_box(
    unit: float, anchor_x: float, top_y: float, image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """The six date boxes, from the label measurements (see settings section 3)."""
    height, width = image_shape[:2]
    x0 = anchor_x + settings.date_row_start_units * unit
    x1 = anchor_x + settings.date_row_length_units * unit
    y0 = top_y - settings.date_row_top_pad_units * unit
    y1 = top_y + settings.date_row_bottom_units * unit
    return max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))


def _template_date_zone(image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Fallback for a Public Bank image whose printed label is unreadable."""
    height, width = image_shape[:2]
    return int(width * 0.60), int(height * 0.10), int(width * 0.99), int(height * 0.38)


def parse_date(value: str) -> dt.date | None:
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


def is_valid_date(value: str) -> bool:
    """True when `value` is a real calendar date (see `parse_date`)."""
    return parse_date(value) is not None


def is_flagged(signature_detected: str, valid_date: str) -> str:
    """YES when the cheque needs a person to look at it: no signature found,
    or the date read is not a real calendar date. Both inputs are YES/NO."""
    return "YES" if signature_detected != "YES" or valid_date != "YES" else "NO"


def _calendar_valid(digits: str) -> bool:
    """Validate a concatenated DDMMYY string (no separators, always 6 digits)."""
    return len(digits) == 6 and is_valid_date(digits)


def _write_csv(output_dir: Path, result: dict[str, Any]) -> None:
    if not settings.save_artifacts:
        return
    with (output_dir / "result.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow({key: str(value) if isinstance(value, list) else value for key, value in result.items()})


def _write_digit_confidence_csv(output_dir: Path, top_candidates: list[list[tuple[str, float]]]) -> None:
    """One row per digit slot, with its top-3 (digit, probability) candidates."""
    if not settings.save_artifacts:
        return
    fieldnames = [
        "digit_index",
        "rank1_digit", "rank1_prob",
        "rank2_digit", "rank2_prob",
        "rank3_digit", "rank3_prob",
    ]
    with (output_dir / "digit_confidence.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, candidates in enumerate(top_candidates):
            row: dict[str, Any] = {"digit_index": index}
            for rank in range(3):
                digit, prob = candidates[rank] if rank < len(candidates) else ("", "")
                row[f"rank{rank + 1}_digit"] = digit
                row[f"rank{rank + 1}_prob"] = round(prob, 4) if prob != "" else ""
            writer.writerow(row)


def _trim_right_to_ink(
    image: np.ndarray, box: tuple[int, int, int, int], unit: float, anchor_x: float
) -> tuple[tuple[int, int, int, int], str]:
    """Pull the crop's right edge in to the last handwritten stroke, so the
    hatched pattern at the cheque's right edge stays out.

    Strokes are blobs darker than `digit_ink_threshold` (the same test
    digit_segmentation uses), big enough to be part of a digit and not wider
    than one digit box. The edge never moves left of the sixth box
    (`date_row_min_length_units`), so a faint last digit is not cut off.
    """
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return box, ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = (gray < settings.digit_ink_threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    right_edges = [
        int(left + w)
        for left, _, w, h, area in stats[1:count]
        if area >= 25 and h >= 0.08 * (y1 - y0) and w <= 1.2 * unit
    ]
    min_x1 = anchor_x + settings.date_row_min_length_units * unit
    if right_edges:
        new_x1 = x0 + max(right_edges) + settings.date_row_ink_margin_units * unit
        note = "right edge trimmed to handwriting"
    else:
        new_x1 = anchor_x + settings.date_row_blank_length_units * unit
        note = "no handwriting found"
    new_x1 = int(min(x1, max(min_x1, new_x1)))
    return (x0, y0, new_x1, y1), note


def find_crop_box(
    image: np.ndarray, lines: list[dict[str, Any]]
) -> tuple[tuple[int, int, int, int], str, str, bool]:
    """Where the handwritten date is: (x0, y0, x1, y1), how it was found,
    the label text it was anchored on, and whether TARIKH had to be inferred
    from DATE. Used by `extract` and by diagnostics/check_date_crop.py."""
    image_shape = image.shape
    tarikh, date = _find_label_lines(lines, image_shape)
    if tarikh is None and date is None:
        return _template_date_zone(image_shape), "Public Bank upper-right fallback", "", False
    unit, anchor_x, top_y, note = _label_geometry(tarikh, date, image_shape)
    crop_box = _date_row_box(unit, anchor_x, top_y, image_shape)
    notes = [note] if note else []
    if settings.date_field_trim_to_ink:
        crop_box, trim_note = _trim_right_to_ink(image, crop_box, unit, anchor_x)
        notes.append(trim_note)
    found_texts = [str(line["text"]) for line in (tarikh, date) if line is not None]
    anchor_text = " / ".join(dict.fromkeys(found_texts))
    found_labels = [name for name, line in (("TARIKH", tarikh), ("DATE", date)) if line is not None]
    anchor_mode = "+".join(found_labels) + f" label (unit {unit:.0f}px)"
    if any(notes):
        anchor_mode += "; " + "; ".join(n for n in notes if n)
    return crop_box, anchor_mode, anchor_text, tarikh is None


def extract(
    image: np.ndarray,
    lines: list[dict[str, Any]],
    output_dir: Path,
    width: int = 450,  # retained for API compatibility; crop uses page margin.
) -> dict[str, Any]:
    """PaddleOCR label -> right-side crop -> grid-sliced digit CNN classification."""
    del width
    if settings.save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
    crop_box, anchor_mode, anchor_text, bounds_estimated = find_crop_box(image, lines)

    x0, y0, x1, y1 = crop_box
    crop = image[y0:y1, x0:x1]
    result: dict[str, Any] = {
        "status": "needs_review",
        "date": "",
        "date_source": "digit_cnn",
        "validation": f"crop source: {anchor_mode}",
        "raw_digits": "",
        "anchor_text": anchor_text,
        "confidence": 0.0,
        "per_digit_confidence": "",
        "crop_box": [x0, y0, x1, y1],
        "label_bounds_estimated": bounds_estimated,
        "digit_top3": [],
    }
    if crop.size == 0:
        result["status"] = "empty_crop"
        _write_csv(output_dir, result)
        return result

    crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    if settings.save_artifacts:
        cv2.imwrite(str(output_dir / "crop.png"), crop)

    canonical_slices = digit_segmentation.slice_digits(crop, output_dir, digit_count=settings.digit_count)
    top_candidates = digit_classifier.classify_digits(canonical_slices, top_k=3)
    _write_digit_confidence_csv(output_dir, top_candidates)

    raw_digits = "".join(candidates[0][0] for candidates in top_candidates)
    per_digit_confidence = [round(candidates[0][1], 4) for candidates in top_candidates]
    overall_confidence = min(per_digit_confidence) if per_digit_confidence else 0.0
    # A real nested list (not a str() repr) -- round-trips through
    # json.dumps for result.json and is directly iterable in the UI
    # template, whether from a live in-memory result or one reloaded from
    # result.json on disk.
    result["digit_top3"] = [
        [[digit, round(prob, 4)] for digit, prob in candidates] for candidates in top_candidates
    ]

    formatted_date = (
        f"{raw_digits[0:2]}/{raw_digits[2:4]}/{raw_digits[4:6]}" if len(raw_digits) == 6 else raw_digits
    )
    if _calendar_valid(raw_digits):
        result.update(
            status="date_candidate",
            date=formatted_date,
            validation=f"format and calendar valid; crop source: {anchor_mode}",
        )
    else:
        result.update(
            status="needs_review",
            date=formatted_date,
            validation=f"best-guess reading, not a valid calendar date -- verify manually; crop source: {anchor_mode}",
        )

    result.update(
        raw_digits=raw_digits,
        confidence=round(overall_confidence, 6),
        per_digit_confidence=str(per_digit_confidence),
    )
    _write_csv(output_dir, result)
    return result