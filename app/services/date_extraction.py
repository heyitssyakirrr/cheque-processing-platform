"""Find a printed date label with PaddleOCR, then classify each handwritten
digit in the date row with a fine-tuned CNN (grid-sliced, not TrOCR)."""

import calendar
import csv
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


def _find_anchor(lines: list[dict[str, Any]], image_shape: tuple[int, ...]) -> dict[str, Any] | None:
    """Find TARIKH/DATE despite predictable OCR spelling errors.

    No handwritten digits are considered here. PaddleOCR's only role in
    this service is returning a trustworthy printed-label bounding box.
    """
    aliases = [_normalise(alias) for alias in settings.date_label_aliases]
    height, width = image_shape[:2]
    best: tuple[float, dict[str, Any]] | None = None
    for line in lines:
        text = _normalise(str(line["text"]))
        if len(text) < 2:
            continue
        direct = [alias for alias in aliases if alias in text or text in alias]
        similarity = max(SequenceMatcher(None, text, alias).ratio() for alias in aliases if len(alias) >= 3)
        x0, y0, x1, y1 = _bounds(line["box"])
        # WN is a known but weak TARIKH/DATE corruption; only trust it in the
        # normal upper-right label area to avoid anchoring on unrelated text.
        weak_wn = text == "WN" and (x0 + x1) / 2 >= width * 0.55 and (y0 + y1) / 2 <= height * 0.50
        if not direct and similarity < 0.66 and not weak_wn:
            continue
        score = float(line["confidence"]) + similarity + (2.0 if text in direct else 0.0)
        if best is None or score > best[0]:
            best = (score, line)
    return best[1] if best else None


def _right_of_anchor(anchor: dict[str, Any], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Crop tightly around the handwritten date cell, right of the printed label.

    Sized in multiples of the label's own OCR'd text height, not fixed
    fractions of the full image. A fixed fraction of image width/height
    doesn't scale with the date field: a higher-res scan has proportionally
    more blank margin and unrelated print around it, so `width * 0.99`
    always reached past the digits into noisy margin, and a symmetric
    vertical pad around the label pulled in unrelated text sitting well
    above it (account-type line, stamp box). `label_height` scales with
    the print itself, so multiples of it stay correct regardless of scan
    resolution or page margin.

    The multipliers live in settings and were calibrated against measured
    Public Bank cheque geometry: the handwritten digit row starts ~0.2
    label-heights below the label's bottom edge and is about as tall as
    the label text; the printed "D D M M Y Y" guide row follows
    immediately after. The defaults add a safety margin on top of that
    single measured sample for handwriting-size variance — if a broader
    cheque sample shows the crop is still too tight or too loose, tune the
    `date_field_*` settings rather than editing this function.
    """
    height, width = image_shape[:2]
    _label_x0, label_y0, label_x1, label_y1 = _bounds(anchor["box"])
    label_height = max(1.0, label_y1 - label_y0)

    crop_x0 = label_x1
    crop_x1 = crop_x0 + label_height * settings.date_field_width_label_heights
    crop_y0 = label_y1 - label_height * settings.date_field_up_pad_label_heights
    crop_y1 = label_y1 + label_height * settings.date_field_down_pad_label_heights

    return (
        max(0, int(crop_x0)),
        max(0, int(crop_y0)),
        min(width, int(crop_x1)),
        min(height, int(crop_y1)),
    )


def _template_date_zone(image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Fallback for a Public Bank image whose printed label is unreadable."""
    height, width = image_shape[:2]
    return int(width * 0.60), int(height * 0.10), int(width * 0.99), int(height * 0.38)


def _calendar_valid(digits: str) -> bool:
    """Validate a concatenated DDMMYY string (no separators, always 6 digits)."""
    if len(digits) != 6 or not digits.isdigit():
        return False
    day, month, year = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
    if not 1 <= month <= 12:
        return False
    return 1 <= day <= calendar.monthrange(2000 + year, month)[1]


def _write_csv(output_dir: Path, result: dict[str, Any]) -> None:
    with (output_dir / "result.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow({key: str(value) if isinstance(value, list) else value for key, value in result.items()})


def extract(
    image: np.ndarray,
    lines: list[dict[str, Any]],
    output_dir: Path,
    width: int = 450,  # retained for API compatibility; crop uses page margin.
) -> dict[str, Any]:
    """PaddleOCR label -> right-side crop -> grid-sliced digit CNN classification."""
    del width
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor = _find_anchor(lines, image.shape)
    if anchor is not None:
        crop_box = _right_of_anchor(anchor, image.shape)
        anchor_text = str(anchor["text"])
        anchor_mode = "OCR/fuzzy label anchor"
    else:
        crop_box = _template_date_zone(image.shape)
        anchor_text = ""
        anchor_mode = "Public Bank upper-right fallback"

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
    }
    if crop.size == 0:
        result["status"] = "empty_crop"
        _write_csv(output_dir, result)
        return result

    crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(output_dir / "crop.png"), crop)

    canonical_slices = digit_segmentation.slice_digits(crop, output_dir, digit_count=settings.digit_count)
    predictions = digit_classifier.classify_digits(canonical_slices)

    raw_digits = "".join(digit for digit, _confidence in predictions)
    per_digit_confidence = [round(confidence, 4) for _digit, confidence in predictions]
    # Weakest-link confidence: a single low-confidence digit should flag the
    # whole date for review, not get averaged away by five confident ones.
    overall_confidence = min(per_digit_confidence) if per_digit_confidence else 0.0

    if _calendar_valid(raw_digits):
        result.update(
            status="date_candidate",
            date=f"{raw_digits[0:2]}/{raw_digits[2:4]}/{raw_digits[4:6]}",
            validation=f"format and calendar valid; crop source: {anchor_mode}",
        )
    else:
        result["validation"] = f"digit classification did not form a valid date; crop source: {anchor_mode}"

    result.update(
        raw_digits=raw_digits,
        confidence=round(overall_confidence, 6),
        per_digit_confidence=str(per_digit_confidence),
    )
    _write_csv(output_dir, result)
    return result