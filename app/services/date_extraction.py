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


def _score_candidates(
    lines: list[dict[str, Any]],
    aliases: tuple[str, ...],
    image_shape: tuple[int, ...],
    allow_weak_wn: bool = False,
) -> dict[str, Any] | None:
    """Find the best-scoring line matching any of `aliases`.

    Same fuzzy/substring scoring as before, just scoped to one alias group
    at a time so TARIKH and DATE -- two distinct printed lines on the
    cheque -- can be located independently instead of as one interchangeable
    "the label" match.
    """
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
    they actually are on a Public Bank cheque (TARIKH above, DATE below) --
    rather than one fuzzy match standing in for "the label", which is what
    let the crop math treat a DATE-line match identically to a TARIKH-line
    match even though they sit at different heights.

    A line that fuzzy-matches both groups at once (PaddleOCR occasionally
    returns "TARIKH" + "日DATE" as one merged box) will simply be returned
    as both `tarikh` and `date` -- see `_vertical_bounds_from_labels` for
    why that case needs no special handling.
    """
    tarikh = _score_candidates(lines, settings.date_label_aliases_tarikh, image_shape)
    date = _score_candidates(
        lines,
        settings.date_label_aliases_date + settings.date_label_aliases_ambiguous,
        image_shape,
        allow_weak_wn=True,
    )

    if tarikh is not None and date is not None and tarikh is not date:
        # Sanity check: DATE must sit at or below TARIKH, and their
        # x-ranges must overlap -- they're two lines of the same stacked
        # label. If not, one of the two fuzzy matches almost certainly
        # landed on unrelated printed text elsewhere on the cheque, and
        # trusting it would corrupt the crop worse than estimating its
        # position from the line we're confident about.
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


def _vertical_bounds_from_labels(
    tarikh: dict[str, Any] | None,
    date: dict[str, Any] | None,
    lines: list[dict[str, Any]],
) -> tuple[float, float, bool]:
    """Compute the crop's top/bottom Y directly from the label line(s) found.

    TARIKH is always the upper of the two stacked label lines, DATE the
    lower one. The handwritten date sits beside that stack -- not above or
    below it -- so the correct vertical span is exactly [TARIKH's own top
    edge, DATE's own bottom edge]. Using each line's *own* edge (rather
    than deriving both edges from whichever single line got matched) is
    what keeps the crop correct no matter which of the two PaddleOCR
    happens to detect.

    If only one line is found, the other's edge is estimated: the missing
    line is assumed to be one typical single-line-height away, plus the
    calibrated gap between the two stacked lines -- i.e. exactly "how far
    it is" from the line we do have to the one we don't.

    Returns (top_y, bottom_y, was_estimated).
    """
    typical_height = _typical_line_height(lines) or 20.0
    line_gap = typical_height * settings.date_label_line_gap_ratio

    if tarikh is not None and date is not None:
        _, top_y, _, _ = _bounds(tarikh["box"])
        _, _, _, bottom_y = _bounds(date["box"])
        return top_y, bottom_y, False

    if tarikh is not None:
        _, top_y, _, tarikh_bottom = _bounds(tarikh["box"])
        estimated_bottom = tarikh_bottom + line_gap + typical_height
        return top_y, estimated_bottom, True

    _, date_top, _, bottom_y = _bounds(date["box"])  # date is not None here
    estimated_top = date_top - line_gap - typical_height
    return estimated_top, bottom_y, True


def _typical_line_height(lines: list[dict[str, Any]]) -> float | None:
    """Robust reference height for one printed line of text on this cheque.

    A cheque carries a dozen-plus short printed lines (BAYAR/PAY, RINGGIT
    MALAYSIA, ATAU PEMBAYARAN, the account-number line, ...) all set in the
    same or similar font size. Their *median* height is a stable, per-image,
    format-invariant yardstick: unlike any single line's own box, it isn't
    thrown off by one OCR line-detection quirk, because a merge/split
    artifact on one line is an outlier against a dozen others.

    Used as the sizing reference for the crop's width and safety padding,
    and to estimate whichever of TARIKH/DATE isn't found directly (see
    `_vertical_bounds_from_labels`) -- never to relocate the anchor itself,
    since the anchor line(s) still drive *where* the crop starts.
    """
    heights = []
    for line in lines:
        text = str(line["text"])
        if len(_normalise(text)) < 2:
            continue
        x0, y0, x1, y1 = _bounds(line["box"])
        h = y1 - y0
        if h > 0:
            heights.append(h)
    if len(heights) < 3:
        return None
    return float(np.median(heights))


def _right_of_labels(
    tarikh: dict[str, Any] | None,
    date: dict[str, Any] | None,
    lines: list[dict[str, Any]],
    image_shape: tuple[int, ...],
) -> tuple[tuple[int, int, int, int], bool]:
    """Crop tightly around the handwritten date cell, right of TARIKH/DATE.

    Vertical bounds come directly from `_vertical_bounds_from_labels` --
    TARIKH's own top edge to DATE's own bottom edge, estimating whichever
    line wasn't found. That replaces the old approach of scaling fixed
    up/down padding off whichever single line got matched, which is what
    broke whenever the matched line was DATE (the lower line) instead of
    TARIKH (the upper one): padding measured from DATE's bottom edge
    systematically overshoots below the digits and undershoots above them.

    Horizontal bound starts right of whichever label line(s) were found
    (the rightmost edge, so a wider TARIKH or DATE text is never clipped
    into), sized in multiples of the document's typical single-line text
    height -- a stable, format-invariant reference (see
    `_typical_line_height`) rather than any one box's own height.

    A final absolute clamp (`date_field_max_*_fraction`) caps the crop
    regardless of what the labels measured, so a bad OCR read can't blow
    the crop out to an unusable size.
    """
    height, width = image_shape[:2]
    top_y, bottom_y, was_estimated = _vertical_bounds_from_labels(tarikh, date, lines)
    typical_height = _typical_line_height(lines) or max(1.0, bottom_y - top_y)

    right_edges = [_bounds(line["box"])[2] for line in (tarikh, date) if line is not None]
    label_x1 = max(right_edges)

    safety_pad = typical_height * settings.date_field_vertical_safety_pad_ratio
    crop_y0 = top_y - safety_pad
    crop_y1 = bottom_y + safety_pad
    crop_x0 = label_x1
    crop_x1 = crop_x0 + typical_height * settings.date_field_width_typical_heights

    # Absolute safety net: independent of how the bounds above were
    # derived, the date cell can never legitimately need more than this
    # fraction of the page. Clamps size, not the anchored top-left origin.
    max_width = width * settings.date_field_max_width_fraction
    max_height = height * settings.date_field_max_height_fraction
    crop_x1 = min(crop_x1, crop_x0 + max_width)
    if crop_y1 - crop_y0 > max_height:
        crop_y1 = crop_y0 + max_height

    crop_box = (
        max(0, int(crop_x0)),
        max(0, int(crop_y0)),
        min(width, int(crop_x1)),
        min(height, int(crop_y1)),
    )
    return crop_box, was_estimated


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
    tarikh, date = _find_label_lines(lines, image.shape)
    bounds_estimated = False
    if tarikh is not None or date is not None:
        crop_box, bounds_estimated = _right_of_labels(tarikh, date, lines, image.shape)
        found_texts = [str(line["text"]) for line in (tarikh, date) if line is not None]
        anchor_text = " / ".join(dict.fromkeys(found_texts))  # dedupe if same merged line
        found_labels = [name for name, line in (("TARIKH", tarikh), ("DATE", date)) if line is not None]
        anchor_mode = "OCR/fuzzy " + "+".join(found_labels) + " anchor"
        if bounds_estimated:
            anchor_mode += " (one line estimated from the other)"
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
        "label_bounds_estimated": bounds_estimated,
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