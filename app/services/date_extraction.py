"""Find a printed date label with PaddleOCR, then classify each handwritten
digit in the date row with a fine-tuned CNN (grid-sliced, not TrOCR)."""

import csv
from difflib import SequenceMatcher
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.dates import is_flagged, is_valid_date, parse_date  # noqa: F401  (re-exported)
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


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _fit_grid(
    centres: list[float], count: int = 6, start_range: tuple[float, float] | None = None
) -> tuple[float, float, int, float] | None:
    """Fit `count` evenly spaced positions (start, pitch, in units) to the
    detected guide-letter centres. Returns (start, pitch, inliers, mean
    error) for the fit that explains the most letters, or None.
    `start_range` limits where the first position may be (default: the
    configured distance right of TARIKH)."""
    low_pitch, high_pitch = settings.date_grid_pitch_units
    low_start, high_start = start_range or settings.date_grid_start_units
    tolerance = settings.date_grid_tolerance_units
    best = None
    ordered = sorted(centres)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            for gap in range(1, count):
                pitch = (b - a) / gap
                if not low_pitch <= pitch <= high_pitch:
                    continue
                for index in range(count - gap):
                    start = a - index * pitch
                    if not low_start <= start <= high_start:
                        continue
                    hits = []
                    for k in range(count):
                        expected = start + k * pitch
                        near = min(ordered, key=lambda c: abs(c - expected))
                        if abs(near - expected) <= tolerance:
                            hits.append((k, near))
                    if len(hits) < 2:
                        continue
                    ks = np.array([h[0] for h in hits], dtype=float)
                    cs = np.array([h[1] for h in hits], dtype=float)
                    pitch_fit, start_fit = np.polyfit(ks, cs, 1)
                    error = float(np.mean(np.abs(start_fit + ks * pitch_fit - cs)))
                    key = (len(hits), -error)
                    if best is None or key > best[0]:
                        best = (key, (float(start_fit), float(pitch_fit), len(hits), error))
    return best[1] if best else None


def _find_guide_grid(
    image: np.ndarray, unit: float, anchor_x: float, top_y: float
) -> dict[str, Any] | None:
    """Find the printed "D D M M Y Y" letters under the date boxes and fit
    the six box positions to them, for this image only.

    The letters are small, light printed marks, one centred under each box.
    Their spacing in TARIKH-widths differs a little per image (0.77-0.84 on
    the samples: OCR box jitter and scan scale), so fitting them beats any
    fixed multiple. Several darkness thresholds are tried relative to the
    local background, since scans differ in contrast.
    """
    gray = _gray(image)
    height, width = gray.shape
    y0 = max(0, int(top_y + settings.date_grid_letters_from_units * unit))
    y1 = min(height, int(top_y + settings.date_grid_letters_to_units * unit))
    x0, x1 = max(0, int(anchor_x)), min(width, int(anchor_x + 5.4 * unit))
    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        return None
    background = float(np.median(band))
    best = None
    for drop in (25, 45, 65, 90):
        threshold = background - drop
        if threshold <= 0:
            continue
        mask = (band < threshold).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        letters = [
            (float(left + w / 2), int(top))
            for left, top, w, h, area in stats[1:count]
            if 0.05 * unit <= h <= 0.3 * unit and w <= 0.3 * unit and area >= 6
        ]
        if not letters:
            continue
        fit = _fit_grid([cx / unit for cx, _ in letters])
        if fit is None:
            continue
        start, pitch, inliers, error = fit
        if best is None or (inliers, -error) > (best["inliers"], -best["error"]):
            tolerance = settings.date_grid_tolerance_units
            tops = [
                top for cx, top in letters
                if min(abs(cx / unit - (start + k * pitch)) for k in range(6)) <= tolerance
            ]
            best = {
                "start_units": start, "pitch_units": pitch, "inliers": inliers, "error": error,
                "threshold": int(threshold), "letters": [(x0 + cx, y0 + top) for cx, top in letters],
                "start_x": x0 + start * unit, "pitch_px": pitch * unit,
                "letters_top": y0 + float(np.median(tops)), "band": (x0, y0, x1, y1),
            }
    if best is not None and best["inliers"] < settings.date_grid_min_letters:
        best["rejected"] = True
    return best


def _search_guide_grid(image: np.ndarray) -> dict[str, Any] | None:
    """When OCR found neither TARIKH nor DATE: look for the six D D M M Y Y
    letters anywhere in the upper right, row by row, and infer where the
    label must be from them (measured on the samples: the first letter is
    ~0.43 TARIKH widths right of TARIKH, their tops ~0.86 below its top).

    A row only counts when the six positions are not surrounded by other
    small marks, so rows of asterisks (banker's cheques) or printed text are
    not mistaken for the guide letters.
    """
    gray = _gray(image)
    height, width = gray.shape
    unit = width * settings.date_label_unit_page_fraction
    x0, x1 = int(width * 0.45), width
    y0, y1 = int(height * 0.05), int(height * 0.50)
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return None
    background = float(np.median(region))
    tolerance = settings.date_grid_tolerance_units
    best = None
    for drop in (15, 25, 35, 45, 65, 90):
        threshold = background - drop
        if threshold <= 0:
            continue
        mask = (region < threshold).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        marks = sorted(
            (float(top), float(left + w / 2))
            for left, top, w, h, area in stats[1:count]
            if 0.05 * unit <= h <= 0.3 * unit and w <= 0.3 * unit and area >= 6
        )
        rows: list[list[tuple[float, float]]] = []
        for top, cx in marks:
            if rows and abs(top - rows[-1][0][0]) <= 0.25 * unit:  # tilted scans drift
                rows[-1].append((top, cx))
            else:
                rows.append([(top, cx)])
        for row in rows:
            if len(row) < settings.date_grid_min_letters_without_label:
                continue
            centres = [cx / unit for _, cx in row]
            fit = _fit_grid(centres, start_range=(-1e9, 1e9))
            if fit is None:
                continue
            start, pitch, inliers, error = fit
            span = [c for c in centres if start - 0.5 * pitch <= c <= start + 5.5 * pitch]
            extras = len(span) - inliers
            if inliers < settings.date_grid_min_letters_without_label or extras > 2:
                continue
            key = (inliers, -extras, -error)
            if best is None or key > best[0]:
                tops = [top for top, cx in row
                        if min(abs(cx / unit - (start + k * pitch)) for k in range(6)) <= tolerance]
                letters_top = y0 + float(np.median(tops))
                start_x = x0 + start * unit
                best = (key, {
                    "start_units": settings.date_grid_letter_offset_units, "pitch_units": pitch,
                    "inliers": inliers, "error": error, "threshold": int(threshold),
                    "letters": [(x0 + cx, y0 + top) for top, cx in row],
                    "start_x": start_x, "pitch_px": pitch * unit, "letters_top": letters_top,
                    "band": (x0, y0, x1, y1), "unit": unit,
                    "anchor_x": start_x - settings.date_grid_letter_offset_units * unit,
                    "top_y": letters_top - settings.date_grid_letters_below_units * unit,
                })
    return best[1] if best else None


def _grid_box(
    grid: dict[str, Any], unit: float, top_y: float, image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Crop exactly the six boxes the grid found: half a pitch either side of
    the first and last letter, from above the boxes to just above the letters."""
    height, width = image_shape[:2]
    half = grid["pitch_px"] / 2
    inset = settings.date_grid_inset_units * unit
    x0 = grid["start_x"] - half + inset
    x1 = grid["start_x"] + 5 * grid["pitch_px"] + half - inset
    y0 = top_y - settings.date_row_top_pad_units * unit
    y1 = grid["letters_top"] - settings.date_grid_above_letters_units * unit
    return max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))


def _expand_to_handwriting(
    image: np.ndarray, box: tuple[int, int, int, int], unit: float, top_y: float, bottom_limit: float
) -> tuple[tuple[int, int, int, int], str]:
    """Grow the crop up or down when a handwritten digit pokes out of it
    (tall or low writing), never above `top_y - 0.75 units` or below
    `bottom_limit`. Strokes: darker than `digit_ink_threshold`."""
    x0, y0, x1, y1 = box
    gray = _gray(image)
    limit_top = max(0, int(top_y - 0.75 * unit))
    limit_bottom = min(gray.shape[0], int(bottom_limit))
    region = gray[limit_top:limit_bottom, x0:x1]
    if region.size == 0:
        return box, ""
    ink = (region < settings.digit_ink_threshold).astype(np.uint8)
    size = max(1, int(0.05 * unit))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    pad = 0.08 * unit
    new_y0, new_y1 = y0, y1
    for _, top, w, h, _ in stats[1:count]:
        top, bottom = top + limit_top, top + limit_top + h
        if h < 0.15 * unit or w > 1.3 * unit or bottom <= y0 or top >= y1:
            continue  # not a stroke, or not touching the crop's rows
        new_y0 = min(new_y0, int(top - pad))
        new_y1 = max(new_y1, int(bottom + pad))
    new_y0, new_y1 = max(limit_top, new_y0), min(limit_bottom, new_y1)
    note = "grown to fit handwriting" if (new_y0, new_y1) != (y0, y1) else ""
    return (x0, new_y0, x1, new_y1), note


def plan_crop(image: np.ndarray, lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Every step of placing the date crop, for `extract` and the
    diagnostics tools: the labels found, the unit, the guide-letter grid
    (if found), the label-only box, and the final box and how it was made."""
    image_shape = image.shape
    plan: dict[str, Any] = {"tarikh": None, "date": None, "unit": None, "grid": None,
                            "label_box": None, "grid_box": None, "notes": []}
    tarikh, date = _find_label_lines(lines, image_shape)
    plan.update(tarikh=tarikh, date=date)
    if tarikh is None and date is None:
        grid = _search_guide_grid(image)
        plan["grid"] = grid
        if grid is None:
            plan.update(box=_template_date_zone(image_shape), mode="fallback",
                        description="Public Bank upper-right fallback (no TARIKH/DATE or guide letters found)")
            return plan
        unit, top_y = grid["unit"], grid["top_y"]
        plan.update(unit=unit, anchor_x=grid["anchor_x"], top_y=top_y)
        box = plan["grid_box"] = _grid_box(grid, unit, top_y, image_shape)
        plan["notes"].append(f"TARIKH/DATE not found; grid from {grid['inliers']}/6 guide letters alone")
        if settings.date_crop_fit_handwriting:
            box, fit_note = _expand_to_handwriting(image, box, unit, top_y, box[3])
            if fit_note:
                plan["notes"].append(fit_note)
        plan.update(box=box, mode="grid",
                    description=f"no label (unit {unit:.0f}px est.), grid; " + "; ".join(plan["notes"]))
        return plan

    unit, anchor_x, top_y, note = _label_geometry(tarikh, date, image_shape)
    plan.update(unit=unit, anchor_x=anchor_x, top_y=top_y)
    if note:
        plan["notes"].append(note)
    label_box = _date_row_box(unit, anchor_x, top_y, image_shape)
    trim_note = ""
    if settings.date_field_trim_to_ink:
        label_box, trim_note = _trim_right_to_ink(image, label_box, unit, anchor_x)
    plan["label_box"] = label_box

    box, mode = label_box, "label"
    bottom_limit = top_y + settings.date_row_bottom_units * unit
    if settings.date_crop_mode == "grid":
        grid = _find_guide_grid(image, unit, anchor_x, top_y)
        plan["grid"] = grid
        if grid and not grid.get("rejected"):
            box = plan["grid_box"] = _grid_box(grid, unit, top_y, image_shape)
            mode, bottom_limit = "grid", box[3]
            plan["notes"].append(f"grid from {grid['inliers']}/6 guide letters")
        else:
            plan["notes"].append("guide letters not found; label measurements used")
    if mode == "label" and trim_note:
        plan["notes"].append(trim_note)
    if settings.date_crop_fit_handwriting:
        box, fit_note = _expand_to_handwriting(image, box, unit, top_y, bottom_limit)
        if fit_note:
            plan["notes"].append(fit_note)

    found = [name for name, line in (("TARIKH", tarikh), ("DATE", date)) if line is not None]
    plan.update(box=box, mode=mode,
                description="+".join(found) + f" label (unit {unit:.0f}px), {mode}; " + "; ".join(plan["notes"]))
    return plan


def find_crop_box(
    image: np.ndarray, lines: list[dict[str, Any]]
) -> tuple[tuple[int, int, int, int], str, str, bool]:
    """Where the handwritten date is: (x0, y0, x1, y1), how it was found,
    the label text it was anchored on, and whether TARIKH had to be inferred
    from DATE. See `plan_crop` for the steps."""
    plan = plan_crop(image, lines)
    found_texts = [str(line["text"]) for line in (plan["tarikh"], plan["date"]) if line is not None]
    return plan["box"], plan["description"], " / ".join(dict.fromkeys(found_texts)), plan["tarikh"] is None


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
    if not canonical_slices:
        result.update(status="no_date", validation=f"no handwriting in the date boxes; crop source: {anchor_mode}")
        _write_csv(output_dir, result)
        return result
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