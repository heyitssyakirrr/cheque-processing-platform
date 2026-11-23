"""Date crop placement (guide-letter grid) and blank-date detection."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import run  # noqa: E402

from app.services import date_extraction, digit_segmentation  # noqa: E402


def test_fit_grid_with_missing_and_extra_letters():
    # Six letters 0.8 apart from 0.42, one missing, plus two specks.
    centres = [0.42, 1.22, 2.02, 3.62, 4.42, 0.1, 2.5]
    start, pitch, inliers, error = date_extraction._fit_grid(centres)
    assert inliers == 5 and abs(start - 0.42) < 0.01 and abs(pitch - 0.8) < 0.01 and error < 0.01


def test_fit_grid_rejects_unrelated_marks():
    fit = date_extraction._fit_grid([0.1, 3.0])
    assert fit is None or fit[2] < 4


def _page_with_letters(pitch: float, unit: int = 100) -> tuple[np.ndarray, float, float]:
    page = np.full((600, 1400, 3), 235, np.uint8)
    anchor_x, top_y = 300.0, 200.0
    for k in range(6):
        cx = int(anchor_x + (0.42 + k * pitch) * unit)
        cv2.putText(page, "D", (cx - 5, int(top_y + 0.9 * unit)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    return page, anchor_x, top_y


def test_guide_grid_found_on_drawn_letters():
    for pitch in (0.77, 0.84):
        page, anchor_x, top_y = _page_with_letters(pitch)
        grid = date_extraction._find_guide_grid(page, 100.0, anchor_x, top_y)
        assert grid and not grid.get("rejected") and grid["inliers"] == 6, grid
        assert abs(grid["pitch_units"] - pitch) < 0.03, grid["pitch_units"]


def test_blank_grey_boxes_are_not_ink():
    crop = np.full((120, 600), 185, np.uint8)
    for x in range(0, 600, 100):
        crop[:, x:x + 8] = 250  # white gaps between grey boxes
    assert digit_segmentation.explain(crop).get("blank")
    assert digit_segmentation.slice_digits(crop, Path("."), 6) == []


def test_printed_mark_in_one_box_is_blank():
    crop = np.full((120, 600), 245, np.uint8)
    crop[40:90, 20:26] = 60  # an "L"-shaped printed mark
    crop[84:90, 20:50] = 60
    assert digit_segmentation.explain(crop).get("blank")


def test_six_written_digits_are_read_as_six():
    crop = np.full((120, 600), 245, np.uint8)
    for k in range(6):
        cv2.putText(crop, str(k + 1), (20 + k * 95, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 40, 6)
    seen = digit_segmentation.explain(crop)
    assert not seen.get("blank") and len(seen["lanes"]) == 6


def test_blank_boxes_on_a_darker_scan_are_not_ink():
    # Darker scan: blank boxes ~150 with texture down to ~120, all below the
    # 170 ceiling; only ink clearly darker than the background counts.
    rng = np.random.default_rng(0)
    crop = (150 + rng.integers(-30, 10, size=(120, 600))).astype(np.uint8)
    assert digit_segmentation.explain(crop).get("blank")


def _page_for_search(with_asterisks: bool) -> np.ndarray:
    page = np.full((1000, 2000, 3), 240, np.uint8)
    unit = 2000 * 0.0535
    if with_asterisks:  # a row of asterisks, as on a banker's cheque
        for k in range(40):
            cv2.putText(page, "*", (1000 + int(k * 0.16 * unit), 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2)
    for k in range(6):  # the guide letters
        cv2.putText(page, "D", (1500 + int(k * 0.8 * unit), 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    return page


def test_guide_letters_found_without_any_label():
    for asterisks in (False, True):
        grid = date_extraction._search_guide_grid(_page_for_search(asterisks))
        assert grid and grid["inliers"] == 6, (asterisks, grid)
        assert 360 <= grid["letters_top"] <= 380, grid["letters_top"]  # the letters, not the asterisks


def test_no_label_crop_uses_the_letters():
    plan = date_extraction.plan_crop(_page_for_search(True), [])
    assert plan["mode"] == "grid" and plan["box"][3] <= 380, plan


if __name__ == "__main__":
    run([value for name, value in list(globals().items()) if name.startswith("test_")])
