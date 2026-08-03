"""Slice a printed date-row crop into individual digit images.

Takes the crop already produced by the PaddleOCR-anchor + right-of-anchor
logic in date_extraction.py, trims the printed guide-letter band ("D D M M
Y Y"), splits the remainder into `digit_count` lanes using ink-gap
snapping (not blind equal division, since adjacent handwritten digits can
touch), and normalizes each lane into a canonical square grayscale image
ready for a classifier's own final resize/normalize step.
"""

from pathlib import Path

import cv2
import numpy as np


def _trim_guide_band(gray: np.ndarray, guide_band_ratio: float) -> np.ndarray:
    """Cut off the bottom band containing the faint 'D D M M Y Y' guide letters."""
    height = gray.shape[0]
    cutoff = int(height * (1 - guide_band_ratio))
    return gray[:cutoff, :]


def _column_ink_profile(gray: np.ndarray) -> np.ndarray:
    """Column-wise count of dark (ink) pixels, used to find gaps between digits."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary.sum(axis=0) // 255


def _snap_dividers(profile: np.ndarray, count: int, search_ratio: float = 0.15) -> list[int]:
    """Start from equal-width dividers, snap each to the nearest local ink minimum.

    Handles handwritten digits that bleed slightly across an equal-width
    boundary (confirmed in real samples: the last two digits of a date can
    share a connected stroke) without needing an ML-based segmenter.
    """
    width = len(profile)
    lane_width = width / count
    dividers = [0]
    for i in range(1, count):
        guess = int(i * lane_width)
        window = max(1, int(lane_width * search_ratio))
        lo, hi = max(0, guess - window), min(width, guess + window)
        local = profile[lo:hi]
        best_offset = int(np.argmin(local)) if len(local) else 0
        dividers.append(lo + best_offset)
    dividers.append(width)
    return dividers


def _normalize_digit(gray_slice: np.ndarray, canvas_size: int = 128) -> np.ndarray:
    """Binarize, invert to white-ink-on-black, center by ink centroid, pad to square.

    Bridges the domain gap toward MNIST-style training data without any
    fine-tuning: real ink is rarely centered or square the way a lane crop
    is, so centering on the ink itself (not the crop's bounding box)
    matters more than the exact threshold method.
    """
    _, binary = cv2.threshold(gray_slice, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    ink = binary[y0 : y1 + 1, x0 : x1 + 1]

    height, width = ink.shape
    scale = (canvas_size * 0.8) / max(height, width)
    new_h, new_w = max(1, int(height * scale)), max(1, int(width * scale))
    resized = cv2.resize(ink, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    top = (canvas_size - new_h) // 2
    left = (canvas_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def slice_digits(
    date_crop: np.ndarray,
    output_dir: Path,
    digit_count: int = 6,
    guide_band_ratio: float = 0.20,
) -> list[np.ndarray]:
    """Split a date-row crop into `digit_count` normalized square digit images.

    Saves each slice to `output_dir/digits/digit_<i>.png` and returns the
    in-memory canonical images (white ink on black, centered, square).
    """
    gray = cv2.cvtColor(date_crop, cv2.COLOR_BGR2GRAY) if date_crop.ndim == 3 else date_crop
    trimmed = _trim_guide_band(gray, guide_band_ratio)

    profile = _column_ink_profile(trimmed)
    dividers = _snap_dividers(profile, digit_count)

    digits_dir = output_dir / "digits"
    digits_dir.mkdir(parents=True, exist_ok=True)

    canonical_slices = []
    for i in range(digit_count):
        lane = trimmed[:, dividers[i] : dividers[i + 1]]
        canonical = _normalize_digit(lane)
        cv2.imwrite(str(digits_dir / f"digit_{i}.png"), canonical)
        canonical_slices.append(canonical)

    return canonical_slices