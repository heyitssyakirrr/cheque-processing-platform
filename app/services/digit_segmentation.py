"""Slice a printed date-row crop into individual digit images.

Takes the crop already produced by the PaddleOCR-anchor + right-of-anchor
logic in date_extraction.py, trims the printed guide-letter band ("D D M M
Y Y"), then segments the remainder into `digit_count` digits by their real
ink gaps (not equal-width division -- a narrow '1' and a wide '8' don't
occupy the same ink-width, so equal slices misalign whenever digit widths
vary this much), splitting or merging specific groups only where digits
touch or noise creates a spurious extra blob. Normalizes each digit into a
canonical square grayscale image ready for a classifier's own final
resize/normalize step.
"""

from pathlib import Path

import cv2
import numpy as np

from app.settings import settings


def _trim_guide_band(gray: np.ndarray, guide_band_ratio: float) -> np.ndarray:
    """Cut off the bottom band containing the faint 'D D M M Y Y' guide letters."""
    height = gray.shape[0]
    cutoff = int(height * (1 - guide_band_ratio))
    return gray[:cutoff, :]


def _column_ink_profile(gray: np.ndarray) -> np.ndarray:
    """Column-wise count of dark (ink) pixels, used to find gaps between digits.

    Uses a fixed threshold rather than Otsu -- see `digit_ink_threshold` in
    settings for why: Otsu can be pulled toward including non-ink gray
    shading as "foreground" depending on what else is in the crop.
    """
    _, binary = cv2.threshold(gray, settings.digit_ink_threshold, 255, cv2.THRESH_BINARY_INV)
    return binary.sum(axis=0) // 255


def _trim_to_ink_extent(gray: np.ndarray, buffer_ratio: float = 0.12) -> np.ndarray:
    """Trim leading/trailing blank columns down to where the ink actually is.

    This is what lets the outer crop (from date_extraction.py) stay
    generously wide without needing to be pixel-precise: any blank margin
    added as a safety buffer against clipping the last digit would otherwise
    skew the equal-width lane division below, since that division assumes
    the crop's full width is occupied by `digit_count` digits. Trimming to
    the real ink extent first removes that assumption entirely.
    """
    profile = _column_ink_profile(gray)
    ink_columns = np.nonzero(profile > 0)[0]
    if len(ink_columns) == 0:
        return gray
    left, right = int(ink_columns[0]), int(ink_columns[-1])
    buffer_px = int((right - left) * buffer_ratio)
    left = max(0, left - buffer_px)
    right = min(gray.shape[1], right + buffer_px + 1)
    return gray[:, left:right]


def _find_ink_groups(profile: np.ndarray, min_gap_width: int = 3) -> list[tuple[int, int]]:
    """Find contiguous ink groups (candidate digits), separated by real gaps.

    Unlike equal-width division, this uses the actual physical gaps between
    digits when they exist -- the normal case, since these are separate
    printed boxes -- instead of assuming every digit occupies the same
    ink-width. A '1' is inherently far narrower in ink than an '8' or '5',
    so equal division of total ink width misaligns whenever digit widths
    differ this much, which is exactly what happened on this cheque.
    """
    is_ink = profile > 0
    groups: list[tuple[int, int]] = []
    start = None
    gap_run = 0
    for i, ink in enumerate(is_ink):
        if ink:
            if start is None:
                start = i
            gap_run = 0
        elif start is not None:
            gap_run += 1
            if gap_run >= min_gap_width:
                groups.append((start, i - gap_run))
                start = None
    if start is not None:
        groups.append((start, len(is_ink) - 1))
    return groups


def _split_widest_group(groups: list[tuple[int, int]], target_count: int) -> list[tuple[int, int]]:
    """Split the widest group(s) until there are `target_count` groups.

    Handles touching/connected digits (e.g. the merged '26'-style stroke
    seen earlier) that produce fewer ink groups than expected -- only the
    specific group(s) actually stuck together get subdivided, not the
    whole row.
    """
    groups = list(groups)
    while len(groups) < target_count:
        widths = [end - start for start, end in groups]
        idx = int(np.argmax(widths))
        start, end = groups.pop(idx)
        mid = (start + end) // 2
        groups.insert(idx, (start, mid))
        groups.insert(idx + 1, (mid + 1, end))
    return sorted(groups)


def _merge_closest_groups(groups: list[tuple[int, int]], target_count: int) -> list[tuple[int, int]]:
    """Merge the closest pair of groups until there are `target_count` groups.

    Handles noise producing spurious extra ink groups (stray marks, box
    line bleed) that would otherwise be mistaken for a real digit.
    """
    groups = sorted(groups)
    while len(groups) > target_count:
        gaps = [groups[i + 1][0] - groups[i][1] for i in range(len(groups) - 1)]
        idx = int(np.argmin(gaps))
        merged = (groups[idx][0], groups[idx + 1][1])
        groups = groups[:idx] + [merged] + groups[idx + 2 :]
    return groups


def _digit_boundaries(profile: np.ndarray, digit_count: int) -> list[int]:
    """Return `digit_count + 1` dividers, one bounding box per real ink group.

    Prefers real detected gaps over any equal-width assumption; only
    splits or merges specific groups when the raw gap count doesn't
    already match `digit_count`, so well-separated digits of very
    different widths (a narrow '1' next to a wide '8') are still bounded
    correctly instead of forced into equal slices.
    """
    groups = _find_ink_groups(profile)
    if not groups:
        # No ink detected at all -- fall back to naive equal division so
        # slicing still returns `digit_count` (blank) images rather than
        # erroring out.
        width = len(profile)
        step = width / digit_count
        return [int(i * step) for i in range(digit_count + 1)]

    if len(groups) < digit_count:
        groups = _split_widest_group(groups, digit_count)
    elif len(groups) > digit_count:
        groups = _merge_closest_groups(groups, digit_count)

    dividers = [max(0, groups[0][0] - 2)]
    for i in range(len(groups) - 1):
        dividers.append((groups[i][1] + groups[i + 1][0]) // 2)
    dividers.append(min(len(profile), groups[-1][1] + 3))
    return dividers


def _normalize_digit(gray_slice: np.ndarray, canvas_size: int = 128) -> np.ndarray:
    """Binarize, invert to white-ink-on-black, center by ink centroid, pad to square.

    Bridges the domain gap toward MNIST-style training data without any
    fine-tuning: real ink is rarely centered or square the way a lane crop
    is, so centering on the ink itself (not the crop's bounding box)
    matters more than the exact threshold method.
    """
    _, binary = cv2.threshold(gray_slice, settings.digit_ink_threshold, 255, cv2.THRESH_BINARY_INV)

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
    trimmed = _trim_to_ink_extent(trimmed)

    profile = _column_ink_profile(trimmed)
    dividers = _digit_boundaries(profile, digit_count)

    digits_dir = output_dir / "digits"
    digits_dir.mkdir(parents=True, exist_ok=True)

    canonical_slices = []
    for i in range(digit_count):
        lane = trimmed[:, dividers[i] : dividers[i + 1]]
        canonical = _normalize_digit(lane)
        cv2.imwrite(str(digits_dir / f"digit_{i}.png"), canonical)
        canonical_slices.append(canonical)

    return canonical_slices