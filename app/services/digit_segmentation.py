"""Slice a printed date-row crop into individual digit images.

Takes the crop already produced by the PaddleOCR-anchor + right-of-anchor
logic in date_extraction.py. That crop's box is only an estimate — it is
deliberately generous so it never truncates the last handwritten digit,
which means it can include blank margin, the printed "D D M M Y Y" guide
band, or even stray text above the date (account labels, stamp boxes).

This module does the precision work in two stages, both driven by ink
content rather than fixed pixel ratios, so results are stable no matter
the source image's resolution, crop aspect ratio, or margin:

1. Isolate the handwritten digit row itself (discard guide letters/noise
   above and below), then tightly crop to the actual ink bounding box
   left-to-right (discard blank margin).
2. Cluster that ink into exactly `digit_count` characters using connected
   components: broken strokes of one digit get merged, digits that touch
   or bleed into each other get split at their thinnest ink point. This
   replaces blind equal-width division, which is what let touching digits
   (e.g. "26") land in one lane while a trailing blank margin produced an
   empty lane.
"""

from pathlib import Path

import cv2
import numpy as np


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold once; ink -> 255, background -> 0."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _row_bands(binary: np.ndarray, min_gap_ratio: float = 0.02) -> list[tuple[int, int]]:
    """Return (top, bottom) y-ranges of horizontal ink bands, separated by
    near-empty rows. A cheque date crop typically contains several such
    bands: stray text above, the handwritten digits, and the small printed
    guide letters below — this just finds the candidates."""
    height, width = binary.shape
    row_ink = binary.sum(axis=1) // 255
    threshold = max(1, int(width * 0.01))
    ink_rows = row_ink > threshold

    bands: list[list[int]] = []
    start = None
    for y, has_ink in enumerate(ink_rows):
        if has_ink and start is None:
            start = y
        elif not has_ink and start is not None:
            bands.append([start, y])
            start = None
    if start is not None:
        bands.append([start, height])

    # Merge bands separated by only a tiny gap (a pen lift mid-stroke
    # shouldn't count as a new band).
    min_gap = max(1, int(height * min_gap_ratio))
    merged: list[list[int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= min_gap:
            merged[-1][1] = band[1]
        else:
            merged.append(band)
    return [(t, b) for t, b in merged]


def _select_digit_band(binary: np.ndarray, bands: list[tuple[int, int]]) -> tuple[int, int]:
    """Pick the row band that is the handwritten digits.

    Scored by total ink area, not just height: six solid handwritten
    characters carry far more ink than a thin row of small printed guide
    letters or a stray line of unrelated printed text, even if that text
    happens to be similarly tall.
    """
    if not bands:
        return 0, binary.shape[0]

    def ink_area(band: tuple[int, int]) -> int:
        top, bottom = band
        return int(binary[top:bottom, :].sum() // 255)

    return max(bands, key=ink_area)


def _best_valley(binary_row: np.ndarray, x0: int, x1: int) -> int:
    """Find the column with the least ink strictly inside (x0, x1),
    avoiding the extreme edges so a split never produces a near-empty
    sliver off one side of a touching-digit pair."""
    width = x1 - x0
    margin = max(1, int(width * 0.2))
    if width <= 2 * margin:
        return x0 + width // 2
    profile = binary_row[:, x0:x1].sum(axis=0)
    search = profile[margin : width - margin]
    return x0 + margin + int(np.argmin(search))


def _equal_split(width: int, digit_count: int) -> list[tuple[int, int]]:
    """Last-resort fallback when no ink survives filtering at all."""
    lane = width / digit_count
    return [(int(i * lane), int((i + 1) * lane)) for i in range(digit_count)]


def _cluster_into_digits(
    binary_row: np.ndarray, digit_count: int, merge_gap_ratio: float = 0.12
) -> list[tuple[int, int]]:
    """Group ink into exactly `digit_count` left-to-right x-ranges.

    1. Connected components -> raw ink blobs.
    2. Merge blobs separated by only a small gap (a detached dot, a
       disconnected loop — pieces of the same character).
    3. If still more blobs than `digit_count`, repeatedly merge the
       closest remaining pair (handles multi-stroke digits).
    4. If fewer blobs than `digit_count`, repeatedly split the widest
       blob at its deepest ink valley (handles digits that touch or bleed
       into their neighbor, e.g. a connected "26").
    """
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_row, connectivity=8)
    band_height = binary_row.shape[0]
    min_area = max(4, int(binary_row.size * 0.0008))  # scale-relative noise floor
    min_height = max(2, int(band_height * 0.3))  # a real digit stroke spans a large
    # fraction of the row's height; a stray mark or print speck usually doesn't,
    # even if it happens to cover enough area to pass min_area alone.
    boxes = [
        (stats[i][0], stats[i][0] + stats[i][2])
        for i in range(1, num_labels)  # skip background label 0
        if stats[i][4] >= min_area and stats[i][3] >= min_height
    ]
    if not boxes:
        return _equal_split(binary_row.shape[1], digit_count)
    boxes.sort()

    avg_width = binary_row.shape[1] / digit_count
    merge_gap = max(1, int(avg_width * merge_gap_ratio))

    merged = [list(boxes[0])]
    for x0, x1 in boxes[1:]:
        if x0 - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    while len(merged) > digit_count:
        gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
        idx = int(np.argmin(gaps))
        merged[idx][1] = merged[idx + 1][1]
        del merged[idx + 1]

    while len(merged) < digit_count:
        widths = [x1 - x0 for x0, x1 in merged]
        idx = int(np.argmax(widths))
        x0, x1 = merged[idx]
        split_at = _best_valley(binary_row, x0, x1)
        merged[idx : idx + 1] = [[x0, split_at], [split_at, x1]]

    return [(x0, x1) for x0, x1 in merged]


def _normalize_digit(binary_slice: np.ndarray, canvas_size: int = 128) -> np.ndarray:
    """Center an already-binary (white-ink-on-black) slice by ink centroid,
    scale it to fit, and pad it to a square canvas.

    Bridges the domain gap toward MNIST-style training data without any
    fine-tuning: real ink is rarely centered or square the way a lane crop
    is, so centering on the ink itself (not the crop's bounding box)
    matters more than the exact threshold method. Takes a binary mask
    directly (not grayscale) since the caller already binarized once,
    upstream, to do band/cluster detection — re-thresholding an already
    binary 0/255 image here would invert ink and background.
    """
    ys, xs = np.nonzero(binary_slice)
    if len(xs) == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    ink = binary_slice[y0 : y1 + 1, x0 : x1 + 1]

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
) -> list[np.ndarray]:
    """Split a date-row crop into `digit_count` normalized square digit images.

    Saves each slice to `output_dir/digits/digit_<i>.png` and returns the
    in-memory canonical images (white ink on black, centered, square).
    """
    gray = cv2.cvtColor(date_crop, cv2.COLOR_BGR2GRAY) if date_crop.ndim == 3 else date_crop
    binary = _binarize(gray)

    digits_dir = output_dir / "digits"
    digits_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: isolate the handwritten digit row, then trim to its ink
    # bounding box so crop margin can never distort lane math downstream.
    bands = _row_bands(binary)
    top, bottom = _select_digit_band(binary, bands)
    row_pad = max(2, int((bottom - top) * 0.08))
    top, bottom = max(0, top - row_pad), min(binary.shape[0], bottom + row_pad)
    row = binary[top:bottom, :]

    xs = np.nonzero(row.sum(axis=0))[0]
    if len(xs) == 0:
        canonical_slices = [np.zeros((128, 128), dtype=np.uint8) for _ in range(digit_count)]
        for i, canonical in enumerate(canonical_slices):
            cv2.imwrite(str(digits_dir / f"digit_{i}.png"), canonical)
        return canonical_slices

    left, right = int(xs.min()), int(xs.max()) + 1
    col_pad = max(2, int((right - left) * 0.01))
    left, right = max(0, left - col_pad), min(row.shape[1], right + col_pad)
    row = row[:, left:right]

    # Stage 2: cluster ink into exactly `digit_count` characters.
    clusters = _cluster_into_digits(row, digit_count)

    canonical_slices = []
    for i, (x0, x1) in enumerate(clusters):
        lane = row[:, x0:x1]
        canonical = _normalize_digit(lane)
        cv2.imwrite(str(digits_dir / f"digit_{i}.png"), canonical)
        canonical_slices.append(canonical)

    return canonical_slices