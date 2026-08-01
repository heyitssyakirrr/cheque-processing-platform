"""Safe, image-wide preprocessing for cheque scans and photographs."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessOptions:
    """User-adjustable preprocessing settings."""

    max_long_edge: int = 2200
    min_short_edge: int = 1000
    max_upscale_factor: float = 4.0
    contrast: float = 1.15
    denoise: bool = True
    deskew: bool = True


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a conservatively deskewed image and its applied rotation."""
    mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    points = np.column_stack(np.where(mask > 0))

    if len(points) < 100:
        return gray, 0.0

    angle = cv2.minAreaRect(points[:, ::-1].astype(np.float32))[2]
    angle = angle - 90 if angle > 45 else angle

    # A cheque should only need a small correction. Reject unreliable angles.
    if abs(angle) > 8:
        return gray, 0.0

    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle)


def preprocess(
    image_bgr: np.ndarray,
    options: PreprocessOptions,
) -> tuple[np.ndarray, dict[str, float | list[int]]]:
    """Normalize a cheque image while keeping its layout coordinates intact."""
    height, width = image_bgr.shape[:2]
    short_edge, long_edge = min(height, width), max(height, width)
    # Normalise small scans before OCR/YOLO.  The prior pipeline only ever
    # downscaled, which left low-resolution cheque photos at scale 1.0.
    scale = 1.0
    if short_edge < options.min_short_edge:
        scale = min(options.min_short_edge / short_edge, options.max_upscale_factor)
    if long_edge * scale > options.max_long_edge:
        scale = options.max_long_edge / long_edge

    if abs(scale - 1.0) > 1e-6:
        image_bgr = cv2.resize(
            image_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if options.denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)

    deskew_angle = 0.0
    if options.deskew:
        gray, deskew_angle = _deskew(gray)

    enhanced = cv2.convertScaleAbs(gray, alpha=options.contrast, beta=0)
    output_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    metadata = {
        "scale": round(scale, 4),
        "original_size": [int(width), int(height)],
        "deskew_degrees": round(deskew_angle, 3),
        "output_size": [int(enhanced.shape[1]), int(enhanced.shape[0])],
    }
    return output_bgr, metadata


def prepare_signature_image(
    image_bgr: np.ndarray,
    options: PreprocessOptions,
) -> tuple[np.ndarray, float]:
    """Resize-only image for signature YOLO.

    The dedicated signature project does not denoise, increase contrast, or
    deskew before inference.  Those transforms can erase light pen strokes,
    so OCR receives its enhanced image while YOLO receives this colour copy.
    """
    height, width = image_bgr.shape[:2]
    short_edge, long_edge = min(height, width), max(height, width)
    scale = 1.0
    if short_edge < options.min_short_edge:
        scale = min(options.min_short_edge / short_edge, options.max_upscale_factor)
    if long_edge * scale > options.max_long_edge:
        scale = options.max_long_edge / long_edge
    if abs(scale - 1.0) <= 1e-6:
        return image_bgr.copy(), 1.0
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=interpolation), float(scale)