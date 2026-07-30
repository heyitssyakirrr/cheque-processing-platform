"""Safe, image-wide preprocessing for cheque scans and photographs."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessOptions:
    """User-adjustable preprocessing settings."""

    max_long_edge: int = 2200
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
    scale = min(1.0, options.max_long_edge / max(height, width))

    if scale < 1:
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
        "deskew_degrees": round(deskew_angle, 3),
        "output_size": [int(enhanced.shape[1]), int(enhanced.shape[0])],
    }
    return output_bgr, metadata