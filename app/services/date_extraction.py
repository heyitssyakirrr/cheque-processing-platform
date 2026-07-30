"""Locate a date label with OCR and read the adjacent handwriting with TrOCR."""

import csv
import re
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.settings import settings


DATE_PATTERN = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
_processor: Any | None = None
_model: Any | None = None
_model_lock = threading.Lock()


def _trocr() -> tuple[Any, Any]:
    """Load the handwriting recognition model once per application process."""
    global _processor, _model

    if _model is None:
        with _model_lock:
            if _model is None:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                _processor = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
                _model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
    return _processor, _model


def _normalize(value: str) -> str:
    """Normalize an OCR string before matching label aliases."""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _find_anchor(lines: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-confidence text line matching a date-label alias."""
    candidates = []

    for line in lines:
        text = _normalize(line["text"])
        matches_alias = any(
            _normalize(alias) in text or text in _normalize(alias)
            for alias in settings.date_label_aliases
        )
        if matches_alias:
            candidates.append(line)

    return max(candidates, key=lambda line: line["confidence"]) if candidates else None


def _write_csv(output_dir: Path, result: dict[str, Any]) -> None:
    with (output_dir / "result.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)


def extract(
    image: np.ndarray,
    lines: list[dict[str, Any]],
    output_dir: Path,
    width: int = 450,
    height_pad: int = 25,
) -> dict[str, Any]:
    """Crop right of DATE/TARIKH and use TrOCR to read the handwritten date."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "anchor_not_found",
        "date": "",
        "raw_text": "",
        "anchor_text": "",
        "confidence": 0.0,
        "crop_box": [],
    }

    anchor = _find_anchor(lines)
    if anchor is None:
        _write_csv(output_dir, result)
        return result

    xs = [point[0] for point in anchor["box"]]
    ys = [point[1] for point in anchor["box"]]
    image_height, image_width = image.shape[:2]

    x0 = max(0, int(max(xs)))
    y0 = max(0, int(min(ys) - height_pad))
    x1 = min(image_width, x0 + width)
    y1 = min(image_height, int(max(ys) + height_pad))
    crop = image[y0:y1, x0:x1]

    if crop.size == 0:
        result["status"] = "empty_crop"
        result["anchor_text"] = anchor["text"]
        _write_csv(output_dir, result)
        return result

    crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(output_dir / "crop.png"), crop)

    processor, model = _trocr()
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixel_values = processor(
        images=Image.fromarray(crop_rgb),
        return_tensors="pt",
    ).pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=16, num_beams=4)
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    match = DATE_PATTERN.search(raw_text.replace(" ", ""))

    result.update(
        status="ok" if match else "needs_review",
        date=match.group(0) if match else "",
        raw_text=raw_text,
        anchor_text=anchor["text"],
        crop_box=[x0, y0, x1, y1],
    )
    _write_csv(output_dir, result)
    return result