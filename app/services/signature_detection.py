"""YOLOv8/TorchScript signature detection and result artifacts."""

import csv
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from app.settings import settings


_model: YOLO | None = None
_model_lock = threading.Lock()


def _model_instance() -> YOLO:
    """Load the local model once, downloading it only when explicitly configured."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                model_path = settings.model_path

                if (
                    not model_path.exists()
                    and settings.hf_yolo_repo_id
                    and settings.hf_yolo_filename
                ):
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    downloaded_path = hf_hub_download(
                        settings.hf_yolo_repo_id,
                        settings.hf_yolo_filename,
                        local_dir=model_path.parent,
                    )
                    model_path = Path(downloaded_path)

                if not model_path.exists():
                    raise FileNotFoundError(
                        "YOLO model missing. Put it at the path in "
                        "YOLO_MODEL_PATH or configure "
                        "HF_YOLO_REPO_ID/HF_YOLO_FILENAME."
                    )

                _model = YOLO(str(model_path))

    return _model


def _write_csv(output_dir: Path, records: list[dict[str, Any]]) -> None:
    fields = ["id", "confidence", "accepted", "x1", "y1", "x2", "y2"]
    with (output_dir / "detections.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def detect(
    image: np.ndarray,
    output_dir: Path,
    decision_threshold: float = 0.35,
    logging_floor: float = 0.10,
    crop_offset: tuple[int, int] = (0, 0),
    full_image: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Detect signatures and save annotated image, accepted crops, and CSV.

    `image` is what inference actually runs on -- pass a zone crop (see
    preprocessing.crop_signature_zone) to search a smaller, targeted area
    instead of the whole cheque. `crop_offset` is that crop's (x, y) origin
    in `full_image`'s coordinate frame; every detected box gets translated
    by this offset before being recorded/drawn, so annotations and crops
    still land correctly on the full cheque, not just the searched zone.
    If `full_image` is omitted, `image` is used for both inference and
    annotation (offset should stay (0, 0) in that case).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    offset_x, offset_y = crop_offset
    canvas = full_image if full_image is not None else image

    inference = _model_instance().predict(
        image,
        # Match the proven signature-test setup: preserve fine pen strokes
        # with a larger inference canvas and retain low-score candidates for
        # review.  Acceptance remains controlled by the UI threshold.
        imgsz=settings.signature_imgsz,
        conf=min(logging_floor, settings.signature_logging_floor),
        iou=settings.signature_iou,
        max_det=2,
        verbose=False,
    )[0]

    records: list[dict[str, Any]] = []
    annotated = canvas.copy()
    crop_dir = output_dir / "crops"

    # Always visible, regardless of detections -- lets you check the zone
    # fractions in settings.py against where a signature actually sits,
    # instead of guessing whether they're right.
    zone_h, zone_w = image.shape[:2]
    cv2.rectangle(
        annotated,
        (offset_x, offset_y),
        (offset_x + zone_w, offset_y + zone_h),
        (255, 140, 0),
        2,
    )
    cv2.putText(
        annotated,
        "search zone",
        (offset_x + 4, offset_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 140, 0),
        2,
    )

    boxes = inference.boxes if inference.boxes is not None else []
    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, x2 = x1 + offset_x, x2 + offset_x
        y1, y2 = y1 + offset_y, y2 + offset_y
        confidence = float(box.conf[0])
        accepted = confidence >= decision_threshold
        record = {
            "id": index,
            "confidence": round(confidence, 4),
            "accepted": accepted,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
        records.append(record)

        color = (0, 170, 0) if accepted else (0, 140, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated,
            f"signature {confidence:.2f}",
            (x1, max(22, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

        if accepted:
            crop_dir.mkdir(exist_ok=True)
            crop = canvas[max(0, y1):y2, max(0, x1):x2]
            cv2.imwrite(str(crop_dir / f"signature_{index}.png"), crop)

    cv2.imwrite(str(output_dir / "annotated.png"), annotated)
    _write_csv(output_dir, records)
    return records