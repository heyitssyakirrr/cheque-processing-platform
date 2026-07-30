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
    decision_threshold: float = 0.50,
    logging_floor: float = 0.10,
) -> list[dict[str, Any]]:
    """Detect signatures and save annotated image, accepted crops, and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    inference = _model_instance().predict(
        image,
        conf=logging_floor,
        iou=0.70,
        verbose=False,
    )[0]

    records: list[dict[str, Any]] = []
    annotated = image.copy()
    crop_dir = output_dir / "crops"

    boxes = inference.boxes if inference.boxes is not None else []
    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
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
            crop = image[max(0, y1):y2, max(0, x1):x2]
            cv2.imwrite(str(crop_dir / f"signature_{index}.png"), crop)

    cv2.imwrite(str(output_dir / "annotated.png"), annotated)
    _write_csv(output_dir, records)
    return records