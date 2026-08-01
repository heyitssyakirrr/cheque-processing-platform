"""Orchestrate one cheque through all independent processing services."""

import csv
import json
import re
from pathlib import Path
from typing import Any

import cv2

from app.services.date_extraction import extract as extract_date
from app.services.field_extraction import extract as extract_fields
from app.services.preprocessing import PreprocessOptions, prepare_signature_image, preprocess
from app.services.signature_detection import detect as detect_signatures
from app.services.text_detection import detect as detect_text
from app.settings import settings


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(value).stem.lower()).strip("-")[:48] or "cheque"


def _create_run_directory(batch_id: str, sequence: int, source_name: str) -> tuple[str, Path]:
    """Keep every run self-contained while making files sort by batch/order."""
    run_id = f"{batch_id}-{sequence:03d}-{_safe_name(source_name)}"
    run_dir = settings.runs_dir / run_id

    for name in ("original", "preprocessed", "signature", "text", "date"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    return run_id, run_dir


def _write_summary(run_dir: Path, result: dict[str, Any]) -> None:
    """Save the most useful result fields in a one-row CSV."""
    row = {
        "run_id": result["run_id"],
        "signature_exists": result["signature"]["exists"],
        "signature_count": len(result["signature"]["detections"]),
        "date": result["date"]["date"],
        "date_status": result["date"]["status"],
        "payee_candidate": result["fields"]["payee_candidate"],
        "amount_candidate": result["fields"]["amount_candidate"],
        "text_count": result["text"]["count"],
    }

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def process(
    source: Path,
    options: PreprocessOptions,
    yolo_threshold: float,
    date_width: int,
    batch_id: str,
    sequence: int,
    source_name: str,
) -> tuple[str, dict[str, Any]]:
    """Process one uploaded cheque and persist all intermediate artifacts."""
    run_id, run_dir = _create_run_directory(batch_id, sequence, source_name)

    original = cv2.imread(str(source))
    if original is None:
        raise ValueError("Unsupported or corrupt image")

    cv2.imwrite(str(run_dir / "original" / "cheque.png"), original)

    preprocessed, preprocessing_metadata = preprocess(original, options)
    cv2.imwrite(str(run_dir / "preprocessed" / "cheque.png"), preprocessed)

    signature_image, signature_scale = prepare_signature_image(original, options)
    preprocessing_metadata["signature_scale"] = round(signature_scale, 4)

    signature_detections = detect_signatures(
        signature_image,
        run_dir / "signature",
        decision_threshold=yolo_threshold,
    )
    text_lines = detect_text(preprocessed, run_dir / "text")
    date_result = extract_date(
        preprocessed,
        text_lines,
        run_dir / "date",
        width=date_width,
    )
    field_result = extract_fields(text_lines)

    result: dict[str, Any] = {
        "run_id": run_id,
        "batch_id": batch_id,
        "sequence": sequence,
        "source_filename": source_name,
        "preprocessing": preprocessing_metadata,
        "signature": {
            "exists": any(item["accepted"] for item in signature_detections),
            "detections": signature_detections,
        },
        "text": {
            "count": len(text_lines),
            "detections": text_lines,
        },
        "date": date_result,
        "fields": field_result,
    }

    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    _write_summary(run_dir, result)
    return run_id, result