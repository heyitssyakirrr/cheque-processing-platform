"""Process every cheque image in Upload/ and write output.csv.

Replaces the removed web app: run it directly, no server, no browser.

Total work per cheque is fixed, so worker_count and threads_per_worker only
divide the same cores up differently. worker_count=1 with
threads_per_worker=0 reproduces the original single-image-on-all-cores run.

Per-image evidence (annotated signature, date crop, digit slices) is only
written under data/batches/<run timestamp>/ when settings.save_artifacts is
on -- it costs roughly 20 file writes per cheque.
"""

import os

# Importing settings is cheap and pulls in no native library, so it can
# safely precede the thread-count environment variables below.
from app.settings import settings

# OpenMP, MKL and OpenBLAS read these once, when the native library first
# loads, so they must be set before cv2/torch/paddle are imported.
if settings.threads_per_worker:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, str(settings.threads_per_worker))

import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from app.pipeline import process  # noqa: E402
from app.services import digit_classifier, signature_detection, text_detection  # noqa: E402
from app.services.preprocessing import PreprocessOptions  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

OUTPUT_FIELDS = [
    "filename",
    "signature_detected",
    "date",
    "start_time",
    "end_time",
    "duration_seconds",
    "pixels",
    "t_read",
    "t_preprocess",
    "t_signature",
    "t_ocr",
    "t_date",
    "error",
]


def _row(
    filename: str,
    started: datetime,
    elapsed: float,
    signature_detected: str = "",
    date: str = "",
    error: str = "",
    pixels: str = "",
    timings: dict | None = None,
) -> dict:
    timings = timings or {}
    return {
        "filename": filename,
        "signature_detected": signature_detected,
        "date": date,
        "start_time": started.strftime(TIMESTAMP_FORMAT),
        "end_time": datetime.now().strftime(TIMESTAMP_FORMAT),
        "duration_seconds": round(elapsed, 2),
        "pixels": pixels,
        "t_read": timings.get("read", ""),
        "t_preprocess": timings.get("preprocess", ""),
        "t_signature": timings.get("signature", ""),
        "t_ocr": timings.get("ocr", ""),
        "t_date": timings.get("date", ""),
        "error": error,
    }


def _warm_up() -> float:
    """Load YOLO, PaddleOCR and the digit model once, up front.

    Each service caches its model in a module-level singleton, so the first
    cheque would otherwise absorb all three load times. Call this once at
    startup -- an API should do the same before accepting requests.
    """
    started = time.perf_counter()
    signature_detection._model_instance()
    if not settings.date_use_template_zone:
        text_detection._engine()
    digit_classifier._load()
    return time.perf_counter() - started


def _init_worker() -> None:
    """Run once per worker process, before it takes any image."""
    import cv2
    import torch

    if settings.threads_per_worker:
        cv2.setNumThreads(settings.threads_per_worker)
        torch.set_num_threads(settings.threads_per_worker)
    _warm_up()


def _limit_cpus(count: int) -> int:
    """Confine this process and its children to `count` CPUs.

    Thread-count settings are per-library and easy to miss one of; affinity is
    enforced by the kernel, so nothing in the tree can exceed it. Linux only.

    Returns the number of CPUs actually usable afterwards.
    """
    if not hasattr(os, "sched_setaffinity"):
        return os.cpu_count() or 1
    available = sorted(os.sched_getaffinity(0))
    if count >= len(available):
        return len(available)
    os.sched_setaffinity(0, set(available[:count]))
    return count


def _resolve_worker_count(task_count: int) -> tuple[int, int, int]:
    """Return (workers, logical_cores_seen, affinity_cores_seen)."""
    logical_cores = os.cpu_count() or 1
    requested_workers = settings.recommend_workers(logical_cores, task_count)
    affinity_cores = _limit_cpus(requested_workers)
    workers = min(requested_workers, affinity_cores, task_count)
    return workers, logical_cores, affinity_cores


def _process_one(task: tuple[int, Path, PreprocessOptions, str]) -> dict:
    """Handle a single cheque and return its output.csv row."""
    sequence, image_path, options, batch_id = task
    started, counter = datetime.now(), time.perf_counter()
    try:
        _run_id, result = process(
            image_path,
            options,
            settings.signature_decision_threshold,
            450,
            batch_id,
            sequence,
            image_path.name,
        )
        return _row(
            image_path.name,
            started,
            time.perf_counter() - counter,
            signature_detected="YES" if result["signature"]["exists"] else "NO",
            date=result["date"]["date"],
            pixels=result.get("pixels", ""),
            timings=result.get("timings"),
        )
    except Exception as error:  # noqa: BLE001 -- one bad image shouldn't abort the rest
        return _row(image_path.name, started, time.perf_counter() - counter, error=str(error))


def main() -> int:
    upload_dir = settings.upload_dir
    if not upload_dir.is_dir():
        print(f"Upload folder not found: {upload_dir}", file=sys.stderr)
        return 1

    images = sorted(
        path for path in upload_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images found in {upload_dir}", file=sys.stderr)
        return 1

    options = PreprocessOptions(
        max_long_edge=settings.preprocess_max_long_edge,
        contrast=settings.preprocess_contrast,
        denoise=settings.preprocess_denoise,
        deskew=settings.preprocess_deskew,
    )
    batch_id = f"{datetime.now():%Y%m%d-%H%M%S}"
    tasks = [(sequence, path, options, batch_id) for sequence, path in enumerate(images, start=1)]

    workers, logical_cores, cores = _resolve_worker_count(len(tasks))
    if workers > cores:
        print(
            f"WORKER_COUNT={workers} exceeds the {cores} usable cores; "
            f"capping to {cores}. Oversubscribing makes every image slower.",
            file=sys.stderr,
        )
    print(
        f"{len(tasks)} images | {logical_cores} logical cores detected "
        f"| {cores} usable cores | {workers} workers "
        f"| {torch.get_num_threads()} torch threads each",
        flush=True,
    )
    if settings.worker_count == 0:
        print(
            "Worker auto mode: "
            f"utilization_target={settings.worker_utilization_target:.2f}, "
            f"reserved_logical_cores={settings.reserved_logical_cores}, "
            f"max_worker_cap={settings.max_worker_cap}",
            flush=True,
        )

    started_at = time.perf_counter()
    rows: list[dict] = []

    if workers == 1:
        print("Warming up models...", flush=True)
        print(f"Models ready in {_warm_up():.2f}s\n", flush=True)
        results = map(_process_one, tasks)
    else:
        print(f"Starting {workers} workers (models load in each)...", flush=True)
        executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
        # map keeps submission order, so rows stay aligned with `images`.
        results = executor.map(_process_one, tasks, chunksize=1)

    try:
        for done, row in enumerate(results, start=1):
            status = row["error"] or row["signature_detected"]
            print(f"[{done}/{len(tasks)}] {row['filename']} -> {status}", flush=True)
            rows.append(row)
    finally:
        if workers > 1:
            executor.shutdown()

    with settings.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    wall = time.perf_counter() - started_at
    failed = sum(1 for row in rows if row["error"])
    print(f"\nWrote {settings.output_csv} ({len(rows)} rows, {failed} failed)")
    print(f"Wall clock {wall:.1f}s -- {len(rows) / wall:.2f} cheques/sec")
    if settings.save_artifacts:
        print(f"Evidence: {settings.batches_dir / batch_id}")
    return 1 if failed == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())