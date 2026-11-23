"""Job 2: process one day's fetched cheques, resumably.

    python jobs/job2_process_cheques.py              today minus DATE_OFFSET_DAYS
    python jobs/job2_process_cheques.py --date 20261119
    python jobs/job2_process_cheques.py --date 20261113 --manual

Reads data/incoming/<date>/ (written by job 1; refuses to start until its
_FETCH_COMPLETE.json exists) and writes data/batches/<date>/<date>_output.csv
as cheques finish. Run the same command again to resume after a crash.

--manual: the .IMG files were copied into data/incoming/<date>/PBB/ and
PIBB/ by hand (no job 1). Job 2 lists what is there and processes it.

A batch folder left by the old ZIP version is renamed to
<date>_old_format_<time> (never deleted) so the day can be processed afresh.

Exit codes: 0 all done, 1 cheques left to retry (run again), 2 done but some
cheques permanently failed (see <date>_failed.csv), 3 job 1 has not finished
the day yet, or (--manual) no .IMG files in the bank folders.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import settings  # noqa: E402

if settings.threads_per_worker:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(settings.threads_per_worker))

import torch  # noqa: E402

from app.batch_date import add_date_argument, default_batch_date  # noqa: E402
from app.ingestion import (  # noqa: E402
    FetchNotComplete, batch_lock, fingerprint, inventory_day, write_json_atomic, write_manual_marker,
)
from app.job_log import setup_job_log  # noqa: E402
from app.manifest import Manifest, set_aside_old_format  # noqa: E402
from app.pipeline import process  # noqa: E402
from app.retention import BATCH_MARKER, apply_retention  # noqa: E402
from app.services import digit_classifier, signature_detection, text_detection  # noqa: E402
from app.services.date_extraction import is_flagged, is_valid_date  # noqa: E402
from app.services.img_conversion import convert_img_bytes_to_jpg  # noqa: E402
from app.services.preprocessing import PreprocessOptions  # noqa: E402

logger = logging.getLogger("job2")


def _warm_up() -> float:
    started = time.perf_counter()
    signature_detection._model_instance()
    if not settings.date_use_template_zone:
        text_detection._engine()
    digit_classifier._load()
    return time.perf_counter() - started


def _init_worker() -> None:
    import cv2
    if settings.threads_per_worker:
        cv2.setNumThreads(settings.threads_per_worker)
        torch.set_num_threads(settings.threads_per_worker)
    _warm_up()


def _completion_path(directory: Path, cheque_id: str) -> Path:
    import hashlib
    return directory / f"{hashlib.sha256(cheque_id.encode()).hexdigest()}.json"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, separators=(",", ":"))
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _process_one(task: dict) -> dict:
    """Worker function: success is made durable before the parent sees it."""
    started, counter = datetime.now(), time.perf_counter()
    jpeg = Path(task["tmp_dir"]) / f"{uuid.uuid4().hex}.jpg"
    try:
        raw = Path(task["image_path"]).read_bytes()
        convert_img_bytes_to_jpg(raw, jpeg)
        _, result = process(jpeg, task["options"], settings.signature_decision_threshold, 450,
                            task["batch_id"], task["sequence"], Path(task["source_path"]).name)
        timings = result.get("timings", {})
        signature = "YES" if result["signature"]["exists"] else "NO"
        valid_date = "YES" if is_valid_date(result["date"]["date"]) else "NO"
        row = {
            "batch_id": task["batch_id"], "cheque_id": task["cheque_id"], "bank_code": task["bank_code"],
            "sequence": task["sequence"], "filename": Path(task["source_path"]).name,
            "signature_detected": signature, "date": result["date"]["date"],
            "valid_date": valid_date, "is_flagged": is_flagged(signature, valid_date),
            "start_time": started.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.perf_counter() - counter, 2), "pixels": result.get("pixels", ""),
            "t_read": timings.get("read", ""), "t_preprocess": timings.get("preprocess", ""),
            "t_signature": timings.get("signature", ""), "t_ocr": timings.get("ocr", ""),
            "t_date": timings.get("date", ""), "error": "",
        }
        _atomic_json(_completion_path(Path(task["completion_dir"]), task["cheque_id"]), row)
        return {"cheque_id": task["cheque_id"], "ok": True}
    except Exception as error:  # A bad scan must not end the whole batch.
        return {"cheque_id": task["cheque_id"], "ok": False, "error": f"{type(error).__name__}: {error}"}
    finally:
        jpeg.unlink(missing_ok=True)


def _import_completions(manifest: Manifest, directory: Path, output_csv: Path) -> int:
    imported = 0
    for path in directory.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            manifest.mark_done(row["cheque_id"], row)
            manifest.append_output(output_csv, row["cheque_id"])
            imported += 1
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError(f"Invalid completion record {path}: {error}") from error
    return imported


def _task(row, batch_id: str, day_dir: Path, options: PreprocessOptions, tmp_dir: Path, completion_dir: Path) -> dict:
    return {"cheque_id": row["cheque_id"], "bank_code": row["bank_code"], "source_path": row["source_path"],
            "image_path": str(day_dir / row["source_path"]), "sequence": row["sequence"], "batch_id": batch_id,
            "options": options, "tmp_dir": str(tmp_dir), "completion_dir": str(completion_dir)}


def _handle_result(manifest: Manifest, result: dict, completion_dir: Path, output_csv: Path) -> None:
    cheque_id = result["cheque_id"]
    if result["ok"]:
        row = json.loads(_completion_path(completion_dir, cheque_id).read_text(encoding="utf-8"))
        manifest.mark_done(cheque_id, row)
        manifest.append_output(output_csv, cheque_id)
    else:
        manifest.mark_failure(cheque_id, result["error"])
        logger.warning("%s failed: %s", cheque_id, result["error"])


def _progress(done: int, total: int, task: dict, ok: bool) -> None:
    # Every cheque on a small run; every 500th (and the last) on a 20,000-cheque day.
    if total <= 50 or done % 500 == 0 or done == total:
        logger.info("[%d/%d] %s -> %s", done, total, task["cheque_id"], "DONE" if ok else "FAILED")


def _run_tasks(manifest: Manifest, tasks: list[dict], workers: int, completion_dir: Path, output_csv: Path) -> None:
    if not tasks:
        return
    if workers == 1:
        logger.info("Warming up models... (%.2fs)", _warm_up())
        for index, task in enumerate(tasks, 1):
            manifest.mark_in_progress(task["cheque_id"])
            result = _process_one(task)
            _handle_result(manifest, result, completion_dir, output_csv)
            _progress(index, len(tasks), task, result["ok"])
        return

    capacity, iterator, completed = workers * settings.batch_inflight_multiplier, iter(tasks), 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        pending = {}
        def submit() -> bool:
            try:
                task = next(iterator)
            except StopIteration:
                return False
            manifest.mark_in_progress(task["cheque_id"])
            pending[executor.submit(_process_one, task)] = task
            return True
        while len(pending) < capacity and submit():
            pass
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"cheque_id": task["cheque_id"], "ok": False, "error": str(error)}
                _handle_result(manifest, result, completion_dir, output_csv)
                completed += 1
                _progress(completed, len(tasks), task, result["ok"])
                submit()


def run_batch(batch_id: str, manual: bool = False) -> int:
    day_dir = settings.incoming_day_dir(batch_id)
    batch_dir = settings.batch_dir(batch_id)
    completion_dir, tmp_dir = batch_dir / "completion", batch_dir / "tmp"
    output_csv, failed_csv = batch_dir / f"{batch_id}_output.csv", batch_dir / f"{batch_id}_failed.csv"
    if manual:
        count = write_manual_marker(day_dir, settings.banks)
        logger.info("--manual: using the %d .IMG file(s) in %s as they are (no job 1).", count, day_dir)
    moved = set_aside_old_format(batch_dir)
    if moved:
        logger.warning("%s was made by the old ZIP version; moved to %s (nothing deleted).", batch_dir, moved)
    try:
        inventory = inventory_day(day_dir, settings.banks)
    except FetchNotComplete as error:
        logger.error("%s", error)
        return 3
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with batch_lock(batch_dir / ".lock"):
        for leftover in tmp_dir.glob("*.jpg"):
            leftover.unlink(missing_ok=True)
        manifest = Manifest(batch_dir / "manifest.db", batch_id, fingerprint(inventory), settings.max_attempts)
        try:
            manifest.seed(inventory)
            manifest.recover_interrupted()
            manifest.rebuild_output(output_csv)  # repairs crash ambiguity before new appends
            recovered = _import_completions(manifest, completion_dir, output_csv)
            todo = manifest.todo()
            workers = min(settings.recommend_workers(os.cpu_count() or 1, len(todo)), len(todo)) if todo else 1
            by_bank = {bank: sum(1 for item in inventory if item.bank_code == bank) for bank in settings.banks}
            logger.info("Batch %s: %d cheques %s; %d to process; %d worker(s).",
                        batch_id, len(inventory), by_bank, len(todo), workers)
            if recovered:
                logger.info("Recovered %d durable completion(s).", recovered)
            options = PreprocessOptions(
                max_long_edge=settings.preprocess_max_long_edge,
                contrast=settings.preprocess_contrast,
                denoise=settings.preprocess_denoise,
                deskew=settings.preprocess_deskew,
            )
            started = time.perf_counter()
            _run_tasks(manifest, [_task(row, batch_id, day_dir, options, tmp_dir, completion_dir) for row in todo],
                       workers, completion_dir, output_csv)
            manifest.write_failures(failed_csv)
            counts = manifest.counts()
            left, failed = len(manifest.todo()), len(manifest.failures())
            logger.info("Output: %s", output_csv)
            logger.info("Status: %s (%.0fs this run)", counts, time.perf_counter() - started)
            if not left:
                write_json_atomic(batch_dir / BATCH_MARKER, {
                    "batch_id": batch_id, "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "counts": counts,
                })
            if failed:
                logger.warning("%d cheque(s) permanently failed: %s", failed, failed_csv)
            return 1 if left else (2 if failed else 0)
        finally:
            manifest.close()


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()  # a crash inside a C library prints where, instead of dying silently
    parser = argparse.ArgumentParser(description="Job 2: process one day's fetched cheques, resumably.")
    add_date_argument(parser)
    parser.add_argument("--manual", action="store_true",
                        help="The .IMG files were put into data/incoming/<date>/<bank>/ by hand: "
                             "use them as they are, without waiting for job 1")
    args = parser.parse_args(argv)
    batch_id = args.date or default_batch_date()
    setup_job_log("job2", batch_id)
    try:
        code = run_batch(batch_id, manual=args.manual)
    except RuntimeError as error:  # a problem with the input folder, explained in the message
        logger.error("%s", error)
        code = 3
    try:
        apply_retention()
    except Exception:  # Housekeeping must never change the batch's result.
        logger.exception("Retention cleanup failed")
    logger.info("Exit code %d", code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
