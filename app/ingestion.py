"""The fetched day folder: its completion marker, inventory, and single-run locking.

Job 1 copies one day's .IMG files into

    data/incoming/<YYYYMMDD>/<bank>/<name>.IMG

and writes ``_FETCH_COMPLETE.json`` there last, listing every file and its
size. Job 2 only starts from that marker, and checks every listed file is
on disk with the same size before processing anything.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FETCH_MARKER = "_FETCH_COMPLETE.json"


@dataclass(frozen=True)
class InventoryItem:
    cheque_id: str      # "<bank>/<name>.IMG", unique within the day
    bank_code: str
    source_path: str    # relative to the day folder (same as cheque_id)
    sequence: int
    size: int


class FetchNotComplete(RuntimeError):
    """The day folder has no completion marker yet (job 1 has not finished)."""


def is_image_name(name: str) -> bool:
    """The source writes upper-case .IMG; Linux is case-sensitive, so match any case."""
    return name.lower().endswith(".img") and not name.startswith(".")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=1)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def read_fetch_marker(day_dir: Path) -> dict[str, Any]:
    marker = day_dir / FETCH_MARKER
    if not marker.is_file():
        raise FetchNotComplete(f"No {FETCH_MARKER} in {day_dir}: job 1 has not finished this day")
    return json.loads(marker.read_text(encoding="utf-8"))


def write_manual_marker(day_dir: Path, banks: list[str]) -> int:
    """For files put into data/incoming/<date>/<bank>/ by hand (job 2
    --manual): write the same _FETCH_COMPLETE.json job 1 writes, listing
    every .IMG file there now. Returns how many files it lists."""
    files = {
        f"{bank}/{path.name}": path.stat().st_size
        for bank in banks
        for path in sorted((day_dir / bank).glob("*"))
        if path.is_file() and is_image_name(path.name)
    }
    if not files:
        loose = [p.name for p in day_dir.glob("*") if p.is_file() and is_image_name(p.name)]
        hint = f" ({len(loose)} .IMG file(s) found directly in {day_dir}: move them into a bank folder)" if loose else ""
        raise RuntimeError(f"No .IMG files in {day_dir}/{{{','.join(banks)}}}/{hint}")
    write_json_atomic(day_dir / FETCH_MARKER, {
        "batch_id": day_dir.name, "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "source": "manual (files put in this folder by hand)",
        "banks": sorted({cheque_id.split("/")[0] for cheque_id in files}),
        "total_files": len(files), "files": files,
    })
    return len(files)


def inventory_day(day_dir: Path, banks: list[str]) -> list[InventoryItem]:
    """The day's cheques, from job 1's marker, after checking each file on disk."""
    marker = read_fetch_marker(day_dir)
    files: dict[str, int] = marker["files"]
    problems = []
    items = []
    for cheque_id in sorted(files):
        parts = cheque_id.split("/")
        if len(parts) != 2 or parts[0] not in banks or not is_image_name(parts[1]) or ".." in parts:
            problems.append(f"unexpected entry {cheque_id!r}")
            continue
        path = day_dir / parts[0] / parts[1]
        if not path.is_file():
            problems.append(f"{cheque_id} is missing")
        elif path.stat().st_size != files[cheque_id]:
            problems.append(f"{cheque_id} is {path.stat().st_size} bytes, the source had {files[cheque_id]}")
        items.append(InventoryItem(cheque_id, parts[0], cheque_id, len(items) + 1, files[cheque_id]))
    if problems:
        shown = "; ".join(problems[:10]) + (f"; and {len(problems) - 10} more" if len(problems) > 10 else "")
        raise RuntimeError(f"{day_dir} does not match {FETCH_MARKER} ({shown}). Rerun job 1 with --force.")
    return items


def fingerprint(items: list[InventoryItem]) -> str:
    """Identifies the day's file set, so a resume never mixes two different fetches."""
    digest = hashlib.sha256()
    for item in items:
        digest.update(f"{item.cheque_id}\t{item.size}\n".encode("utf-8"))
    return digest.hexdigest()


@contextmanager
def batch_lock(path: Path):
    """Prevent two manual/scheduler processes from owning one batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"Already running (lock: {path})") from error
            try:
                yield
            finally:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(f"Already running (lock: {path})") from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
