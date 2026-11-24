"""The SQLite source of truth for one resumable batch."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from Cheque_Processing.app.ingestion import InventoryItem
from Cheque_Processing.app.report import OUTPUT_FIELDS, normalize



def set_aside_old_format(batch_dir: Path) -> Path | None:
    """A batch folder made by the old ZIP version (its manifest has no
    source_path column) can never be resumed: rename it to
    <date>_old_format_<time>, never delete it. Returns the new path."""
    manifest = batch_dir / "manifest.db"
    if not manifest.is_file():
        return None
    db = sqlite3.connect(manifest)
    try:
        columns = {row[1] for row in db.execute("PRAGMA table_info(cheques)")}
    finally:
        db.close()
    if not columns or "source_path" in columns:
        return None
    target = batch_dir.with_name(f"{batch_dir.name}_old_format_{datetime.now():%Y%m%d%H%M%S}")
    batch_dir.rename(target)
    return target


class Manifest:
    def __init__(self, path: Path, batch_id: str, fingerprint: str, max_attempts: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.batch_id, self.max_attempts = path, batch_id, max_attempts
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cheques (
              cheque_id TEXT PRIMARY KEY, bank_code TEXT NOT NULL, source_path TEXT NOT NULL,
              sequence INTEGER NOT NULL, size INTEGER NOT NULL,
              status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
              last_error TEXT, result_json TEXT, output_emitted INTEGER NOT NULL DEFAULT 0,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        found = self._meta("fingerprint")
        if found and found != fingerprint:
            raise RuntimeError(
                f"{path} belongs to a different set of files for {batch_id} (the day was fetched again "
                "after processing started). Move that batch folder away to start the day over."
            )
        if not found:
            self._set_meta("batch_id", batch_id)
            self._set_meta("fingerprint", fingerprint)
            self.db.commit()

    def close(self) -> None:
        self.db.close()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, value))

    def seed(self, items: Iterable[InventoryItem]) -> None:
        rows = [(i.cheque_id, i.bank_code, i.source_path, i.sequence, i.size, "PENDING") for i in items]
        with self.db:
            self.db.executemany("""
                INSERT OR IGNORE INTO cheques
                (cheque_id, bank_code, source_path, sequence, size, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, rows)
        count = self.db.execute("SELECT COUNT(*) FROM cheques").fetchone()[0]
        if count != len(rows):
            raise RuntimeError("The day's file list differs from the existing manifest; refusing an unsafe resume.")

    def recover_interrupted(self) -> None:
        with self.db:
            self.db.execute("UPDATE cheques SET status='PENDING', updated_at=CURRENT_TIMESTAMP WHERE status='IN_PROGRESS'")

    def todo(self) -> list[sqlite3.Row]:
        return self.db.execute("""
            SELECT * FROM cheques WHERE status IN ('PENDING', 'RETRYABLE_FAILED')
            ORDER BY sequence
        """).fetchall()

    def mark_in_progress(self, cheque_id: str) -> None:
        with self.db:
            self.db.execute("""
                UPDATE cheques SET status='IN_PROGRESS', updated_at=CURRENT_TIMESTAMP
                WHERE cheque_id=? AND status IN ('PENDING', 'RETRYABLE_FAILED')
            """, (cheque_id,))

    def mark_done(self, cheque_id: str, result: dict[str, Any]) -> None:
        with self.db:
            self.db.execute("""
                UPDATE cheques SET status='DONE', result_json=?, last_error=NULL,
                updated_at=CURRENT_TIMESTAMP WHERE cheque_id=?
            """, (json.dumps(result, separators=(",", ":")), cheque_id))

    def mark_failure(self, cheque_id: str, error: str) -> None:
        row = self.db.execute("SELECT attempts FROM cheques WHERE cheque_id=?", (cheque_id,)).fetchone()
        attempts = int(row[0]) + 1
        status = "PERMANENT_FAILED" if attempts >= self.max_attempts else "RETRYABLE_FAILED"
        with self.db:
            self.db.execute("""
                UPDATE cheques SET status=?, attempts=?, last_error=?, updated_at=CURRENT_TIMESTAMP
                WHERE cheque_id=?
            """, (status, attempts, error[:4000], cheque_id))

    def done_not_emitted(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM cheques WHERE status='DONE' AND output_emitted=0 ORDER BY sequence").fetchall()

    def mark_emitted(self, cheque_id: str) -> None:
        with self.db:
            self.db.execute("UPDATE cheques SET output_emitted=1 WHERE cheque_id=?", (cheque_id,))

    def done_rows(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM cheques WHERE status='DONE' ORDER BY sequence").fetchall()

    def failures(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM cheques WHERE status='PERMANENT_FAILED' ORDER BY sequence").fetchall()

    def counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT status, COUNT(*) FROM cheques GROUP BY status").fetchall())

    def _output_row(self, row: sqlite3.Row) -> dict[str, Any]:
        result = json.loads(row["result_json"])
        return normalize(result)  # also converts rows written before the current layout

    def rebuild_output(self, path: Path) -> None:
        _atomic_csv(path, OUTPUT_FIELDS, (self._output_row(row) for row in self.done_rows()))
        with self.db:
            self.db.execute("UPDATE cheques SET output_emitted=1 WHERE status='DONE'")

    def append_output(self, path: Path, cheque_id: str) -> None:
        row = self.db.execute("SELECT * FROM cheques WHERE cheque_id=? AND status='DONE'", (cheque_id,)).fetchone()
        if not row or row["output_emitted"]:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(self._output_row(row))
            file.flush()
            os.fsync(file.fileno())
        self.mark_emitted(cheque_id)

    def write_failures(self, path: Path) -> None:
        fields = ["batch_date", "no", "bank", "file_name", "attempts", "last_error"]
        _atomic_csv(path, fields, ({
            "batch_date": self.batch_id, "no": row["sequence"], "bank": row["bank_code"],
            "file_name": row["cheque_id"].split("/", 1)[-1], "attempts": row["attempts"],
            "last_error": row["last_error"],
        } for row in self.failures()))


def _atomic_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
