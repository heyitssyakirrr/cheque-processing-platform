"""Job 2's input checks (app/ingestion.py), the manifest, and retention (app/retention.py)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import DATE, TempProject, run  # noqa: E402

from Cheque_Processing.app import fetch, retention  # noqa: E402
from Cheque_Processing.app.ingestion import (  # noqa: E402
    FetchNotComplete, fingerprint, inventory_day, write_json_atomic, write_manual_marker,
)
from Cheque_Processing.app.manifest import Manifest, set_aside_old_format  # noqa: E402
from Cheque_Processing.app.settings import settings  # noqa: E402


def _fetched(project: TempProject, pbb: int = 3, pibb: int = 2) -> Path:
    project.add_images("PBB", pbb), project.add_images("PIBB", pibb)
    project.add_trigger("PBB"), project.add_trigger("PIBB")
    assert fetch.fetch_day(DATE).code == 0
    return settings.incoming_day_dir(DATE)


def _raises(error_type, function, text=""):
    try:
        function()
    except error_type as error:
        assert text in str(error), error
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_inventory_needs_the_marker():
    with TempProject():
        _raises(FetchNotComplete, lambda: inventory_day(settings.incoming_day_dir(DATE), settings.banks))


def test_inventory_checks_every_file():
    with TempProject() as project:
        day = _fetched(project)
        items = inventory_day(day, settings.banks)
        assert [item.sequence for item in items] == [1, 2, 3, 4, 5]
        assert {item.bank_code for item in items} == {"PBB", "PIBB"}
        first, second = sorted((day / "PBB").iterdir())[:2]
        first.unlink()
        second.write_bytes(b"truncated")
        _raises(RuntimeError, lambda: inventory_day(day, settings.banks), "is missing")


def test_fingerprint_changes_with_the_file_set():
    with TempProject() as project:
        day = _fetched(project)
        before = fingerprint(inventory_day(day, settings.banks))
        assert before == fingerprint(inventory_day(day, settings.banks))
        project.add_images("PIBB", 4)
        assert fetch.fetch_day(DATE, force=True).code == 0
        assert fingerprint(inventory_day(day, settings.banks)) != before


def test_manifest_seeds_and_refuses_another_file_set():
    with TempProject() as project:
        day = _fetched(project)
        items = inventory_day(day, settings.banks)
        path = settings.batch_dir(DATE) / "manifest.db"
        manifest = Manifest(path, DATE, fingerprint(items), 3)
        manifest.seed(items)
        assert len(manifest.todo()) == 5 and manifest.todo()[0]["source_path"] == items[0].source_path
        manifest.close()
        _raises(RuntimeError, lambda: Manifest(path, DATE, "different", 3), "different set of files")


def _day_folders(*days: str, finished: tuple[str, ...] = ()) -> None:
    for day in days:
        (settings.incoming_day_dir(day) / "PBB").mkdir(parents=True)
        settings.batch_dir(day).mkdir(parents=True)
        if day in finished:
            write_json_atomic(settings.batch_dir(day) / retention.BATCH_MARKER, {"batch_id": day})


def test_retention_deletes_only_old_finished_images():
    with TempProject():
        _day_folders("20261001", "20261005", "20261020", "20261119",
                     finished=("20261001", "20261020", "20261119"))
        (settings.incoming_dir / "notes").mkdir()  # not a date: never touched
        report = retention.apply_retention(today=date(2026, 11, 19))  # cutoff 2026-10-20
        assert report["incoming_deleted"] == ["20261001"]
        assert report["kept_unfinished"] == ["incoming/20261005"]
        assert sorted(p.name for p in settings.incoming_dir.iterdir()) == ["20261005", "20261020", "20261119", "notes"]
        assert len(list(settings.batches_dir.iterdir())) == 4  # BATCH_RETENTION_DAYS=0 keeps output


def test_batch_retention_when_set():
    with TempProject():
        settings.batch_retention_days = 40
        _day_folders("20261001", "20261015", finished=("20261001", "20261015"))
        report = retention.apply_retention(today=date(2026, 11, 19))  # batches cutoff 2026-10-10
        assert report["incoming_deleted"] == ["20261001", "20261015"]
        assert report["batches_deleted"] == ["20261001"]


def test_retention_zero_keeps_everything():
    with TempProject():
        settings.incoming_retention_days = 0
        _day_folders("20250101", finished=("20250101",))
        assert retention.apply_retention(today=date(2026, 11, 19))["incoming_deleted"] == []


def test_manual_marker_lists_hand_copied_files():
    with TempProject():
        day = settings.incoming_day_dir(DATE)
        for bank, names in (("PBB", ["A.IMG", "b.img"]), ("PIBB", ["C.IMG"])):
            (day / bank).mkdir(parents=True)
            for name in names:
                (day / bank / name).write_bytes(b"x" * 10)
        (day / "PBB" / "notes.txt").write_text("not a cheque")
        assert write_manual_marker(day, settings.banks) == 3
        items = inventory_day(day, settings.banks)
        assert [i.cheque_id for i in items] == ["PBB/A.IMG", "PBB/b.img", "PIBB/C.IMG"]


def test_manual_marker_refuses_an_empty_day():
    with TempProject():
        day = settings.incoming_day_dir(DATE)
        day.mkdir(parents=True)
        (day / "LOOSE.IMG").write_bytes(b"x")
        try:
            write_manual_marker(day, settings.banks)
        except RuntimeError as error:
            assert "move them into a bank folder" in str(error)
        else:
            raise AssertionError("expected RuntimeError")


def test_old_zip_batch_folder_is_set_aside():
    import sqlite3

    with TempProject():
        batch_dir = settings.batch_dir(DATE)
        batch_dir.mkdir(parents=True)
        db = sqlite3.connect(batch_dir / "manifest.db")
        db.execute("CREATE TABLE cheques (cheque_id TEXT PRIMARY KEY, zip_member TEXT, crc INTEGER)")
        db.commit()
        db.close()
        moved = set_aside_old_format(batch_dir)
        assert moved and moved.name.startswith(f"{DATE}_old_format_") and (moved / "manifest.db").is_file()
        assert not batch_dir.exists()
        # A current-format batch folder is left alone.
        Manifest(batch_dir / "manifest.db", DATE, "abc", 3).close()
        assert set_aside_old_format(batch_dir) is None and batch_dir.exists()


if __name__ == "__main__":
    run([value for name, value in list(globals().items()) if name.startswith("test_")])
