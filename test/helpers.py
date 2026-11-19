"""Shared test setup: a temporary project root and a fake source tree."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import settings  # noqa: E402

DATE = "20261119"
TRIGGER = "{root}/{bank}/{date}/GWC/DONE.TRG"


class TempProject:
    """Points settings at a temp folder (root + a local fake source); restores them after."""

    OVERRIDES = dict(source_transport="local", source_banks="PBB,PIBB", source_trigger_pattern=TRIGGER,
                     source_images_pattern="{root}/{bank}/{date}/GWC/Images", fetch_workers=3,
                     fetch_file_attempts=2, fetch_poll_minutes=10, fetch_give_up_at="20:00",
                     incoming_root="data/incoming", incoming_retention_days=30, batch_retention_days=0)

    def __enter__(self) -> "TempProject":
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "source" / "Centric"
        self.saved = {name: getattr(settings, name) for name in [*self.OVERRIDES, "root", "source_root", "sftp_host",
                                                      "sftp_username", "sftp_password", "sftp_key_path"]}
        for name, value in self.OVERRIDES.items():
            setattr(settings, name, value)
        settings.root = self.tmp / "project"
        settings.source_root = str(self.source)
        return self

    def __exit__(self, *exc) -> None:
        for name, value in self.saved.items():
            setattr(settings, name, value)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def images_dir(self, bank: str, date: str = DATE) -> Path:
        return self.source / bank / date / "GWC" / "Images"

    def add_images(self, bank: str, count: int, date: str = DATE, size: int = 100) -> dict[str, int]:
        folder = self.images_dir(bank, date)
        folder.mkdir(parents=True, exist_ok=True)
        made = {}
        for index in range(count):
            name = f"2026101302330410110{index:05d}0009.IMG"
            data = bytes([index % 256]) * (size + index)
            (folder / name).write_bytes(data)
            made[f"{bank}/{name}"] = len(data)
        return made

    def add_trigger(self, bank: str, date: str = DATE) -> None:
        path = Path(TRIGGER.format(root=self.source, bank=bank, date=date))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("done")


def run(tests: list) -> None:
    """Tiny runner so each test file works as `python test/<file>.py` (also from a prefixed copy)."""
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok    {test.__name__}")
        except Exception:
            failed += 1
            import traceback
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed", file=sys.stderr)
    raise SystemExit(1 if failed else 0)
