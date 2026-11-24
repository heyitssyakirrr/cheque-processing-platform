"""Job 1 (app/fetch.py) against a fake local source tree; SftpSource against a fake SFTP session."""

from __future__ import annotations

import json
import stat
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import DATE, TempProject, run  # noqa: E402

from Cheque_Processing.app import fetch  # noqa: E402
from Cheque_Processing.app.ingestion import FETCH_MARKER, inventory_day  # noqa: E402
from Cheque_Processing.app.settings import settings  # noqa: E402


def _marker() -> dict:
    return json.loads((settings.incoming_day_dir(DATE) / FETCH_MARKER).read_text(encoding="utf-8"))


def test_fetches_both_banks_and_writes_marker():
    with TempProject() as project:
        expected = project.add_images("PBB", 7) | project.add_images("PIBB", 4)
        (project.images_dir("PBB") / "notes.txt").write_text("not a cheque")
        (project.images_dir("PBB") / "subdir").mkdir()
        project.add_trigger("PBB"), project.add_trigger("PIBB")
        result = fetch.fetch_day(DATE)
        assert result.code == 0, result
        assert _marker()["files"] == expected
        assert _marker()["banks"]["PBB"]["listed"] == 7
        items = inventory_day(settings.incoming_day_dir(DATE), settings.banks)
        assert [item.cheque_id for item in items] == sorted(expected)
        assert not list(settings.incoming_day_dir(DATE).rglob("*.part"))


def test_rerun_after_marker_does_nothing():
    with TempProject() as project:
        project.add_images("PBB", 2), project.add_images("PIBB", 2)
        project.add_trigger("PBB"), project.add_trigger("PIBB")
        assert fetch.fetch_day(DATE).code == 0
        opened = []
        assert fetch.fetch_day(DATE, open_source=lambda: opened.append(1) or fetch.LocalSource()).code == 0
        assert opened == []


def test_waits_for_both_triggers_then_gives_up():
    with TempProject() as project:
        project.add_images("PBB", 2), project.add_images("PIBB", 2)
        project.add_trigger("PBB")  # PIBB never arrives
        clock = [datetime(2026, 11, 19, 18, 0)]
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += timedelta(seconds=seconds)

        result = fetch.fetch_day(DATE, now=lambda: clock[0], sleep=sleep)
        assert result.code == 1
        assert len(sleeps) == 12 and set(sleeps) == {600}  # 18:00 .. 20:00 every 10 min
        assert not (settings.incoming_day_dir(DATE) / FETCH_MARKER).exists()


def test_trigger_arriving_while_waiting():
    with TempProject() as project:
        project.add_images("PBB", 2), project.add_images("PIBB", 3)
        project.add_trigger("PBB")
        clock = [datetime(2026, 11, 19, 18, 0)]

        def sleep(seconds):
            clock[0] += timedelta(seconds=seconds)
            project.add_trigger("PIBB")

        assert fetch.fetch_day(DATE, now=lambda: clock[0], sleep=sleep).code == 0
        assert _marker()["total_files"] == 5


def test_started_after_give_up_checks_once():
    with TempProject() as project:
        project.add_images("PBB", 1)
        sleeps = []
        result = fetch.fetch_day(DATE, now=lambda: datetime(2026, 11, 19, 21, 0), sleep=sleeps.append)
        assert result.code == 1 and sleeps == []


def test_copies_only_missing_or_wrong_size_and_removes_strays():
    with TempProject() as project:
        expected = project.add_images("PBB", 5) | project.add_images("PIBB", 1)
        project.add_trigger("PBB"), project.add_trigger("PIBB")
        local = settings.incoming_day_dir(DATE) / "PBB"
        local.mkdir(parents=True)
        names = sorted(path.name for path in project.images_dir("PBB").iterdir())
        (local / names[0]).write_bytes((project.images_dir("PBB") / names[0]).read_bytes())  # good copy
        (local / names[1]).write_bytes(b"short")                                            # wrong size
        (local / "OLD000.IMG").write_bytes(b"gone from source")                             # stray
        (local / f"{names[2]}.part").write_bytes(b"half")                                   # crash leftover
        copied = []
        real_download = fetch.LocalSource.download

        def counting_download(self, remote, path):
            copied.append(Path(remote).relative_to(project.source).as_posix())
            real_download(self, remote, path)

        with mock.patch.object(fetch.LocalSource, "download", counting_download):
            assert fetch.fetch_day(DATE).code == 0
        pbb = f"PBB/{DATE}/GWC/Images"
        assert f"{pbb}/{names[0]}" not in copied and f"{pbb}/{names[1]}" in copied and len(copied) == 5
        assert not (local / "OLD000.IMG").exists() and not list(local.glob("*.part"))
        assert _marker()["files"] == expected


def test_failed_download_retries_then_leaves_no_marker_and_rerun_finishes():
    with TempProject() as project:
        project.add_images("PBB", 4), project.add_images("PIBB", 2)
        project.add_trigger("PBB"), project.add_trigger("PIBB")
        bad = sorted(path.name for path in project.images_dir("PBB").iterdir())[1]
        attempts = []
        real_download = fetch.LocalSource.download

        def flaky(self, remote, path):
            if Path(remote) == project.images_dir("PBB") / bad:
                attempts.append(1)
                path.write_bytes(b"x")  # a partial file must not survive
                raise OSError("connection reset")
            real_download(self, remote, path)

        with mock.patch.object(fetch.LocalSource, "download", flaky), mock.patch.object(fetch.time, "sleep"):
            result = fetch.fetch_day(DATE)
        assert result.code == 1 and result.failed == [f"PBB/{bad}"]
        assert len(attempts) == settings.fetch_file_attempts
        assert not (settings.incoming_day_dir(DATE) / FETCH_MARKER).exists()
        assert not list(settings.incoming_day_dir(DATE).rglob("*.part"))
        assert fetch.fetch_day(DATE).code == 0
        assert _marker()["total_files"] == 6


def test_missing_images_folder_with_trigger_counts_zero():
    with TempProject() as project:
        project.add_images("PBB", 3)
        project.add_trigger("PBB"), project.add_trigger("PIBB")  # PIBB: trigger but no Images folder
        assert fetch.fetch_day(DATE).code == 0
        assert _marker()["banks"]["PIBB"]["listed"] == 0 and _marker()["total_files"] == 3


def test_trigger_pattern_required_unless_skipped():
    with TempProject() as project:
        project.add_images("PBB", 1), project.add_images("PIBB", 1)
        settings.source_trigger_pattern = ""
        try:
            fetch.fetch_day(DATE)
            raise AssertionError("expected ConfigError")
        except fetch.ConfigError:
            pass
        assert fetch.fetch_day(DATE, skip_trigger=True).code == 0


def test_force_fetches_again():
    with TempProject() as project:
        project.add_images("PBB", 2), project.add_images("PIBB", 1)
        project.add_trigger("PBB"), project.add_trigger("PIBB")
        assert fetch.fetch_day(DATE).code == 0
        project.add_images("PIBB", 3)  # source grew after the first fetch
        assert fetch.fetch_day(DATE).code == 0 and _marker()["total_files"] == 3
        assert fetch.fetch_day(DATE, force=True).code == 0 and _marker()["total_files"] == 5


def test_sftp_source_methods():
    regular, folder = stat.S_IFREG | 0o644, stat.S_IFDIR | 0o755
    entries = [SimpleNamespace(filename="A1.IMG", st_size=10, st_mode=regular),
               SimpleNamespace(filename="b2.img", st_size=20, st_mode=regular),
               SimpleNamespace(filename="DONE.TRG", st_size=1, st_mode=regular),
               SimpleNamespace(filename="X.IMG", st_size=0, st_mode=folder)]
    calls = []

    class FakeSftp:
        def listdir_attr(self, path):
            if path == "/missing":
                raise FileNotFoundError(path)
            return entries

        def stat(self, path):
            if path == "/none":
                raise FileNotFoundError(path)
            return SimpleNamespace(st_mode=regular)

        def get(self, remote, local):
            calls.append((remote, local))

    source = object.__new__(fetch.SftpSource)
    source._sftp = FakeSftp()
    assert source.list_images("/Centric/PBB") == {"A1.IMG": 10, "b2.img": 20}
    assert source.list_images("/missing") is None
    assert source.exists("/Centric/PBB/x/DONE.TRG") and not source.exists("/none")
    source.download("/r/A1.IMG", Path("local.part"))
    assert calls == [("/r/A1.IMG", "local.part")]


def test_sftp_settings_required():
    with TempProject():
        settings.source_transport, settings.sftp_host = "sftp", ""
        try:
            fetch.SftpSource()
            raise AssertionError("expected ConfigError")
        except fetch.ConfigError as error:
            assert "SFTP_HOST" in str(error)
        settings.sftp_host, settings.sftp_username, settings.sftp_password, settings.sftp_key_path = "h", "u", "", ""
        try:
            fetch.SftpSource()
            raise AssertionError("expected ConfigError")
        except fetch.ConfigError as error:
            assert "SFTP_PASSWORD or SFTP_KEY_PATH" in str(error)
        settings.sftp_host = settings.sftp_username = ""


if __name__ == "__main__":
    run([value for name, value in list(globals().items()) if name.startswith("test_")])
