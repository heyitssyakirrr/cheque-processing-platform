"""Job 1's work: copy one day's .IMG files from the source server to
data/incoming/<YYYYMMDD>/<bank>/, then write _FETCH_COMPLETE.json last.

The source is only read: nothing there is ever deleted, moved or written.

    1. wait until every bank's trigger file exists (poll until FETCH_GIVE_UP_AT)
    2. list each bank's Images folder once (names + sizes in one call)
    3. download only files missing locally or with a different size, in
       parallel (FETCH_WORKERS connections), each to <name>.part, size
       checked, then renamed
    4. write _FETCH_COMPLETE.json with every file and its size

A rerun copies only what is still missing. Once the marker exists, the day
is not fetched again unless forced.
"""

from __future__ import annotations

import logging
import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from Cheque_Processing.app.ingestion import FETCH_MARKER, batch_lock, is_image_name, write_json_atomic
from Cheque_Processing.app.settings import settings

logger = logging.getLogger("job1")


class ConfigError(RuntimeError):
    """A setting is missing or wrong; retrying will not help."""


class Source(Protocol):
    def exists(self, path: str) -> bool: ...
    def list_images(self, folder: str) -> dict[str, int] | None:
        """{name: size} of the .IMG files in folder, or None if the folder does not exist."""
    def download(self, remote: str, local: Path) -> None: ...
    def close(self) -> None: ...


class LocalSource:
    """The source tree is a folder on this machine (mounted share, or a fake tree for tests)."""

    def exists(self, path: str) -> bool:
        return Path(path).is_file()

    def list_images(self, folder: str) -> dict[str, int] | None:
        base = Path(folder)
        if not base.is_dir():
            return None
        return {entry.name: entry.stat().st_size for entry in os.scandir(base)
                if entry.is_file() and is_image_name(entry.name)}

    def download(self, remote: str, local: Path) -> None:
        with open(remote, "rb") as src, local.open("wb") as dst:
            while block := src.read(1024 * 1024):
                dst.write(block)

    def close(self) -> None:
        pass


class SftpSource:
    """One SSH connection + SFTP session (paramiko). Each fetch worker opens its own."""

    def __init__(self) -> None:
        import paramiko

        if not settings.sftp_host or not settings.sftp_username:
            raise ConfigError("Set SFTP_HOST and SFTP_USERNAME in .env")
        if not settings.sftp_password and not settings.sftp_key_path:
            raise ConfigError("Set SFTP_PASSWORD or SFTP_KEY_PATH in .env")
        self._client = paramiko.SSHClient()
        self._client.load_system_host_keys()
        if settings.sftp_known_hosts:
            self._client.load_host_keys(os.path.expanduser(settings.sftp_known_hosts))
        # Never trust an unknown server automatically: its key must be in known_hosts.
        self._client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            self._client.connect(
                settings.sftp_host, port=settings.sftp_port, username=settings.sftp_username,
                password=settings.sftp_password or None,
                key_filename=os.path.expanduser(settings.sftp_key_path) if settings.sftp_key_path else None,
                look_for_keys=False, allow_agent=False, timeout=settings.sftp_timeout_seconds,
                banner_timeout=settings.sftp_timeout_seconds, auth_timeout=settings.sftp_timeout_seconds,
            )
        except paramiko.SSHException as error:
            self._client.close()
            if "not found in known_hosts" in str(error):
                raise ConfigError(
                    f"{settings.sftp_host} is not in known_hosts. Check its key fingerprint with the "
                    f"server's team, then add it: ssh-keyscan -p {settings.sftp_port} {settings.sftp_host} "
                    ">> ~/.ssh/known_hosts"
                ) from error
            if isinstance(error, paramiko.AuthenticationException):
                raise ConfigError(f"SFTP login to {settings.sftp_host} refused: {error}") from error
            raise
        self._sftp = self._client.open_sftp()
        self._sftp.get_channel().settimeout(settings.sftp_timeout_seconds * 4)

    def exists(self, path: str) -> bool:
        try:
            return stat.S_ISREG(self._sftp.stat(path).st_mode or 0)
        except FileNotFoundError:
            return False

    def list_images(self, folder: str) -> dict[str, int] | None:
        try:
            entries = self._sftp.listdir_attr(folder)
        except FileNotFoundError:
            return None
        return {entry.filename: entry.st_size for entry in entries
                if stat.S_ISREG(entry.st_mode or 0) and is_image_name(entry.filename)}

    def download(self, remote: str, local: Path) -> None:
        self._sftp.get(remote, str(local))

    def close(self) -> None:
        self._client.close()


def open_configured_source() -> Source:
    return LocalSource() if settings.source_transport == "local" else SftpSource()


def _path(pattern: str, bank: str, batch_id: str) -> str:
    return pattern.format(root=settings.source_root.rstrip("/\\"), bank=bank, date=batch_id)


@dataclass
class FetchResult:
    code: int                     # 0 complete, 1 not ready / some files failed, 2 configuration problem
    banks: dict[str, dict[str, int]] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)


def _wait_for_triggers(open_source: Callable[[], Source], batch_id: str, now: Callable[[], datetime],
                       sleep: Callable[[float], None]) -> bool:
    hour, minute = map(int, settings.fetch_give_up_at.split(":"))
    give_up = now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    while True:
        source = open_source()
        try:
            missing = [bank for bank in settings.banks
                       if not source.exists(_path(settings.source_trigger_pattern, bank, batch_id))]
        finally:
            source.close()
        if not missing:
            logger.info("Trigger files found for %s", ", ".join(settings.banks))
            return True
        if now() >= give_up:
            logger.error("No trigger file for %s by %s; giving up for now (rerun job 1 later)",
                         ", ".join(missing), settings.fetch_give_up_at)
            return False
        logger.info("Waiting for the trigger file of %s; checking again in %d min",
                    ", ".join(missing), settings.fetch_poll_minutes)
        sleep(settings.fetch_poll_minutes * 60)


def fetch_day(batch_id: str, *, force: bool = False, skip_trigger: bool = False,
              open_source: Callable[[], Source] = open_configured_source,
              now: Callable[[], datetime] = datetime.now, sleep: Callable[[float], None] = time.sleep) -> FetchResult:
    day_dir = settings.incoming_day_dir(batch_id)
    marker = day_dir / FETCH_MARKER
    if marker.is_file() and not force:
        logger.info("%s is already fetched (%s exists); use --force to fetch it again", batch_id, marker)
        return FetchResult(0)
    if not skip_trigger and not settings.source_trigger_pattern:
        raise ConfigError("SOURCE_TRIGGER_PATTERN is not set in .env (the file that marks a bank's day "
                          "complete); or run by hand with --skip-trigger")

    day_dir.mkdir(parents=True, exist_ok=True)
    with batch_lock(day_dir / ".fetch.lock"):
        marker.unlink(missing_ok=True)
        if not skip_trigger and not _wait_for_triggers(open_source, batch_id, now, sleep):
            return FetchResult(1)

        # 1. List the source once per bank.
        source = open_source()
        try:
            listings = {}
            for bank in settings.banks:
                folder = _path(settings.source_images_pattern, bank, batch_id)
                listing = source.list_images(folder)
                if listing is None:
                    logger.warning("%s: %s does not exist; counting it as 0 cheques", bank, folder)
                    listing = {}
                listings[bank] = listing
        finally:
            source.close()

        # 2. Compare with what is already here; drop leftovers the source no longer lists.
        todo: list[tuple[str, str, Path, int]] = []
        result = FetchResult(0)
        for bank, listing in listings.items():
            local_dir = day_dir / bank
            local_dir.mkdir(parents=True, exist_ok=True)
            for leftover in local_dir.glob("*.part"):
                leftover.unlink()
            local = {entry.name: entry.stat().st_size for entry in os.scandir(local_dir)
                     if entry.is_file() and is_image_name(entry.name)}
            for name in sorted(set(local) - set(listing)):
                logger.warning("%s/%s is no longer on the source; removed the local copy", bank, name)
                (local_dir / name).unlink()
            need = [name for name, size in listing.items() if local.get(name) != size]
            todo += [(bank, name, local_dir / name, listing[name]) for name in sorted(need)]
            result.banks[bank] = {"listed": len(listing), "already_here": len(listing) - len(need),
                                  "to_copy": len(need), "bytes": sum(listing.values())}
            logger.info("%s: %d file(s) on the source (%.1f MB), %d already here, %d to copy",
                        bank, len(listing), sum(listing.values()) / 1e6, len(listing) - len(need), len(need))

        # 3. Download in parallel; each worker thread keeps its own connection.
        if todo:
            started = time.perf_counter()
            result.failed = _download_all(todo, open_source, batch_id)
            copied_mb = sum(size for *_, size in todo) / 1e6
            logger.info("Copied %d file(s), %.1f MB in %.0fs; %d failed",
                        len(todo) - len(result.failed), copied_mb, time.perf_counter() - started, len(result.failed))
        if result.failed:
            for cheque_id in result.failed[:20]:
                logger.error("Not copied: %s", cheque_id)
            logger.error("%d file(s) not copied; %s not written. Rerun job 1 to copy the rest.",
                         len(result.failed), FETCH_MARKER)
            result.code = 1
            return result

        # 4. Check everything on disk, then write the marker last.
        files = {}
        for bank, listing in listings.items():
            for name, size in listing.items():
                actual = (day_dir / bank / name).stat().st_size
                if actual != size:
                    raise RuntimeError(f"{bank}/{name} is {actual} bytes after copying, the source has {size}")
                files[f"{bank}/{name}"] = size
        write_json_atomic(marker, {
            "batch_id": batch_id, "fetched_at": now().isoformat(timespec="seconds"),
            "source": f"{settings.source_transport}:{settings.sftp_host or ''}{settings.source_root}",
            "banks": result.banks, "total_files": len(files), "files": files,
        })
        logger.info("%s written: %d cheque file(s) ready for job 2", marker, len(files))
        return result


def _download_all(todo: list[tuple[str, str, Path, int]], open_source: Callable[[], Source], batch_id: str) -> list[str]:
    local_state = threading.local()
    opened: list[Source] = []
    lock = threading.Lock()

    def source() -> Source:
        if getattr(local_state, "source", None) is None:
            local_state.source = open_source()
            with lock:
                opened.append(local_state.source)
        return local_state.source

    def drop_source() -> None:
        current = getattr(local_state, "source", None)
        local_state.source = None
        if current is not None:
            try:
                current.close()
            except Exception:
                pass

    def copy(item: tuple[str, str, Path, int]) -> str | None:
        bank, name, local, size = item
        remote = f"{_path(settings.source_images_pattern, bank, batch_id)}/{name}"
        partial = local.with_name(local.name + ".part")
        for attempt in range(1, settings.fetch_file_attempts + 1):
            try:
                source().download(remote, partial)
                got = partial.stat().st_size
                if got != size:
                    raise IOError(f"got {got} bytes, expected {size}")
                os.replace(partial, local)
                return None
            except ConfigError:
                raise
            except Exception as error:
                partial.unlink(missing_ok=True)
                logger.warning("%s/%s attempt %d/%d failed: %s", bank, name, attempt,
                               settings.fetch_file_attempts, error)
                drop_source()  # reconnect on the next try
                if attempt < settings.fetch_file_attempts:
                    time.sleep(min(2 ** attempt, 30))
        return f"{bank}/{name}"

    try:
        with ThreadPoolExecutor(max_workers=min(settings.fetch_workers, len(todo))) as pool:
            failed = [cheque_id for cheque_id in pool.map(copy, todo) if cheque_id]
    finally:
        for item in opened:
            try:
                item.close()
            except Exception:
                pass
    return failed
