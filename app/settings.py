"""Application configuration.

One flat, validated settings object, loaded from `.env`. 
Fields are grouped below by which part of the pipeline they configure:

    1. Paths & environment          -- where things live on disk
    2. Signature detection (YOLO)   -- app/services/signature_detection.py
    3. Date field location (OCR)    -- app/services/date_extraction.py
    4. Digit classifier             -- app/services/digit_classifier.py
    5. Preprocessing                -- app/services/preprocessing.py
    6. Debug output                 -- per-cheque artifacts
    7. Job 1: fetch from the source -- jobs/job1_fetch_cheques.py, app/fetch.py
    8. Job 2: batch processing      -- jobs/job2_process_cheques.py, app/ingestion.py, app/manifest.py
    9. Worker process sizing        -- job 2's ProcessPoolExecutor
   10. Retention                    -- app/retention.py

Every field can be overridden by an environment variable of the same name
(case-insensitive), or by editing `.env`. 
"""

import math
import re
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Absolute, so the .env is found regardless of the launch directory.
    model_config = SettingsConfigDict(env_file=_ROOT / ".env", extra="ignore")

    # ------------------------------------------------------------------
    # 1. Paths & environment
    # ------------------------------------------------------------------
    root: Path = _ROOT

    # ------------------------------------------------------------------
    # 2. Signature detection (YOLO)
    # ------------------------------------------------------------------
    yolo_model_path: str = "models/yolov8s.pt"
    """Local path to the signature-detection weights, relative to `root`."""

    hf_yolo_repo_id: str = ""
    hf_yolo_filename: str = ""
    """Optional Hugging Face Hub source to fetch the YOLO weights from if
    `yolo_model_path` doesn't exist locally. Leave both blank to require the
    local file instead."""

    signature_imgsz: int = 640
    """YOLO inference resolution (its `imgsz` argument). Source scans are
    only ~700x350, so the signature crop is roughly 490x124 -- 640 already
    upscales that; 1280 was ~4x the compute for pixels that carry no extra
    information."""

    signature_iou: float = 0.70
    """YOLO's non-max-suppression IoU threshold: how much two candidate
    signature boxes may overlap before the weaker one is discarded."""

    signature_logging_floor: float = 0.01
    """Minimum confidence score YOLO will even report to us for logging /
    debugging, independent of `signature_decision_threshold` below (which
    decides pass/fail). Effectively "don't bother mentioning anything below
    1% confidence"."""

    signature_max_long_edge: int = 1600
    """Resize cap applied only before signature detection. Deliberately
    separate from `preprocess_max_long_edge` (2200) -- that value is what
    date_extraction.py's geometry is calibrated against, so it can't move.
    Signature detection has no such calibration dependency and benefits from
    more resolution (matches the earlier standalone test's proven max_dim=3000), so it gets
    its own cap. This is a ceiling, not a target: a ~700px scan gains
    nothing from being interpolated up to it before detection."""

    signature_zone_left_fraction: float = 0.30
    signature_zone_top_fraction: float = 0.55
    signature_zone_bottom_fraction: float = 0.92
    """Fixed-fraction crop window (of the resized signature image) where a
    signature reliably sits on this cheque template -- generous on purpose.
    Tune these three based on what you see once tested, rather than the
    model/resolution settings above."""

    signature_decision_threshold: float = 0.30
    """Confidence above which a detected box counts as "signature present"
    in the final result. (Contrast with `signature_logging_floor`, which
    only controls what gets logged.)"""

    # ------------------------------------------------------------------
    # 3. Date field location (OCR)
    # ------------------------------------------------------------------
    date_width_reference_edge: int = 2200
    """Reference width that all date-field measurements below are expressed
    against, so date geometry remains stable when preprocessing rescales a
    cheque to a different size."""

    date_label_aliases_tarikh: tuple[str, ...] = (
        "TARIKH", "TARlKH", "TARlK", "TARKH", "TARK",
    )
    date_label_aliases_date: tuple[str, ...] = (
        "DATE", "DATF", "D4TE", "DAE", "BDATE",
    )
    """TARIKH and "日DATE" are two separate, stacked printed lines (TARIKH on
    top, DATE beneath it) -- not one label. Kept as two alias groups (common
    OCR misreads of each) so the crop geometry can key off *which* line was
    actually found -- see `date_extraction._find_label_lines` /
    `_vertical_bounds_from_labels` -- instead of treating either match the
    same way."""

    date_label_aliases_ambiguous: tuple[str, ...] = ("WN",)
    """WN is a known but weak corruption of either line; only trusted in the
    normal upper-right label area (see the `weak_wn` check in
    `date_extraction._score_candidates`)."""

    # Date-box geometry. Unit = printed width of the word TARIKH (about 108px
    # on a 2010px-wide image). The printed label scales with the cheque, so
    # these hold at any image size. Measured on Public Bank cheques: the six
    # boxes start at TARIKH's right edge and are ~0.78 units apart; their top
    # is ~0.2 units above TARIKH's top; the printed "D D M M Y Y" under them
    # starts ~0.8 units below TARIKH's top. Check any change with
    # diagnostics/check_date_crop.py.
    date_row_start_units: float = 0.04
    """Start this far right of TARIKH's right edge: the italic H leans into
    the gap, and any sliver of it would be read as a stroke. The first box's
    digit sits well inside the box, so nothing is lost."""

    date_row_length_units: float = 4.75
    """Search area: from TARIKH's right edge to the end of the sixth box."""

    date_row_top_pad_units: float = 0.3
    """Crop top: this far above the top of TARIKH."""

    date_row_bottom_units: float = 0.78
    """Crop bottom: this far below the top of TARIKH (just above "D D M M Y Y")."""

    date_field_trim_to_ink: bool = True
    """Pull the right edge in to the last handwritten stroke plus
    `date_row_ink_margin_units`, so the hatched pattern at the cheque's right
    edge stays out. Strokes are pixels darker than `digit_ink_threshold`."""

    date_row_ink_margin_units: float = 0.25
    """Space kept right of the last stroke."""

    date_row_min_length_units: float = 4.45
    """The trim never cuts the crop shorter than this (inside the sixth box),
    so a faint last digit is kept."""

    date_row_blank_length_units: float = 4.6
    """Crop length when no handwriting is found (blank date boxes)."""

    date_label_unit_page_fraction: float = 0.0535
    """Expected TARIKH width as a fraction of the image width. Only a sanity
    check: a measured width outside `date_label_unit_tolerance` x this (OCR
    misread the label) is replaced by it."""

    date_label_unit_tolerance: tuple[float, float] = (0.7, 1.4)

    date_label_date_width_units: float = 1.23
    date_label_date_right_gap_units: float = 0.05
    date_label_tarikh_height_units: float = 0.3
    """When only DATE is found: its box (which includes the printed Chinese
    characters) is this many units wide, ends this far left of TARIKH's right
    edge, and its top is this far below TARIKH's top."""

    ocr_zone_left_fraction: float = 0.40
    ocr_zone_top_fraction: float = 0.00
    ocr_zone_bottom_fraction: float = 0.45
    """Region OCR is restricted to when locating the TARIKH/DATE labels.
    Deliberately wider than `_template_date_zone` (0.60-0.99 x, 0.10-0.38 y)
    so the printed labels stay in frame."""

    date_use_template_zone: bool = False
    """Debug aid only. When true, skips OCR and takes the date field from
    `_template_date_zone`'s fixed fractions -- wrong for real input, since
    cheques arrive from several banks at varying sizes and the date is not
    at a fixed offset. Finding the TARIKH/DATE labels is what makes the crop
    portable, so leave this off in production."""

    # ------------------------------------------------------------------
    # 4. Digit classifier
    # ------------------------------------------------------------------
    digit_model_backend: str = "torch_pickle"
    """Which digit-classifier implementation to load: "transformers" (a
    ViT-base model) or "torch_pickle" (a small CNN). ViT-base costs ~17.5
    GFLOPs per digit -- ~105 GFLOPs per cheque to read six MNIST-style
    digits, which dominated the date stage; the small CNN is built for
    exactly this input and is the default for that reason."""

    digit_model_path: str = "models/developerPratik-mnist-cnn/best_model.pth"
    """Path to the digit-classifier weights matching `digit_model_backend`
    above. Swap both together -- e.g. "models/farleyknight-vit-mnist" pairs
    with backend "transformers"."""

    digit_count: int = 6
    """Expected number of handwritten date digits per cheque (DDMMYY)."""

    digit_ink_threshold: int = 140
    """Fixed (not Otsu-adaptive) grayscale cutoff below which a pixel counts
    as ink. Real ink is near-black; gray shading/gradient bleed elsewhere in
    the crop is lighter gray. Otsu picks its threshold dynamically per image
    and can be pulled toward including that shading as "foreground" -- a
    fixed cutoff isn't swayed by what else happens to be in the crop."""

    # ------------------------------------------------------------------
    # 5. Preprocessing
    # ------------------------------------------------------------------
    preprocess_max_long_edge: int = 3000
    preprocess_contrast: float = 1.15
    preprocess_deskew: bool = True
    """Resize cap, contrast boost and deskew applied before date reading."""

    preprocess_denoise: bool = False
    """Non-local-means denoising costs roughly a second per cheque and only
    feeds the OCR/date path. Off by default; turn on if date accuracy needs
    it."""

    # ------------------------------------------------------------------
    # 6. Debug output
    # ------------------------------------------------------------------
    save_artifacts: bool = False
    """Whether to write per-cheque debug output (annotated images, crops,
    digit slices, CSVs, result.json) under data/batches/<date>/<seq>-<name>/
    -- roughly 20 files per cheque, so ~400,000 files for a 20,000-cheque
    day. Off for daily runs; turn on when investigating a specific batch."""

    # ------------------------------------------------------------------
    # 7. Job 1: fetch the day's .IMG files from the source server
    # ------------------------------------------------------------------
    source_transport: str = "sftp"
    """"sftp" (production: paramiko, settings below) or "local" (the source
    tree is a folder on this machine -- a mounted share, or a fake tree for
    testing on a laptop). Both use the same path patterns."""

    source_root: str = "/Centric"
    """Folder above PBB/ and PIBB/ on the source (an SFTP path, or a local
    folder when source_transport=local)."""

    source_banks: str = "PBB,PIBB"
    """Comma-separated bank folders fetched each day; each is also the first
    part of every cheque ID (see `banks`)."""

    source_images_pattern: str = "{root}/{bank}/{date}/GWC/Images"
    """Where one bank's .IMG files for one day are. {root}=source_root,
    {bank}=one of source_banks, {date}=YYYYMMDD."""

    source_trigger_pattern: str = ""
    """Full path of the file that marks one bank's day as complete, with the
    same placeholders, e.g. "{root}/{bank}/{date}/GWC/<name>". To be
    confirmed; job 1 refuses to run while it is empty (unless
    --skip-trigger is given by hand). The trigger is only read, never
    deleted."""

    sftp_host: str = ""
    sftp_port: int = 22
    sftp_username: str = ""
    sftp_password: str = ""
    """Leave blank to log in with sftp_key_path instead."""

    sftp_key_path: str = ""
    """Private key file (e.g. /home/<you>/.ssh/id_rsa); used when set."""

    sftp_known_hosts: str = ""
    """Extra known_hosts file to trust. ~/.ssh/known_hosts is always read.
    An unknown server is refused, never trusted automatically."""

    sftp_timeout_seconds: int = 30

    fetch_workers: int = 6
    """Parallel SFTP connections. ~20,000 small files a day spend their
    time on per-file round trips, not bandwidth, so a few connections in
    parallel are much faster than one."""

    fetch_file_attempts: int = 3
    """Tries per file before job 1 gives up on it (and exits 1)."""

    fetch_poll_minutes: int = 10
    """How often job 1 checks again while a trigger file is missing."""

    fetch_give_up_at: str = "20:00"
    """Local time (HH:MM) after which job 1 stops waiting for the trigger
    files and exits 1. A run started after this time checks once."""

    incoming_root: str = "data/incoming"
    """Where job 1 puts each day: <incoming_root>/<YYYYMMDD>/<bank>/*.IMG."""

    # ------------------------------------------------------------------
    # 8. Job 2: batch processing
    # ------------------------------------------------------------------
    date_offset_days: int = 0
    """Without --date, both jobs use today minus this many days. To be
    replaced by the batch date your senior confirms."""

    max_attempts: int = 3
    """Total attempts per cheque before it's marked permanently failed --
    this counts attempts, not retries (so the default allows 2 retries
    after an initial failure)."""

    batch_inflight_multiplier: int = 2
    """How many futures to keep queued per worker (`workers *
    batch_inflight_multiplier`), so the runner never holds all ~20,000 tasks
    in memory at once."""

    # ------------------------------------------------------------------
    # 9. Worker process sizing (job 2's ProcessPoolExecutor)
    # ------------------------------------------------------------------
    worker_count: int = 0
    """Fixed worker process count. 0 means auto-size from the logical-core
    budget below instead."""

    worker_utilization_target: float = 0.40
    """Fraction of *usable* logical cores (after reserving
    `reserved_logical_cores`) to spend on workers when `worker_count=0`.
    Example: 128 logical cores * 0.40 target - 8 reserved = 43 workers."""

    reserved_logical_cores: int = 8
    """Logical cores left for the OS / other processes on a shared server
    when auto-sizing workers."""

    max_worker_cap: int = 0
    """Optional hard ceiling applied after auto-sizing. 0 disables the
    cap."""

    threads_per_worker: int = 1
    """Threads each worker's native libraries (OpenMP/MKL/BLAS) may use. 0
    leaves them unrestricted -- meaning every worker starts its own
    16-thread pool, so e.g. 25 workers end up fighting over 25 cores with
    400 spinning threads between them. Keep
    `worker_count * threads_per_worker` at or under the core count."""

    # ------------------------------------------------------------------
    # 10. Retention (run at the end of job 2)
    # ------------------------------------------------------------------
    incoming_retention_days: int = 30
    """Days to keep data/incoming/<date>/ (the fetched .IMG files) after
    that date. Only finished batches are deleted. 0 keeps them forever."""

    batch_retention_days: int = 0
    """Days to keep data/batches/<date>/ (output CSV, failed CSV, manifest,
    any debug artifacts). Only finished batches are deleted. 0 keeps them
    forever."""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _validate_worker_budget(self) -> "Settings":
        if self.worker_count < 0:
            raise ValueError("worker_count must be >= 0")
        if not (0.05 <= self.worker_utilization_target <= 1.0):
            raise ValueError("worker_utilization_target must be between 0.05 and 1.0")
        if self.reserved_logical_cores < 0:
            raise ValueError("reserved_logical_cores must be >= 0")
        if self.max_worker_cap < 0:
            raise ValueError("max_worker_cap must be >= 0")
        return self

    @model_validator(mode="after")
    def _validate_batch_ingestion(self) -> "Settings":
        if self.source_transport not in {"sftp", "local"}:
            raise ValueError("source_transport must be 'sftp' or 'local'")
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", self.fetch_give_up_at):
            raise ValueError("fetch_give_up_at must be HH:MM (24-hour)")
        for name in ("fetch_workers", "fetch_file_attempts", "fetch_poll_minutes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("date_offset_days", "incoming_retention_days", "batch_retention_days"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.batch_retention_days and self.batch_retention_days < self.incoming_retention_days:
            # The batch folder holds the "finished" marker the image cleanup checks.
            raise ValueError("batch_retention_days must be 0 or >= incoming_retention_days")
        if not self.banks or any(not re.fullmatch(r"[A-Za-z0-9_-]+", bank) for bank in self.banks):
            raise ValueError("source_banks must be comma-separated folder names, e.g. PBB,PIBB")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.batch_inflight_multiplier < 1:
            raise ValueError("batch_inflight_multiplier must be >= 1")
        return self

    @model_validator(mode="after")
    def _absolutise_digit_model_path(self) -> "Settings":
        """transformers treats a relative path that misses as a Hub repo id
        and downloads it, so this must never stay relative to the launch
        directory."""
        path = Path(self.digit_model_path)
        if not path.is_absolute():
            self.digit_model_path = str((self.root / path).resolve())
        return self

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        return self.root / self.yolo_model_path

    @property
    def banks(self) -> list[str]:
        return [bank.strip() for bank in self.source_banks.split(",") if bank.strip()]

    @property
    def incoming_dir(self) -> Path:
        value = Path(self.incoming_root)
        return value if value.is_absolute() else self.root / value

    def incoming_day_dir(self, batch_id: str) -> Path:
        return self.incoming_dir / batch_id

    @property
    def batches_dir(self) -> Path:
        return self.root / "data" / "batches"

    def batch_dir(self, batch_id: str) -> Path:
        return self.batches_dir / batch_id

    @property
    def logs_dir(self) -> Path:
        return self.root / "data" / "logs"

    @property
    def _ocr_models_base(self) -> Path:
        p = Path(self.ocr_models_dir)
        return p if p.is_absolute() else (self.root / p).resolve()

    ocr_models_dir: str = "../FileReader/models"
    """Path to the shared FileReader models/ folder, so the PaddleOCR
    weights live in one place instead of being duplicated into every
    project that needs them. Set in .env."""

    @property
    def ocr_det_dir(self) -> Path:
        return self._ocr_models_base / "en_PP-OCRv3_det_infer"

    @property
    def ocr_rec_dir(self) -> Path:
        return self._ocr_models_base / "en_PP-OCRv3_rec_infer"

    @property
    def ocr_cls_dir(self) -> Path:
        return self._ocr_models_base / "ch_ppocr_mobile_v2.0_cls_infer"

    # ------------------------------------------------------------------
    # Worker sizing logic
    # ------------------------------------------------------------------
    def recommend_workers(self, logical_cores: int, task_count: int) -> int:
        """Compute the worker count from either fixed or auto mode."""
        if task_count <= 0:
            return 1

        if self.worker_count > 0:
            requested = self.worker_count
        else:
            usable = max(1, logical_cores - self.reserved_logical_cores)
            requested = max(1, math.floor(usable * self.worker_utilization_target))

        if self.max_worker_cap > 0:
            requested = min(requested, self.max_worker_cap)

        return max(1, min(requested, task_count))


settings = Settings()