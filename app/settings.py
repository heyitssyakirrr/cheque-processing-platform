import math
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Absolute, so the .env is found regardless of the launch directory.
    model_config = SettingsConfigDict(env_file=_ROOT / ".env", extra="ignore")
    root: Path = _ROOT
    max_upload_mb: int = 15
    yolo_model_path: str = "models/yolov8s.pt"
    hf_yolo_repo_id: str = ""
    hf_yolo_filename: str = ""
    # Absolute (or relative-to-root) path to the shared FileReader models/
    # folder, so the PaddleOCR weights live in one place instead of being
    # duplicated into every project that needs them. Set in .env.
    ocr_models_dir: str = "../FileReader/models"
    # Source scans are only ~700x350, so the signature crop is roughly
    # 490x124. 640 already upscales that; 1280 was ~4x the compute for
    # pixels that carry no extra information.
    signature_imgsz: int = 640
    signature_iou: float = 0.70
    signature_logging_floor: float = 0.01
    # Deliberately separate from PreprocessOptions.max_long_edge (2200) --
    # that value is what date_extraction.py's geometry is calibrated
    # against, so it can't move. Signature detection has no such
    # calibration dependency and benefits from more resolution (matches
    # script.py's proven max_dim=3000), so it gets its own cap instead of
    # sharing the OCR/date path's.
    # Ceiling, not a target: a ~700px scan gains nothing from being
    # interpolated up to it before detection.
    signature_max_long_edge: int = 1600
    # Fixed-fraction crop window (of the resized signature image) where a
    # signature reliably sits on this cheque template -- generous on
    # purpose, tune these three numbers based on what you see once
    # tested, rather than the model/resolution settings above.
    signature_zone_left_fraction: float = 0.30
    signature_zone_top_fraction: float = 0.55
    signature_zone_bottom_fraction: float = 0.92
    # Measurements in the UI are expressed against this reference width, so
    # date geometry remains stable when preprocessing rescales a cheque.
    date_width_reference_edge: int = 2200
    # TARIKH and "日DATE" are two separate, stacked printed lines (TARIKH on
    # top, DATE beneath it) -- not one label. Kept as two alias groups so the
    # crop geometry can key off *which* line was actually found (see
    # date_extraction._find_label_lines / _vertical_bounds_from_labels)
    # instead of treating either match the same way.
    date_label_aliases_tarikh: tuple[str, ...] = (
        "TARIKH", "TARlKH", "TARlK", "TARKH", "TARK",
    )
    date_label_aliases_date: tuple[str, ...] = (
        "DATE", "DATF", "D4TE", "DAE", "BDATE",
    )
    # WN is a known but weak corruption of either line; only trusted in the
    # normal upper-right label area (see the weak_wn check in
    # date_extraction._score_candidates).
    date_label_aliases_ambiguous: tuple[str, ...] = ("WN",)
    # Vertical crop bound = TARIKH's own top edge to DATE's own bottom edge
    # (see _vertical_bounds_from_labels docstring for why). If only one line
    # is found, the other's edge is estimated as this many typical
    # single-line-heights away, plus the calibrated gap below.
    date_label_line_gap_ratio: float = 0.2
    # Small safety margin added above/below that measured span, since
    # handwritten digits can slightly overshoot the printed label's own
    # height. Deliberately much smaller than the old fixed multipliers --
    # those existed to compensate for using only one, possibly-wrong edge;
    # with both edges measured directly this only needs to cover natural
    # handwriting variance.
    date_field_vertical_safety_pad_ratio: float = 0.15
    # Extra padding added only below DATE's own bottom edge, on top of the
    # symmetric safety pad above. Handwritten descenders and closed loops
    # (a looped '6' or '2') extend further down than a printed line's own
    # bottom edge predicts -- this is asymmetric on purpose; the top edge
    # needs no equivalent since TARIKH's top edge has no handwriting above it.
    date_field_bottom_extra_pad_ratio: float = 0.35
    # Horizontal width, in multiples of the document's typical single-line
    # text height (a stable, format-invariant reference -- see
    # `_typical_line_height`), not the anchor's own box height.
    date_field_width_typical_heights: float = 17.0
    # Hard ceiling on the crop, independent of anything OCR reports, as a
    # last line of defense against any future anchor-measurement error.
    # Expressed as a fraction of the (already format-normalised) page size.
    date_field_max_width_fraction: float = 0.55
    date_field_max_height_fraction: float = 0.32
    # ViT-base costs ~17.5 GFLOPs per digit -- ~105 GFLOPs per cheque to read
    # six MNIST-style digits, which dominated the date stage. The small CNN
    # is built for exactly this input.
    digit_model_backend: str = "torch_pickle" #transformers or torch_pickle
    digit_model_path: str = "models/developerPratik-mnist-cnn/best_model.pth" #models/farleyknight-vit-mnist or models/developerPratik-mnist-cnn/best_model.pth
    digit_count: int = 6
    # Fixed (not Otsu-adaptive) grayscale cutoff below which a pixel counts
    # as ink. Real ink is near-black; gray shading/gradient bleed elsewhere
    # in the crop is lighter gray. Otsu picks its threshold dynamically per
    # image and can be pulled toward including that shading as "foreground"
    # -- a fixed cutoff isn't swayed by what else happens to be in the crop.
    digit_ink_threshold: int = 140
    # CSV of known-correct answers for cheques you run repeatedly, so
    # accuracy scoring doesn't require re-pasting ground truth every batch.
    # Columns: filename, signature_count, date (DDMMYY or "no date").
    ground_truth_path: str = "data/ground_truth.csv"
    # Folder scanned for cheque images, and the single results file.
    upload_dir_name: str = "Upload"
    output_csv_name: str = "output.csv"
    # These replace the sliders the removed web form used to supply.
    preprocess_max_long_edge: int = 3000
    preprocess_contrast: float = 1.15
    # Non-local-means denoising costs roughly a second per cheque and only
    # feeds the OCR/date path. Off by default; turn on if date accuracy needs it.
    preprocess_denoise: bool = False
    preprocess_deskew: bool = True
    signature_decision_threshold: float = 0.30
    # Per-cheque annotated images, crops, digit slices, CSVs and result.json.
    # Roughly 20 files per cheque, so this is left off for production volume
    # and turned on when investigating a specific batch.
    save_artifacts: bool = False
    # Region OCR is restricted to when locating the TARIKH/DATE labels.
    # Deliberately wider than _template_date_zone (0.60-0.99 x, 0.10-0.38 y)
    # so the printed labels and enough neighbouring lines stay in frame for
    # _typical_line_height to remain representative.
    ocr_zone_left_fraction: float = 0.40
    ocr_zone_top_fraction: float = 0.00
    ocr_zone_bottom_fraction: float = 0.45
    # Debug aid only. Skips OCR and takes the date field from
    # _template_date_zone's fixed fractions, which is wrong for real input:
    # cheques arrive from several banks at varying sizes, so the date is not
    # at a fixed offset. Finding the TARIKH/DATE labels is what makes the
    # crop portable, so leave this off in production.
    date_use_template_zone: bool = False
    # Worker process count for run.py.
    # - >0: fixed worker count.
    # - 0: auto-size from the logical-core budget below.
    worker_count: int = 0
    # Shared-server tuning for worker_count=0.
    # Example: 128 logical cores * 0.40 target - 8 reserved = 43 workers.
    worker_utilization_target: float = 0.40
    reserved_logical_cores: int = 8
    # Optional hard ceiling after auto-sizing. 0 disables this cap.
    max_worker_cap: int = 0
    # Threads each worker's native libraries may use. 0 leaves them
    # unrestricted, which means every worker starts its own 16-thread OpenMP
    # pool -- 25 workers then fight over 25 cores with 400 spinning threads.
    # worker_count * threads_per_worker should stay at or under the core count.
    threads_per_worker: int = 1

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

    @property
    def upload_dir(self) -> Path: return self.root / self.upload_dir_name
    @property
    def output_csv(self) -> Path: return self.root / self.output_csv_name

    @property
    def batches_dir(self) -> Path: return self.root / "data" / "batches"
    @property
    def model_path(self) -> Path: return self.root / self.yolo_model_path

    @model_validator(mode="after")
    def _absolutise_digit_model_path(self) -> "Settings":
        """transformers treats a relative path that misses as a Hub repo id and
        downloads it, so this must never stay relative to the launch directory."""
        path = Path(self.digit_model_path)
        if not path.is_absolute():
            self.digit_model_path = str((self.root / path).resolve())
        return self

    @property
    def _ocr_models_base(self) -> Path:
        p = Path(self.ocr_models_dir)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def ocr_det_dir(self) -> Path: return self._ocr_models_base / "en_PP-OCRv3_det_infer"
    @property
    def ocr_rec_dir(self) -> Path: return self._ocr_models_base / "en_PP-OCRv3_rec_infer"
    @property
    def ocr_cls_dir(self) -> Path: return self._ocr_models_base / "ch_ppocr_mobile_v2.0_cls_infer"


settings = Settings()