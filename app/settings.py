from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    root: Path = Path(__file__).resolve().parents[1]
    max_upload_mb: int = 15
    yolo_model_path: str = "models/signature-yolov8.torchscript"
    hf_yolo_repo_id: str = ""
    hf_yolo_filename: str = ""
    signature_imgsz: int = 1280
    signature_iou: float = 0.70
    signature_logging_floor: float = 0.01
    # Deliberately separate from PreprocessOptions.max_long_edge (2200) --
    # that value is what date_extraction.py's geometry is calibrated
    # against, so it can't move. Signature detection has no such
    # calibration dependency and benefits from more resolution (matches
    # script.py's proven max_dim=3000), so it gets its own cap instead of
    # sharing the OCR/date path's.
    signature_max_long_edge: int = 3000
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
    digit_model_backend: str = "transformers" #transformers or torch_pickle
    digit_model_path: str = "models/farleyknight-vit-mnist" #models/farleyknight-vit-mnist or models/developerPratik-mnist-cnn/best_model.pth
    digit_count: int = 6
    # Fixed (not Otsu-adaptive) grayscale cutoff below which a pixel counts
    # as ink. Real ink is near-black; gray shading/gradient bleed elsewhere
    # in the crop is lighter gray. Otsu picks its threshold dynamically per
    # image and can be pulled toward including that shading as "foreground"
    # -- a fixed cutoff isn't swayed by what else happens to be in the crop.
    digit_ink_threshold: int = 140

    @property
    def batches_dir(self) -> Path: return self.root / "data" / "batches"
    @property
    def model_path(self) -> Path: return self.root / self.yolo_model_path


settings = Settings()