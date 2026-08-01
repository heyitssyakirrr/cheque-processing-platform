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
    # Measurements in the UI are expressed against this reference width, so
    # date geometry remains stable when preprocessing rescales a cheque.
    date_width_reference_edge: int = 2200
    # Keep common OCR confusions; normalized matching removes spaces/punctuation.
    date_label_aliases: tuple[str, ...] = (
        "TARIKH", "DATE", "DATF", "D4TE", "TARlKH", "TARlK",
        "TARKH", "TARK", "DAE", "BDATE", "WN",
    )

    @property
    def runs_dir(self) -> Path: return self.root / "data" / "runs"
    @property
    def batches_dir(self) -> Path: return self.root / "data" / "batches"
    @property
    def model_path(self) -> Path: return self.root / self.yolo_model_path


settings = Settings()