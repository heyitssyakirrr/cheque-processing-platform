from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    root: Path = Path(__file__).resolve().parents[1]
    max_upload_mb: int = 15
    yolo_model_path: str = "models/signature-yolov8.torchscript"
    hf_yolo_repo_id: str = ""
    hf_yolo_filename: str = ""
    # Keep common OCR confusions; normalized matching removes spaces/punctuation.
    date_label_aliases: tuple[str, ...] = ("TARIKH", "DATE", "DATF", "D4TE", "TARlKH", "TARlK" )

    @property
    def runs_dir(self) -> Path: return self.root / "data" / "runs"
    @property
    def model_path(self) -> Path: return self.root / self.yolo_model_path


settings = Settings()
