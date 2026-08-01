"""FastAPI web application for batch cheque processing."""

import csv
import re
import uuid
from datetime import datetime
import uvicorn
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from app.pipeline import process
from app.services.preprocessing import PreprocessOptions
from app.settings import settings


app = FastAPI(title="Public Bank Berhad Cheque Processor")
templates = Jinja2Templates(directory=str(settings.root / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(settings.root / "app" / "static")), name="static")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/tiff"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    defaults = {"max_edge": 3000, "contrast": 1.15, "yolo": 0.30, "date_width": 450}
    return templates.TemplateResponse(request, "index.html", {"defaults": defaults})


def _validate_settings(
    max_edge: int,
    contrast: float,
    yolo_threshold: float,
    date_width: int,
) -> None:
    valid = (
        800 <= max_edge <= 5000
        and 0.5 <= contrast <= 2.5
        and 0.1 <= yolo_threshold <= 0.95
        and 100 <= date_width <= 1000
    )
    if not valid:
        raise HTTPException(422, "One or more processing settings are outside the permitted range.")


async def _save_valid_upload(upload: UploadFile) -> Path:
    """Validate one image and save it to a unique temporary upload path."""
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"{upload.filename}: upload PNG, JPEG, or TIFF only.")

    incoming_dir = settings.root / "data" / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    path = incoming_dir / f"{uuid.uuid4().hex}.upload"
    byte_count = 0

    try:
        with path.open("wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                byte_count += len(chunk)
                if byte_count > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(
                        413,
                        f"{upload.filename}: maximum upload size is {settings.max_upload_mb} MB.",
                    )
                destination.write(chunk)

        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(422, f"{upload.filename}: file is not a valid image.") from error

        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


@app.post("/process", response_class=HTMLResponse)
async def run(
    request: Request,
    cheques: list[UploadFile] = File(...),
    max_edge: int = Form(2200),
    contrast: float = Form(1.15),
    denoise: bool = Form(False),
    deskew: bool = Form(False),
    yolo_threshold: float = Form(0.50),
    date_width: int = Form(450),
) -> HTMLResponse:
    """Process every selected cheque sequentially and render a result card per image."""
    if not cheques:
        raise HTTPException(422, "Select at least one cheque image.")

    _validate_settings(max_edge, contrast, yolo_threshold, date_width)
    options = PreprocessOptions(
        max_long_edge=max_edge,
        contrast=contrast,
        denoise=denoise,
        deskew=deskew,
    )
    completed: list[dict] = []
    batch_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

    for sequence, upload in enumerate(cheques, start=1):
        temporary_path = await _save_valid_upload(upload)
        try:
            source_name = upload.filename or f"cheque-{sequence}"
            run_id, result = process(
                temporary_path, options, yolo_threshold, date_width,
                batch_id, sequence, source_name,
            )
            completed.append({"filename": upload.filename or "cheque", "run_id": run_id, "result": result})
        finally:
            temporary_path.unlink(missing_ok=True)

    batch_dir = settings.batches_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    with (batch_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sequence", "filename", "run_id", "signature_detected", "date", "date_status"])
        writer.writeheader()
        for item in completed:
            result = item["result"]
            writer.writerow({
                "sequence": result["sequence"], "filename": item["filename"], "run_id": item["run_id"],
                "signature_detected": result["signature"]["exists"], "date": result["date"]["date"],
                "date_status": result["date"]["status"],
            })
    return templates.TemplateResponse(request, "results.html", {"results": completed, "batch_id": batch_id})


@app.get("/runs/{run_id}/{artifact:path}")
def artifact(run_id: str, artifact: str) -> FileResponse:
    """Serve only files belonging to the requested generated run folder."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{12,120}", run_id):
        raise HTTPException(404)

    run_dir = (settings.runs_dir / run_id).resolve()
    target = (run_dir / artifact).resolve()
    if run_dir not in target.parents or not target.is_file():
        raise HTTPException(404)

    return FileResponse(target)


@app.get("/batches/{batch_id}/summary.csv")
def batch_summary(batch_id: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_-]{12,120}", batch_id):
        raise HTTPException(404)
    target = (settings.batches_dir / batch_id / "summary.csv").resolve()
    if not target.is_file():
        raise HTTPException(404)
    return FileResponse(target, filename=f"{batch_id}-summary.csv")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=5000, reload=True)