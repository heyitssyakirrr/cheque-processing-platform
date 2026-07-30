"""FastAPI web application for batch cheque processing."""

import uuid
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
    defaults = {"max_edge": 2200, "contrast": 1.15, "yolo": 0.50, "date_width": 450}
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

    for upload in cheques:
        temporary_path = await _save_valid_upload(upload)
        try:
            run_id, result = process(temporary_path, options, yolo_threshold, date_width)
            completed.append({"filename": upload.filename or "cheque", "run_id": run_id, "result": result})
        finally:
            temporary_path.unlink(missing_ok=True)

    return templates.TemplateResponse(request, "results.html", {"results": completed})


@app.get("/runs/{run_id}/{artifact:path}")
def artifact(run_id: str, artifact: str) -> FileResponse:
    """Serve only files belonging to the requested generated run folder."""
    if len(run_id) != 12 or any(char not in "0123456789abcdef" for char in run_id):
        raise HTTPException(404)

    run_dir = (settings.runs_dir / run_id).resolve()
    target = (run_dir / artifact).resolve()
    if run_dir not in target.parents or not target.is_file():
        raise HTTPException(404)

    return FileResponse(target)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=5000, reload=True)