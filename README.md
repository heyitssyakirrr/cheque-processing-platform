# Public Bank Berhad cheque processing

An auditable, modular FastAPI application that processes one cheque at a time:

`upload -> preprocess -> YOLOv8 signature detection + PaddleOCR -> date label anchor -> TrOCR -> structured results`

## What is included

- A Public Bank Berhad-inspired red user interface with progress states.
- Adjustable, safe preprocessing defaults: maximum long edge 2200 px, 300-DPI equivalent scale, grayscale/contrast denoise, and automatic deskew. The untouched original is retained.
- YOLOv8 `.pt` signature inference, annotated output, per-signature crops, and CSV/JSON results.
- PaddleOCR text detection/recognition with quadrilateral coordinates and confidence-labelled annotation.
- Date-field finding through robust OCR aliases (`TARIKH`, `DATE`, likely OCR mistakes), followed by TrOCR handwriting recognition.
- Run folders that keep every artifact together and a convenient `summary.csv`.

## Quick start on Windows

Requires Python 3.10 or 3.11 (PaddleOCR support is substantially better than with Python 3.9).

```powershell
cd cheque-processing-platform
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your Hugging Face YOLOv8 TorchScript file in `models/signature-yolov8.torchscript` (the default used by your signature-detection project). Alternatively, set `YOLO_MODEL_PATH` to a compatible local `.pt`/`.torchscript` file or set both `HF_YOLO_REPO_ID` and `HF_YOLO_FILENAME` in `.env`; it downloads on first use and caches locally.

Start the application:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. First model use can take a little time because PaddleOCR and TrOCR download their public weights. For offline deployment, run a test cheque once while online, then keep the cache/model folders.

## Output layout

Each upload gets `data/runs/<run-id>/`:

```
original/cheque.png
preprocessed/cheque.png
signature/annotated.png, crops/*.png, detections.csv
text/annotated.png, detections.csv
date/crop.png, result.csv
result.json
summary.csv
```

`result.json` is the complete machine-readable record. CSVs are intentionally split by task so they can be inspected in Excel without parsing nested data.

## Code layout

```
app/
  main.py             # web routes and upload validation
  pipeline.py         # orchestration only
  services/
    preprocessing.py
    signature_detection.py
    text_detection.py
    date_extraction.py
    field_extraction.py
  templates/          # UI pages
  static/             # CSS and browser assets
```

## Important operating notes

- Treat date/name/amount extraction as review-assisted: cheque templates and handwriting vary, so operationally validate fields before acting on them.
- The initial date locator is template-independent (label + geometry). If your Public Bank templates differ, edit `app/settings.py` aliases and date crop dimensions, then use the UI controls for per-run tuning.
- Files are limited to PNG/JPEG/TIFF, validated by Pillow, stored under a generated run ID, and never executed. Deploy behind authentication/TLS before exposing beyond localhost.
