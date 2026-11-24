"""The output CSV's columns, and scoring a batch against an answer key.

Output CSV (data/batches/<date>/<date>_output.csv), one row per cheque:

    batch_date, no, bank, file_name          which cheque
    signature                                YES / NO
    date                                     DD/MM/YYYY when VALID; what was
                                             read when INVALID; NO DATE
    date_status                              VALID / INVALID / NO DATE
    flagged, flag_reason                     YES/NO and why (NO SIGNATURE,
                                             NO DATE, INVALID DATE; "; " joined)
    processed_on, start_time, end_time       date, then HH:MM:SS (on their own
                                             so Excel shows the seconds)
    duration_sec, error
    image_size, t_*_sec                      technical timings

Scored report (<date>_scored.csv, written by export_batch.py when the batch
folder has answer_key.csv): the cheque, what the model said, what the answer
key says, 1/0 accuracy per row, then TOTAL and PERCENTAGE rows.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from app.dates import parse_date

NO_DATE = "NO DATE"
VALID, INVALID = "VALID", "INVALID"

OUTPUT_FIELDS = [
    "batch_date", "no", "bank", "file_name",
    "signature", "date", "date_status", "flagged", "flag_reason",
    "processed_on", "start_time", "end_time", "duration_sec", "error",
    "image_size", "t_read_sec", "t_preprocess_sec", "t_signature_sec", "t_ocr_sec", "t_date_sec",
]
ANSWER_FIELDS = ["file_name", "signature", "date", "note"]
SCORED_FIELDS = [
    "no", "bank", "file_name",
    "signature", "signature_expected", "signature_accuracy",
    "date", "date_status", "date_expected", "date_accuracy",
    "flagged", "flag_reason", "duration_sec", "note",
]
TIMINGS = ("read", "preprocess", "signature", "ocr", "date")


# --- one output row ----------------------------------------------------------

def date_fields(raw: str | None) -> tuple[str, str]:
    """(date, date_status) for what the reader returned: "" means blank."""
    raw = (raw or "").strip()
    if not raw or raw.upper() == NO_DATE:
        return NO_DATE, NO_DATE
    parsed = parse_date(raw)
    if parsed:
        return parsed.strftime("%d/%m/%Y"), VALID
    return raw, INVALID


def flag_fields(signature: str, date_status: str) -> tuple[str, str]:
    """(flagged, flag_reason): YES when a person should check the cheque."""
    reasons = []
    if signature != "YES":
        reasons.append("NO SIGNATURE")
    if date_status == NO_DATE:
        reasons.append("NO DATE")
    elif date_status != VALID:
        reasons.append("INVALID DATE")
    return ("YES" if reasons else "NO"), "; ".join(reasons)


def output_row(
    *, batch_id: str, sequence: int, bank: str, file_name: str, signature_found: bool,
    raw_date: str, started: datetime, ended: datetime, duration: float,
    timings: dict[str, Any], image_size: str = "", error: str = "",
) -> dict[str, Any]:
    signature = "YES" if signature_found else "NO"
    date, status = date_fields(raw_date)
    flagged, reason = flag_fields(signature, status)
    row = {
        "batch_date": batch_id, "no": sequence, "bank": bank, "file_name": file_name,
        "signature": signature, "date": date, "date_status": status,
        "flagged": flagged, "flag_reason": reason,
        "processed_on": started.strftime("%Y-%m-%d"),
        "start_time": started.strftime("%H:%M:%S"), "end_time": ended.strftime("%H:%M:%S"),
        "duration_sec": round(duration, 2), "error": error, "image_size": image_size,
    }
    for name in TIMINGS:
        row[f"t_{name}_sec"] = timings.get(name, "")
    return row


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    """Any row, new layout or one written before it (cheque_id, filename,
    signature_detected, valid_date, is_flagged, "YYYY-MM-DD HH:MM:SS"
    times), in the current OUTPUT_FIELDS layout."""
    if "file_name" in row:
        return {field: row.get(field, "") for field in OUTPUT_FIELDS}
    signature = row.get("signature_detected", "")
    date, status = date_fields(row.get("date", ""))
    flagged, reason = flag_fields(signature, status)

    def split(stamp: str) -> tuple[str, str]:
        stamp = str(stamp or "")
        return (stamp[:10], stamp[11:19]) if len(stamp) >= 19 else ("", stamp)

    processed_on, start_time = split(row.get("start_time", ""))
    _, end_time = split(row.get("end_time", ""))
    new = {
        "batch_date": row.get("batch_id", ""), "no": row.get("sequence", ""),
        "bank": row.get("bank_code", ""), "file_name": row.get("filename", ""),
        "signature": signature, "date": date, "date_status": status,
        "flagged": flagged, "flag_reason": reason, "processed_on": processed_on,
        "start_time": start_time, "end_time": end_time,
        "duration_sec": row.get("duration_seconds", ""), "error": row.get("error", ""),
        "image_size": row.get("pixels", ""),
    }
    for name in TIMINGS:
        new[f"t_{name}_sec"] = row.get(f"t_{name}", "")
    return new


def read_output(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = [normalize(row) for row in csv.DictReader(file)]
    return sorted(rows, key=lambda r: int(r["no"]) if str(r["no"]).isdigit() else 0)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --- answer key and scoring ---------------------------------------------------

MONTHS = {name: number for number, names in enumerate((
    ("JAN", "JANUARY"), ("FEB", "FEBRUARY"), ("MAR", "MARCH"), ("APR", "APRIL"), ("MAY",), ("JUN", "JUNE"),
    ("JUL", "JULY"), ("AUG", "AUGUST"), ("SEP", "SEPT", "SEPTEMBER"), ("OCT", "OCTOBER"),
    ("NOV", "NOVEMBER"), ("DEC", "DECEMBER"),
), start=1) for name in names}


def _named_month_date(value: str) -> str | None:
    """"01 AUG 2011", "1-Aug-11", "01 August 2011" -> "01/08/2011"; else None."""
    parts = re.split(r"[\s/.\-]+", value.strip().upper())
    if len(parts) != 3 or parts[1] not in MONTHS or not parts[0].isdigit() or not parts[2].isdigit():
        return None
    year = int(parts[2]) + (2000 if len(parts[2]) == 2 else 0)
    parsed = parse_date(f"{int(parts[0]):02d}{MONTHS[parts[1]]:02d}{year:04d}")
    return parsed.strftime("%d/%m/%Y") if parsed else None


def expected_date(text: str) -> str:
    """An answer-key date as NO DATE or DD/MM/YYYY. Accepts 310811, 31082011,
    31/08/11, 31/08/2011, the leading zero Excel drops (10811 = 010811), and
    a month name as on a stamp or as Excel shows dates (01 AUG 2011,
    1-Aug-11, 01 August 2011)."""
    value = (text or "").strip()
    if value.upper().replace(" ", "") in {"NODATE", "BLANK", "EMPTY", "-"}:
        return NO_DATE
    named = _named_month_date(value)
    if named:
        return named
    digits = value.replace("/", "").replace("-", "").replace(".", "").replace(" ", "")
    if digits.isdigit() and len(digits) in (5, 7):
        digits = "0" + digits  # Excel turned 010811 into 10811
    parsed = parse_date(digits if digits.isdigit() else value)
    if parsed is None:
        raise ValueError(f"not a date or NO DATE: {text!r}")
    return parsed.strftime("%d/%m/%Y")


def write_answer_template(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, ANSWER_FIELDS, ({"file_name": r["file_name"]} for r in rows))


def load_answers(path: Path) -> dict[str, dict[str, str]]:
    """file_name -> {signature, date (NO DATE or DD/MM/YYYY), note}. Rows with
    both signature and date empty are not filled in yet and are skipped.
    Raises ValueError listing every bad row."""
    answers, problems = {}, []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:  # Excel's "CSV (Comma delimited)" saves in the Windows code page
        text = path.read_text(encoding="cp1252")
    for line, row in enumerate(csv.DictReader(text.splitlines()), start=2):
        name = (row.get("file_name") or "").strip()
        signature = (row.get("signature") or "").strip().upper()
        date_text = (row.get("date") or "").strip()
        if not name or (not signature and not date_text):
            continue
        if signature in {"Y", "1"}:
            signature = "YES"
        elif signature in {"N", "0"}:
            signature = "NO"
        if signature not in {"YES", "NO"}:
            problems.append(f"line {line} ({name}): signature must be YES or NO, not {signature!r}")
            continue
        try:
            date = expected_date(date_text)
        except ValueError as error:
            problems.append(f"line {line} ({name}): {error}")
            continue
        if name in answers:
            problems.append(f"line {line}: {name} is listed twice")
            continue
        answers[name] = {"signature": signature, "date": date, "note": (row.get("note") or "").strip()}
    if problems:
        raise ValueError(f"{path} has {len(problems)} problem(s): " + "; ".join(problems))
    return answers


def scored_rows(rows: list[dict[str, Any]], answers: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Each output row with the answer and 1/0 accuracy, then TOTAL and
    PERCENTAGE rows. Date accuracy is strict: NO DATE is only right when the
    model also found no date; a date only when the model read that same,
    valid date. Rows missing from the answer key are shown but not scored.
    Raises ValueError when a file name appears in more than one bank."""
    names = [row["file_name"] for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"the same file name is in more than one bank ({', '.join(duplicates[:3])}), "
                         "so the answer key cannot tell them apart")
    scored, sig_right, date_right, answered, flagged, seconds = [], 0, 0, 0, 0, 0.0
    for row in rows:
        out = {field: row.get(field, "") for field in SCORED_FIELDS}
        answer = answers.get(row["file_name"])
        flagged += row["flagged"] == "YES"
        try:
            seconds += float(row["duration_sec"])
        except (TypeError, ValueError):
            pass
        if answer:
            answered += 1
            out["signature_expected"], out["date_expected"], out["note"] = (
                answer["signature"], answer["date"], answer["note"])
            out["signature_accuracy"] = int(row["signature"] == answer["signature"])
            if answer["date"] == NO_DATE:
                out["date_accuracy"] = int(row["date_status"] == NO_DATE)
            else:
                out["date_accuracy"] = int(row["date_status"] == VALID and row["date"] == answer["date"])
            sig_right += out["signature_accuracy"]
            date_right += out["date_accuracy"]
        scored.append(out)

    def pct(part: int, whole: int) -> str:
        return f"{100 * part / whole:.1f}%" if whole else ""

    total = len(rows)
    scored.append({"no": "TOTAL", "signature_accuracy": f"{sig_right}/{answered}",
                   "date_accuracy": f"{date_right}/{answered}", "flagged": f"{flagged}/{total}",
                   "duration_sec": round(seconds, 2)})
    scored.append({"no": "PERCENTAGE", "signature_accuracy": pct(sig_right, answered),
                   "date_accuracy": pct(date_right, answered), "flagged": pct(flagged, total)})
    return scored
