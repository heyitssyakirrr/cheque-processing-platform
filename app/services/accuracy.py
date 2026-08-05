"""Compare user-supplied ground truth against a batch's actual results.

Two things get scored per cheque:
- Signature count: does the number of accepted signature detections match
  what a human counted on the actual cheque?
- Date digits: compared position-by-position against the ground truth
  6-digit string (or, if the ground truth says there's no date at all,
  scored as correct only if the pipeline also produced nothing usable).
"""

from pathlib import Path
from typing import Any


def parse_ground_truth(raw_text: str) -> dict[str, dict[str, Any]]:
    """Parse the simple pasted format, one cheque per line:

        <filename-stem>: <signature_count>, <date as DDMMYY or 'no date'>

    e.g. "bank_cheque_1: 2, 100726" or "bank_cheque_5: 2, no date".
    Keyed by lowercased filename stem so it matches regardless of the
    original file's extension or case.
    """
    ground_truth: dict[str, dict[str, Any]] = {}
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name_part, rest = line.split(":", 1)
        pieces = [piece.strip() for piece in rest.split(",")]
        if len(pieces) < 2 or not pieces[0].lstrip("-").isdigit():
            continue
        date_text = pieces[1].strip().lower()
        date_value = None if date_text in ("no date", "none", "", "-") else "".join(filter(str.isdigit, pieces[1]))
        ground_truth[name_part.strip().lower()] = {
            "signature_count": int(pieces[0]),
            "date": date_value or None,
        }
    return ground_truth


def _score_signature(predicted_count: int, gt_count: int) -> dict[str, Any]:
    return {
        "gt_count": gt_count,
        "predicted_count": predicted_count,
        "match": predicted_count == gt_count,
        "difference": predicted_count - gt_count,
    }


def _score_digits(predicted_raw: str, gt_date: str | None) -> dict[str, Any]:
    if gt_date is None:
        # Ground truth says the cheque has no date at all. Since there's
        # currently no dedicated "no date detected" status, this is scored
        # as correct only if the classifier didn't produce a full 6-digit
        # reading -- an empty/short raw_digits string.
        no_date_predicted = not predicted_raw or len(predicted_raw) != 6
        return {
            "gt_date": "no date",
            "predicted_raw": predicted_raw or "(none)",
            "total_digits": None,
            "correct_digits": None,
            "digit_accuracy_pct": None,
            "per_digit_correct": None,
            "exact_match": no_date_predicted,
        }

    predicted = (predicted_raw or "")[:6].ljust(6, "?")
    per_digit_correct = [p == g for p, g in zip(predicted, gt_date)]
    correct = sum(per_digit_correct)
    return {
        "gt_date": gt_date,
        "predicted_raw": predicted_raw or "(none)",
        "total_digits": 6,
        "correct_digits": correct,
        "digit_accuracy_pct": round(correct / 6 * 100, 1),
        "per_digit_correct": per_digit_correct,
        "exact_match": predicted_raw == gt_date,
    }


def build_accuracy_report(ground_truth: dict[str, dict[str, Any]], completed: list[dict]) -> dict[str, Any]:
    """Score every completed cheque that has matching ground truth.

    Cheques with no corresponding ground truth entry are silently skipped
    (not everything in a batch necessarily has known-correct answers yet).
    """
    rows: list[dict[str, Any]] = []
    for item in completed:
        stem = Path(item["filename"]).stem.lower()
        gt = ground_truth.get(stem)
        if gt is None:
            continue

        result = item["result"]
        predicted_signature_count = sum(1 for d in result["signature"]["detections"] if d["accepted"])
        rows.append(
            {
                "filename": item["filename"],
                "run_id": item["run_id"],
                "signature": _score_signature(predicted_signature_count, gt["signature_count"]),
                "digit": _score_digits(result["date"]["raw_digits"], gt["date"]),
            }
        )

    signature_matches = sum(1 for row in rows if row["signature"]["match"])
    scored_for_digits = [row for row in rows if row["digit"]["total_digits"] is not None]
    total_digit_correct = sum(row["digit"]["correct_digits"] for row in scored_for_digits)
    total_digit_count = sum(row["digit"]["total_digits"] for row in scored_for_digits)
    exact_date_matches = sum(1 for row in rows if row["digit"]["exact_match"])

    return {
        "rows": rows,
        "summary": {
            "cheques_scored": len(rows),
            "signature_matches": signature_matches,
            "signature_accuracy_pct": round(signature_matches / len(rows) * 100, 1) if rows else None,
            "total_digit_correct": total_digit_correct,
            "total_digit_count": total_digit_count,
            "digit_accuracy_pct": round(total_digit_correct / total_digit_count * 100, 1) if total_digit_count else None,
            "exact_date_matches": exact_date_matches,
            "exact_date_match_pct": round(exact_date_matches / len(rows) * 100, 1) if rows else None,
        },
    }