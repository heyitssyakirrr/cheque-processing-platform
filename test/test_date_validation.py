"""Date checks (app/dates.py), the output columns and scoring (app/report.py),
and export_batch.py end to end on a small fake batch."""

from __future__ import annotations

import csv
import importlib.util
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import DATE, TempProject, run  # noqa: E402

from Cheque_Processing.app import report  # noqa: E402
from Cheque_Processing.app.dates import is_flagged, is_valid_date, parse_date  # noqa: E402
from Cheque_Processing.app.settings import settings  # noqa: E402


def test_valid_dates():
    assert parse_date("02/08/11") == date(2011, 8, 2)
    assert parse_date("02/08/2011") == date(2011, 8, 2)
    assert parse_date("310811") == date(2011, 8, 31)
    assert parse_date("29/02/24") == date(2024, 2, 29)


def test_invalid_dates():
    for value in ("88/55/28", "01/46/74", "15/00/81", "00/08/11", "31/02/11", "29/02/23",
                  "88/88/88", "", "1008", "abcdef", None):
        assert not is_valid_date(value), value


def test_is_flagged():
    assert is_flagged("YES", "YES") == "NO"
    assert is_flagged("NO", "YES") == "YES"
    assert is_flagged("YES", "NO") == "YES"


def test_date_status_and_flag_reason():
    assert report.date_fields("31/08/11") == ("31/08/2011", "VALID")
    assert report.date_fields("18/21/71") == ("18/21/71", "INVALID")
    assert report.date_fields("") == ("NO DATE", "NO DATE")
    assert report.flag_fields("YES", "VALID") == ("NO", "")
    assert report.flag_fields("NO", "NO DATE") == ("YES", "NO SIGNATURE; NO DATE")
    assert report.flag_fields("YES", "INVALID") == ("YES", "INVALID DATE")


def test_output_row_keeps_seconds_apart():
    row = report.output_row(
        batch_id=DATE, sequence=3, bank="PBB", file_name="A.IMG", signature_found=True, raw_date="02/08/11",
        started=datetime(2026, 11, 23, 11, 47, 51), ended=datetime(2026, 11, 23, 11, 47, 58),
        duration=7.123, timings={"ocr": 4.3})
    assert list(row) == report.OUTPUT_FIELDS
    assert (row["processed_on"], row["start_time"], row["end_time"]) == ("2026-11-23", "11:47:51", "11:47:58")
    assert (row["date"], row["date_status"], row["flagged"], row["duration_sec"]) == ("02/08/2011", "VALID", "NO", 7.12)


def test_older_rows_are_converted():
    old = {"batch_id": DATE, "cheque_id": "PBB/A.IMG", "bank_code": "PBB", "sequence": "7", "filename": "A.IMG",
           "signature_detected": "NO", "date": "", "valid_date": "NO", "is_flagged": "YES",
           "start_time": "2026-11-23 11:47:51", "end_time": "2026-11-23 11:47:58", "duration_seconds": "7.1",
           "pixels": "2000x1000", "t_ocr": "4.2", "error": ""}
    new = report.normalize(old)
    assert list(new) == report.OUTPUT_FIELDS
    assert (new["no"], new["bank"], new["file_name"], new["signature"]) == ("7", "PBB", "A.IMG", "NO")
    assert (new["date"], new["date_status"], new["flag_reason"]) == ("NO DATE", "NO DATE", "NO SIGNATURE; NO DATE")
    assert (new["processed_on"], new["start_time"], new["t_ocr_sec"]) == ("2026-11-23", "11:47:51", "4.2")


def test_answer_dates_in_any_form():
    for text in ("310811", "31082011", "31/08/11", "31/08/2011", "31-08-2011"):
        assert report.expected_date(text) == "31/08/2011", text
    assert report.expected_date("10811") == "01/08/2011"  # Excel dropped the leading zero of 010811
    for text in ("01 AUG 2011", "1-Aug-11", "01 August 2011", "01/aug/2011"):  # stamps, Excel's display
        assert report.expected_date(text) == "01/08/2011", text
    for text in ("31 SEP 2011", "01 XYZ 2011"):
        try:
            report.expected_date(text)
        except ValueError:
            pass
        else:
            raise AssertionError(text)
    for text in ("NO DATE", "no date", "blank", "-"):
        assert report.expected_date(text) == "NO DATE"


def _row(no, name, signature, raw_date):
    return report.output_row(batch_id=DATE, sequence=no, bank="PBB", file_name=name, signature_found=signature,
                             raw_date=raw_date, started=datetime(2026, 1, 1), ended=datetime(2026, 1, 1),
                             duration=2.0, timings={})


def test_scoring_is_strict_with_totals():
    rows = [
        _row(1, "A.IMG", True, "31/08/11"),   # right date, right signature
        _row(2, "B.IMG", True, "18/21/71"),   # scribble on a blank date: INVALID is not NO DATE -> 0
        _row(3, "C.IMG", False, ""),          # no date found, answer NO DATE -> 1; signature missed -> 0
        _row(4, "D.IMG", True, "22/06/11"),   # valid but wrong date -> 0
        _row(5, "E.IMG", True, "02/08/11"),   # not in the answer key: shown, not scored
    ]
    answers = {
        "A.IMG": {"signature": "YES", "date": "31/08/2011", "note": ""},
        "B.IMG": {"signature": "YES", "date": "NO DATE", "note": "scribble"},
        "C.IMG": {"signature": "YES", "date": "NO DATE", "note": ""},
        "D.IMG": {"signature": "YES", "date": "28/06/2011", "note": "8 looks like 2"},
    }
    scored = report.scored_rows(rows, answers)
    assert [r["date_accuracy"] for r in scored[:5]] == [1, 0, 1, 0, ""]
    assert [r["signature_accuracy"] for r in scored[:5]] == [1, 1, 0, 1, ""]
    total, percentage = scored[-2], scored[-1]
    assert (total["no"], total["signature_accuracy"], total["date_accuracy"], total["flagged"]) == \
        ("TOTAL", "3/4", "2/4", "2/5")
    assert (percentage["signature_accuracy"], percentage["date_accuracy"], percentage["flagged"]) == \
        ("75.0%", "50.0%", "40.0%")
    assert scored[1]["note"] == "scribble"


def test_bad_answer_key_lists_every_problem():
    with TempProject():
        path = settings.batch_dir(DATE) / "answer_key.csv"
        path.parent.mkdir(parents=True)
        path.write_text("file_name,signature,date,note\nA.IMG,YES,310811,\nB.IMG,MAYBE,310811,\n"
                        "C.IMG,NO,99/99/99,\nD.IMG,,,\n", encoding="utf-8")
        try:
            report.load_answers(path)
        except ValueError as error:
            assert "2 problem(s)" in str(error) and "B.IMG" in str(error) and "C.IMG" in str(error)
        else:
            raise AssertionError("expected ValueError")


def _export_module():
    path = Path(__file__).resolve().parents[1] / "export_batch.py"
    spec = importlib.util.spec_from_file_location("export_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_converts_and_scores():
    export_batch = _export_module()
    with TempProject():
        batch_dir = settings.batch_dir(DATE)
        batch_dir.mkdir(parents=True)
        (batch_dir / f"{DATE}_output.csv").write_text(  # an output written before the new layout
            "batch_id,cheque_id,bank_code,sequence,filename,signature_detected,date,valid_date,is_flagged,"
            "start_time,end_time,duration_seconds,error\n"
            f"{DATE},PBB/A.IMG,PBB,1,A.IMG,YES,31/08/11,YES,NO,2026-11-23 11:47:51,2026-11-23 11:47:58,7.1,\n"
            f"{DATE},PBB/B.IMG,PBB,2,B.IMG,NO,,NO,YES,2026-11-23 11:47:58,2026-11-23 11:48:05,6.9,\n",
            encoding="utf-8")
        assert export_batch.main(["--batch-id", DATE]) == 0
        out = batch_dir / "batch output"
        with (out / f"{DATE}_output.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == report.OUTPUT_FIELDS
            rows = list(reader)
        assert [r["date"] for r in rows] == ["31/08/2011", "NO DATE"]
        assert not (out / f"{DATE}_scored.csv").exists()

        assert export_batch.main(["--batch-id", DATE, "--write-answer-key"]) == 0
        key = batch_dir / "answer_key.csv"
        assert key.read_text(encoding="utf-8").splitlines() == ["file_name,signature,date,note", "A.IMG,,,", "B.IMG,,,"]
        key.write_text("file_name,signature,date,note\nA.IMG,YES,310811,\nB.IMG,YES,NO DATE,missed\n", encoding="utf-8")
        assert export_batch.main(["--batch-id", DATE]) == 0
        with (out / f"{DATE}_scored.csv").open(newline="", encoding="utf-8") as file:
            scored = list(csv.DictReader(file))
        assert [r["no"] for r in scored] == ["1", "2", "TOTAL", "PERCENTAGE"]
        assert (scored[2]["signature_accuracy"], scored[2]["date_accuracy"], scored[2]["flagged"]) == ("1/2", "2/2", "1/2")
        assert (scored[3]["signature_accuracy"], scored[3]["date_accuracy"]) == ("50.0%", "100.0%")


def _old_output(batch_dir: Path) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / f"{DATE}_output.csv").write_text(
        "batch_id,cheque_id,bank_code,sequence,filename,signature_detected,date,valid_date,is_flagged,"
        "start_time,end_time,duration_seconds,error\n"
        f"{DATE},PBB/A.IMG,PBB,1,A.IMG,YES,31/08/11,YES,NO,2026-11-23 11:47:51,2026-11-23 11:47:58,7.1,\n",
        encoding="utf-8")


def test_production_without_answer_key_gets_output_only():
    export_batch = _export_module()
    with TempProject():
        batch_dir = settings.batch_dir(DATE)
        _old_output(batch_dir)
        out = batch_dir / "batch output"
        out.mkdir(parents=True)
        stale = out / f"{DATE}_scored.csv"
        stale.write_text("left from an earlier answer key", encoding="utf-8")
        assert export_batch.main(["--batch-id", DATE]) == 0
        assert (out / f"{DATE}_output.csv").is_file()
        assert not stale.exists()  # no answer key now: no scored report, not even an old one


def test_answer_key_saved_by_excel_in_windows_encoding():
    with TempProject():
        path = settings.batch_dir(DATE) / "answer_key.csv"
        path.parent.mkdir(parents=True)
        path.write_bytes("file_name,signature,date,note\nA.IMG,Y,10811,café stamp\n".encode("cp1252"))
        answers = report.load_answers(path)
        assert answers["A.IMG"] == {"signature": "YES", "date": "01/08/2011", "note": "café stamp"}


def test_same_file_name_in_two_banks_is_refused():
    rows = [_row(1, "A.IMG", True, "31/08/11"), _row(2, "A.IMG", True, "")]
    rows[1]["bank"] = "PIBB"
    try:
        report.scored_rows(rows, {"A.IMG": {"signature": "YES", "date": "31/08/2011", "note": ""}})
    except ValueError as error:
        assert "more than one bank" in str(error)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    run([value for name, value in list(globals().items()) if name.startswith("test_")])
