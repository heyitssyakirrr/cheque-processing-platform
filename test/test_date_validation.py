"""valid_date column: app/services/date_extraction.is_valid_date and export_batch's CSV copy."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import run  # noqa: E402

from app.services.date_extraction import is_valid_date, parse_date  # noqa: E402


def test_valid_dates():
    assert parse_date("02/08/11") == date(2011, 8, 2)
    assert parse_date("02/08/2011") == date(2011, 8, 2)
    assert parse_date("310811") == date(2011, 8, 31)
    assert parse_date("29/02/24") == date(2024, 2, 29)


def test_invalid_dates():
    for value in ("88/55/28", "01/46/74", "15/00/81", "00/08/11", "31/02/11", "29/02/23",
                  "88/88/88", "", "1008", "abcdef", None):
        assert not is_valid_date(value), value


def test_export_adds_column_after_date():
    # Loaded by path: export_batch.py sits at the project root, not in app/.
    path = Path(__file__).resolve().parents[1] / "export_batch.py"
    spec = importlib.util.spec_from_file_location("export_batch", path)
    export_batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_batch)

    with tempfile.TemporaryDirectory() as folder:
        source, target = Path(folder) / "in.csv", Path(folder) / "out.csv"
        source.write_text("cheque_id,date,error\nA,02/08/11,\nB,88/55/28,\n", encoding="utf-8")
        export_batch._copy_with_valid_date(source, target)
        with target.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == ["cheque_id", "date", "valid_date", "error"]
            assert [row["valid_date"] for row in reader] == ["YES", "NO"]


if __name__ == "__main__":
    run([value for name, value in list(globals().items()) if name.startswith("test_")])
