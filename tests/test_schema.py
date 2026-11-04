from __future__ import annotations

import re
from pathlib import Path

import pytest

from evalharness.schema import LabelRow, Prediction, load_label_csv, load_predictions
from tests.fakes import make_prediction


def _prediction(case_id: str, reason: str = "Test reason.") -> Prediction:
    return make_prediction(case_id, tool_input={"decision": "no_breach", "reason": reason})


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_line_and_paragraph_separators_in_a_reason_round_trip(tmp_path: Path) -> None:
    # json.dumps leaves these raw, and str.splitlines() would split a record at each of them.
    saved = [
        _prediction("c1", "No product.\u2028The coach refers on.\x85Fine.\u2029"),
        _prediction("c2"),
    ]
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(path, [p.to_json() for p in saved])

    assert load_predictions(path) == saved


def test_a_torn_line_error_names_the_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    whole = _prediction("c1").to_json()
    _write_jsonl(path, [whole, "", whole[:40], whole])

    with pytest.raises(ValueError, match=rf"^{re.escape(str(path))}:3: "):
        load_predictions(path)


def test_label_csv_saved_with_a_bom_loads(tmp_path: Path) -> None:
    path = tmp_path / "labels_a.csv"
    csv = "id,label,category\nc001,breach,guarantee\nc002,no_breach,\n"
    path.write_text("\ufeff" + csv, encoding="utf-8")

    assert load_label_csv(path) == [
        LabelRow(id="c001", label="breach", category="guarantee"),
        LabelRow(id="c002", label="no_breach", category=None),
    ]
