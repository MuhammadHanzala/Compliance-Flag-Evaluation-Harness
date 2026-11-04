from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message

from evalharness import cli
from evalharness.classifier import DEFAULT_MODEL
from evalharness.schema import CATEGORIES, LABELS, TRAPS, sha256_file
from tests.fakes import (
    PRICING_TOML,
    FakeClient,
    answer_by_case,
    assessment_input,
    make_case,
    make_message,
    make_prediction,
    write_cases,
)

BREACH_1 = make_case("b1", label="breach", category="guarantee")
BREACH_2 = make_case("b2", label="breach", category="tax_legal")
CLEAN_1 = make_case("n1")
CLEAN_2 = replace(make_case("n2"), borderline=True)
CASES = [BREACH_1, BREACH_2, CLEAN_1, CLEAN_2]


def _write_run(run_dir: Path, eval_path: Path, flagged: dict[str, bool]) -> Path:
    """A run directory as the runner leaves it: manifest plus one prediction per case id."""
    run_dir.mkdir(parents=True)
    manifest = {
        "name": run_dir.name,
        "model": DEFAULT_MODEL,
        "effort": "low",
        "rubric_sha256": "0" * 64,
        "eval_set_sha256": sha256_file(eval_path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    lines = [
        make_prediction(case_id, "breach" if hit else "no_breach").to_json() + "\n"
        for case_id, hit in flagged.items()
    ]
    (run_dir / "predictions.jsonl").write_text("".join(lines), encoding="utf-8")
    return run_dir


@pytest.fixture
def eval_path(tmp_path: Path) -> Path:
    return write_cases(tmp_path / "eval_set.jsonl", CASES)


@pytest.fixture
def run_v1(tmp_path: Path, eval_path: Path) -> Path:
    """Catches both breaches and falsely flags both non-breaches."""
    return _write_run(
        tmp_path / "runs" / "v1", eval_path, dict.fromkeys(["b1", "b2", "n1", "n2"], True)
    )


def _compare(run_a: Path, run_b: Path, eval_path: Path, *extra: str) -> int:
    reports = run_a.parent.parent / "reports"  # tmp_path/reports, beside tmp_path/runs
    args = ["compare", str(run_a), str(run_b), "--eval", str(eval_path)]
    return cli.main([*args, "--reports-dir", str(reports), *extra])


def test_validate_prints_counts_for_a_valid_set(
    eval_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["validate", str(eval_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "4 cases, all valid" in out[0]
    for line in ("breach: 2", "  guarantee: 1", "  tax_legal: 1", "no_breach: 2"):
        assert line in out
    assert "  budgeting_action: 2" in out
    assert "borderline: 1" in out


def test_validate_returns_1_on_an_invalid_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_cases(tmp_path / "eval_set.jsonl", [CLEAN_1, CLEAN_1])
    assert cli.main(["validate", str(path)]) == 1
    assert "n1: duplicate id" in capsys.readouterr().err


def test_validate_returns_2_on_a_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["validate", str(tmp_path / "missing.jsonl")]) == 2
    assert capsys.readouterr().err.startswith("error:")


def _write_labels(path: Path, rows: list[str]) -> Path:
    text = "id,label,category,notes\n" + "".join(f"{row}\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def drafted_path(tmp_path: Path) -> Path:
    """Drafted labels for c1-c4; c4 is drafted as a breach."""
    drafted = [
        make_case("c1", label="breach", category="personal_advice"),
        make_case("c2", label="breach", category="guarantee"),
        make_case("c3"),
        make_case("c4", label="breach", category="tax_legal"),
    ]
    return write_cases(tmp_path / "eval_set.jsonl", drafted)


def _agreement(tmp_path: Path, drafted_path: Path, rows_a: list[str], rows_b: list[str]) -> int:
    labels_a = _write_labels(tmp_path / "labels_a.csv", rows_a)
    labels_b = _write_labels(tmp_path / "labels_b.csv", rows_b)
    return cli.main(["agreement", str(labels_a), str(labels_b), "--eval", str(drafted_path)])


def test_agreement_prints_kappa(
    tmp_path: Path, drafted_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean = ["c3,no_breach,,", "c4,no_breach,,"]
    rows_a = ["c1,breach,personal_advice,", "c2,breach,guarantee,", *clean]
    rows_b = ["c1,breach,guarantee,", "c2,no_breach,,", *clean]
    assert _agreement(tmp_path, drafted_path, rows_a, rows_b) == 0
    out = capsys.readouterr().out
    # Binary: observed 0.75, chance 0.5 -> 0.5. Five classes: observed 0.5, chance 7/16 -> 1/9.
    assert "Cohen's kappa, breach vs no_breach: 0.50" in out
    assert "Cohen's kappa, five classes (no_breach or breach type): 0.11" in out
    assert "c1: A personal_advice, B guarantee" in out
    assert "c2: A guarantee, B no_breach" in out
    assert "Both labellers differ from the drafted label: 1" in out
    assert "c4: drafted tax_legal, both no_breach" in out


def test_agreement_returns_2_when_the_labels_miss_an_eval_case(
    tmp_path: Path, drafted_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = ["c1,no_breach,,", "c2,no_breach,,", "c3,no_breach,,"]
    assert _agreement(tmp_path, drafted_path, rows, rows) == 2
    assert "the labels do not cover the eval set; missing: c4" in capsys.readouterr().err


def test_agreement_explains_a_blank_template(
    tmp_path: Path, drafted_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blank = ["c1,,,", "c2,,,", "c3,,,", "c4,,,"]
    assert _agreement(tmp_path, drafted_path, blank, blank) == 2
    err = capsys.readouterr().err
    assert "labels_a.csv has 4 of 4 rows still unlabelled" in err
    assert "data/labelling_sheet.md" in err


def test_sheet_shows_transcripts_without_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hidden = replace(BREACH_1, rationale="Drafted reason the labellers must not see.")
    path = write_cases(tmp_path / "eval_set.jsonl", [hidden, CLEAN_2])
    assert cli.main(["sheet", str(path)]) == 0
    out = capsys.readouterr().out
    assert "## b1" in out
    assert "## n2" in out
    assert "2. **coach:** Let's set a weekly spending limit." in out
    for word in (*LABELS, *CATEGORIES, *TRAPS, "rationale", "borderline", "Drafted reason"):
        assert word not in out


def test_compare_fails_the_gate_when_b_introduces_a_new_miss(
    tmp_path: Path, eval_path: Path, run_v1: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = {"b1": True, "b2": False, "n1": False, "n2": False}
    run_v2 = _write_run(tmp_path / "runs" / "v2", eval_path, flags)
    # A recall floor of 0 leaves the new miss as the only reason to fail.
    assert _compare(run_v1, run_v2, eval_path, "--recall-floor", "0") == 1
    out = capsys.readouterr().out
    assert "Gate: FAIL" in out
    assert "new misses" in out
    assert "b2" in out
    report = (tmp_path / "reports" / "report_v1_vs_v2.md").read_text(encoding="utf-8")
    assert "## Gate: FAIL" in report
    assert not list((tmp_path / "runs").glob("report_*"))  # reports stay out of runs/


def test_compare_passes_an_improving_run(
    tmp_path: Path, eval_path: Path, run_v1: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = {"b1": True, "b2": True, "n1": False, "n2": False}
    run_v2 = _write_run(tmp_path / "runs" / "v2", eval_path, flags)
    assert _compare(run_v1, run_v2, eval_path) == 0
    out = capsys.readouterr().out
    assert "false flags fixed 2, new false flags 0, new misses 0, misses fixed 0" in out
    assert "Gate: PASS" in out
    assert (tmp_path / "reports" / "report_v1_vs_v2.md").exists()


def test_compare_returns_2_on_an_incomplete_run(
    tmp_path: Path, eval_path: Path, run_v1: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_v2 = _write_run(tmp_path / "runs" / "v2", eval_path, {"b1": True, "b2": True, "n1": False})
    assert _compare(run_v1, run_v2, eval_path) == 2
    err = capsys.readouterr().err
    assert f"{run_v2} is incomplete" in err
    assert "missing: n2" in err
    assert not (tmp_path / "reports" / "report_v1_vs_v2.md").exists()


def test_compare_rejects_a_negative_max_new_misses(
    eval_path: Path, run_v1: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _compare(run_v1, run_v1, eval_path, "--max-new-misses", "-1")
    assert exit_info.value.code == 2
    assert "must be 0 or more" in capsys.readouterr().err


def test_compare_help_shows_the_gate_defaults(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PYTHON_COLORS", "0")  # no ANSI codes in the help text
    monkeypatch.setenv("COLUMNS", "200")  # no line wrapping
    with pytest.raises(SystemExit):
        cli.main(["compare", "--help"])
    out = capsys.readouterr().out
    assert "run_b misses (default: 0)" in out
    assert "below this (default: 0.95)" in out


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory with a 3-case eval set, a rubric and prices, as `run` expects."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "pricing.toml").write_text(PRICING_TOML, encoding="utf-8")
    (tmp_path / "rubric.md").write_text("Flag personal advice.", encoding="utf-8")
    write_cases(tmp_path / "eval_set.jsonl", CASES[:3])
    return tmp_path


RUN_ARGS = ["run", "--rubric", "rubric.md", "--name", "t1", "--eval", "eval_set.jsonl"]


def test_run_saves_predictions_with_an_injected_client(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = FakeClient(lambda _: make_message(assessment_input()))
    monkeypatch.setattr(cli, "make_client", lambda: client)
    assert cli.main(RUN_ARGS) == 0
    lines = (run_dir / "runs" / "t1" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(client.calls) == 3
    assert "3 completed, 0 failed" in capsys.readouterr().out


def test_run_returns_1_when_a_case_fails(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = make_message(assessment_input())
    answers = {"b1": ok, "b2": make_message(stop_reason="refusal"), "n1": ok}
    monkeypatch.setattr(cli, "make_client", lambda: FakeClient(answer_by_case(answers)))
    assert cli.main(RUN_ARGS) == 1
    assert "b2: ClassifierError" in capsys.readouterr().err


def test_run_stops_after_a_first_call_the_api_rejects(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def rejected(_kwargs: dict[str, Any]) -> Message:
        request = httpx2.Request("POST", "http://test.invalid/v1/messages")
        response = httpx2.Response(400, request=request)
        raise anthropic.BadRequestError("Invalid schema", response=response, body=None)

    client = FakeClient(rejected)
    monkeypatch.setattr(cli, "make_client", lambda: client)

    assert cli.main(RUN_ARGS) == 1
    assert len(client.calls) == 1
    assert not (run_dir / "runs" / "t1" / "predictions.jsonl").exists()
    out, err = capsys.readouterr()
    assert "3 cases, 0 already saved, 0 completed, 1 failed" in out
    assert (
        "Stopped after the first call: the API rejected the request "
        "(BadRequestError: Invalid schema), so the other 2 cases were not sent. "
        "Fix the cause and rerun the same command."
    ) in err


def test_run_without_a_key_points_to_env_example(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["run", "--rubric", "rubric.md", "--name", "t1"]) == 2
    assert ".env.example" in capsys.readouterr().err


def test_make_client_builds_the_sdk_client_with_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = cli.make_client()
    assert isinstance(client, anthropic.Anthropic)
    assert (client.max_retries, client.timeout) == (4, 120.0)
