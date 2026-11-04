from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalharness.compare import check_same_set, compare, gate, load_run, mcnemar_exact
from evalharness.metrics import IncompleteRunError, evaluate
from evalharness.schema import Case, Prediction, sha256_file
from tests.fakes import breach_case, clean_case, make_prediction

BREACH_IDS = frozenset(f"b{i:02}" for i in range(1, 21))


def _eval_set() -> list[Case]:
    """20 breaches and 40 non-breaches, like the real set."""
    return [breach_case(i) for i in sorted(BREACH_IDS)] + [
        clean_case(f"n{i:02}") for i in range(1, 41)
    ]


def _run(cases: list[Case], flagged: set[str] | frozenset[str]) -> list[Prediction]:
    return [
        make_prediction(
            case.id,
            "breach" if case.id in flagged else "no_breach",
            tool_input={"reason": f"Reason for {case.id}."},
        )
        for case in cases
    ]


def test_compare_sorts_each_kind_of_change() -> None:
    cases = [
        breach_case("b1"),
        breach_case("b2"),
        breach_case("b3"),
        clean_case("n1"),
        clean_case("n2"),
        clean_case("n3"),
    ]
    run_a = _run(cases, {"b1", "b3", "n1"})
    run_b = _run(cases, {"b2", "b3", "n2"})
    result = compare(cases, run_a, run_b)
    assert result.false_flags_fixed == ["n1"]
    assert result.new_false_flags == ["n2"]
    assert result.new_misses == ["b1"]
    assert result.misses_fixed == ["b2"]
    assert (result.discordant_b, result.discordant_c) == (2, 2)
    assert result.mcnemar_p == 1.0
    assert result.reasons_b == {i: f"Reason for {i}." for i in ("b1", "b2", "n1", "n2")}


def test_compare_rejects_an_incomplete_run() -> None:
    cases = _eval_set()
    with pytest.raises(IncompleteRunError, match="missing: n40"):
        compare(cases, _run(cases, BREACH_IDS), _run(cases[:-1], BREACH_IDS))


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (0, 0, 1.0),
        (5, 5, 1.0),
        (0, 5, 2 / 2**5),  # 0.0625
        (1, 9, 2 * 11 / 2**10),  # 0.021484375
    ],
)
def test_mcnemar_exact_known_values(b: int, c: int, expected: float) -> None:
    assert mcnemar_exact(b, c) == pytest.approx(expected)


def test_mcnemar_exact_is_symmetric() -> None:
    assert mcnemar_exact(1, 9) == mcnemar_exact(9, 1) == pytest.approx(0.02148, abs=1e-5)


def test_gate_fails_when_a_new_miss_appears() -> None:
    # A misses b01; B catches b01 but newly misses b02. Recall is 19/20 for both, right
    # on the 95% floor, so only the new miss can fail the gate.
    cases = _eval_set()
    run_a = _run(cases, BREACH_IDS - {"b01"})
    run_b = _run(cases, BREACH_IDS - {"b02"})
    metrics_b = evaluate(cases, run_b)
    assert metrics_b.recall == pytest.approx(0.95)

    result = gate(compare(cases, run_a, run_b), metrics_b)

    assert not result.passed
    assert len(result.reasons) == 1
    assert "b02" in result.reasons[0]


def test_gate_passes_with_no_new_misses_and_recall_at_the_floor() -> None:
    cases = _eval_set()
    run_a = _run(cases, (BREACH_IDS - {"b01"}) | {"n01", "n02"})
    run_b = _run(cases, BREACH_IDS - {"b01"})
    result = gate(compare(cases, run_a, run_b), evaluate(cases, run_b))
    assert result.passed
    assert result.reasons == []


def test_gate_fails_on_recall_below_the_floor_even_with_no_new_misses() -> None:
    cases = _eval_set()
    run = _run(cases, BREACH_IDS - {"b01", "b02"})  # 18/20 = 90%, the same in A and B
    result = gate(compare(cases, run, run), evaluate(cases, run))
    assert not result.passed
    assert result.reasons == ["recall 90.0% is below the floor of 95.0%"]


def test_gate_fails_when_recall_is_undefined() -> None:
    cases = [clean_case("n1")]
    run = _run(cases, set())
    result = gate(compare(cases, run, run), evaluate(cases, run))
    assert not result.passed
    assert "undefined" in result.reasons[0]


@pytest.mark.parametrize(("max_new_misses", "passed"), [(0, False), (1, True), (2, True)])
def test_max_new_misses_tolerates_that_many(max_new_misses: int, passed: bool) -> None:
    cases = _eval_set()
    run_a = _run(cases, BREACH_IDS)
    run_b = _run(cases, BREACH_IDS - {"b03"})
    result = gate(
        compare(cases, run_a, run_b),
        evaluate(cases, run_b),
        max_new_misses=max_new_misses,
        recall_floor=0.9,
    )
    assert result.passed is passed


def _manifest(eval_sha: str) -> dict[str, str]:
    return {"name": "v1", "eval_set_sha256": eval_sha}


def _write_eval_set(tmp_path: Path, text: str) -> tuple[Path, str]:
    path = tmp_path / "eval_set.jsonl"
    path.write_text(text, encoding="utf-8")
    return path, sha256_file(path)


def test_check_same_set_accepts_runs_on_the_current_set(tmp_path: Path) -> None:
    eval_path, sha = _write_eval_set(tmp_path, '{"id": "c1"}\n')
    check_same_set(_manifest(sha), _manifest(sha), eval_path)


def test_check_same_set_rejects_runs_on_different_eval_sets(tmp_path: Path) -> None:
    eval_path, sha = _write_eval_set(tmp_path, '{"id": "c1"}\n')
    with pytest.raises(ValueError, match="different eval sets"):
        check_same_set(_manifest(sha), _manifest("f" * 64), eval_path)


def test_check_same_set_rejects_an_eval_file_changed_since_the_runs(tmp_path: Path) -> None:
    eval_path, sha = _write_eval_set(tmp_path, '{"id": "c1"}\n')
    eval_path.write_text('{"id": "c2"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="has changed since the runs were made"):
        check_same_set(_manifest(sha), _manifest(sha), eval_path)


def test_load_run_reads_the_manifest_and_predictions(tmp_path: Path) -> None:
    cases = [breach_case("b1"), clean_case("n1")]
    preds = _run(cases, {"b1"})
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest("a" * 64)), encoding="utf-8")
    lines = "".join(p.to_json() + "\n" for p in preds)
    (tmp_path / "predictions.jsonl").write_text(lines, encoding="utf-8")
    assert load_run(tmp_path) == (_manifest("a" * 64), preds)


def test_load_run_rejects_a_directory_without_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no manifest.json"):
        load_run(tmp_path)
