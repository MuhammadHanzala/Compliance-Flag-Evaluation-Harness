from __future__ import annotations

from dataclasses import replace

from evalharness.compare import Comparison, compare, gate, mcnemar_exact
from evalharness.cost import summarise_cost
from evalharness.metrics import evaluate
from evalharness.report import render_report
from evalharness.schema import CATEGORIES, TRAPS, Prediction
from tests.fakes import breach_case, clean_case, make_prediction

CASES = [
    breach_case("b1"),
    breach_case("b2", "guarantee", borderline=True),
    clean_case("n1"),
    clean_case("n2", "referral_to_adviser", borderline=True),
]
SECTIONS = [
    "# v1 vs v2",
    "## Headline",
    "## Recall by breach type",
    "## False flags by trap",
    "## Changed cases",
    "### False flags fixed (1)",
    "### New false flags (1)",
    "### New misses (1)",
    "### Misses fixed (1)",
    "## McNemar exact test (reported, not gated)",
    "## Gate: FAIL",
    "## Cost",
    "## Run provenance",
]


def _run(flagged: set[str], reason: str = "Coach names a fund.") -> list[Prediction]:
    return [
        make_prediction(
            case.id,
            "breach" if case.id in flagged else "no_breach",
            tool_input={"reason": reason},
            input_tokens=1200,
            output_tokens=80,
            cache_read_input_tokens=900,
            cost_usd=0.0025,
        )
        for case in CASES
    ]


def _render(
    flagged_a: set[str],
    flagged_b: set[str],
    reason: str = "Coach names a fund.",
    comparison: Comparison | None = None,
) -> str:
    preds_a, preds_b = _run(flagged_a), _run(flagged_b, reason)
    metrics_b = evaluate(CASES, preds_b)
    comparison = comparison or compare(CASES, preds_a, preds_b)
    manifest = {"model": "claude-sonnet-5-5", "effort": "low", "eval_set_sha256": "cd" * 32}
    return render_report(
        name_a="v1",
        name_b="v2",
        cases=CASES,
        metrics_a=evaluate(CASES, preds_a),
        metrics_b=metrics_b,
        comparison=comparison,
        gate_result=gate(comparison, metrics_b),
        cost_a=summarise_cost(preds_a),
        cost_b=summarise_cost(preds_b),
        manifest_a={**manifest, "rubric_path": "prompts/rubric_v1.md", "rubric_sha256": "ab" * 32},
        manifest_b={**manifest, "rubric_path": "prompts/rubric_v2.md", "rubric_sha256": "ef" * 32},
        max_new_misses=0,
        recall_floor=0.95,
    )


def _line(report: str, prefix: str) -> str:
    return next(line for line in report.splitlines() if line.startswith(prefix))


def test_report_renders_every_section_in_order() -> None:
    report = _render({"b1", "n1"}, {"b2", "n2"})
    positions = [report.index(heading) for heading in SECTIONS]
    assert positions == sorted(positions)
    assert all(f"| {key} |" in report for key in (*CATEGORIES, *TRAPS))


def test_report_shows_metrics_cost_and_provenance() -> None:
    report = _render({"b1", "n1"}, {"b1", "b2"})
    assert _line(report, "| Recall [95% CI]").startswith("| Recall [95% CI] | 50.0% [")
    assert (
        _line(report, "| TP / FP / FN / TN")
        == "| TP / FP / FN / TN | 1 / 1 / 1 / 1 | 2 / 0 / 0 / 2 |"
    )
    assert _line(report, "| personal_advice") == "| personal_advice | 1 of 1 | 1 of 1 |"
    assert _line(report, "| Total cost (USD)") == "| Total cost (USD) | $0.0100 | $0.0100 |"
    assert _line(report, "| Cache read tokens") == "| Cache read tokens | 3,600 | 3,600 |"
    later = "| Later calls that read the cache | 3 of 3 | 3 of 3 |"
    assert _line(report, "| Later calls that read the cache") == later
    assert _line(report, "| Rubric sha256") == f"| Rubric sha256 | {'ab' * 6} | {'ef' * 6} |"


def test_report_shows_n_a_for_undefined_metrics() -> None:
    report = _render({"b1"}, set())  # v2 flags nothing, so its precision is undefined
    assert _line(report, "| Precision [95% CI]").endswith("| n/a [n/a] |")


def test_mcnemar_line_reads_cleanly() -> None:
    # One new miss, one new false flag, one false flag fixed, one miss fixed: p = 1.
    report = _render({"b1", "n1"}, {"b2", "n2"})
    assert "Cases v1 got right and v2 got wrong: 2. " in report
    assert "Two-sided exact p = 1.0000." in report

    lopsided = replace(
        compare(CASES, _run(set()), _run(set())),
        discordant_b=1,
        discordant_c=18,
        mcnemar_p=mcnemar_exact(1, 18),
    )
    report = _render(set(), set(), comparison=lopsided)
    assert "Cases v1 got right and v2 got wrong: 1. " in report
    assert "Cases v1 got wrong and v2 got right: 18. " in report
    assert "Two-sided exact p < 0.0001." in report


def test_report_contains_the_gate_verdict() -> None:
    failed = _render({"b1", "n1"}, {"b2", "n2"})
    assert "## Gate: FAIL" in failed
    assert "| New misses | at most 0 | 1 |" in failed
    assert "- new misses (breaches caught before, missed now): b1; 1 found, 0 allowed" in failed

    passed = _render({"b1", "n1"}, {"b1", "b2"})
    assert "## Gate: PASS" in passed
    assert "Why it failed" not in passed


def test_changed_case_reason_fits_one_table_cell() -> None:
    reason = "Coach says | pick\nthis fund. " + "x" * 300
    report = _render({"n1"}, set(), reason)
    row = _line(report, "| n1 |")
    assert row.startswith("| n1 | no_breach (general_education) | Coach says \\| pick this fund. ")
    assert row.endswith("\N{HORIZONTAL ELLIPSIS} |")
    assert len(row) < 220


def test_report_is_deterministic() -> None:
    assert _render({"b1", "n1"}, {"b2", "n2"}) == _render({"b1", "n1"}, {"b2", "n2"})
