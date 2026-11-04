from __future__ import annotations

from dataclasses import replace

import pytest

from evalharness.metrics import (
    Confusion,
    EvidenceStats,
    IncompleteRunError,
    Metric,
    agreement,
    bootstrap_ci,
    both_differ_from_draft,
    cohen_kappa,
    confusion,
    evaluate,
    evidence_stats,
    f1,
    false_flags_by_trap,
    false_positive_rate,
    pair,
    precision,
    recall,
    recall_by_category,
)
from evalharness.schema import CATEGORIES, TRAPS, Case, LabelRow, Prediction
from tests.fakes import breach_case, clean_case, make_prediction


def _hand_counted_run() -> tuple[list[Case], list[Prediction]]:
    """3 TP, 2 FP, 1 FN, 4 TN."""
    cases = [breach_case(f"b{i}") for i in range(4)] + [clean_case(f"n{i}") for i in range(6)]
    flagged = {"b0", "b1", "b2", "n0", "n1"}
    preds = [make_prediction(c.id, "breach" if c.id in flagged else "no_breach") for c in cases]
    return cases, preds


def test_metrics_match_a_hand_computed_confusion() -> None:
    c = confusion(pair(*_hand_counted_run()))
    assert c == Confusion(tp=3, fp=2, fn=1, tn=4)
    assert precision(c) == pytest.approx(3 / 5)
    assert recall(c) == pytest.approx(3 / 4)
    assert f1(c) == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
    assert false_positive_rate(c) == pytest.approx(2 / 6)


@pytest.mark.parametrize("metric", [precision, recall, f1, false_positive_rate])
def test_metrics_are_none_when_the_denominator_is_zero(metric: Metric) -> None:
    assert metric(Confusion(tp=0, fp=0, fn=0, tn=0)) is None


def test_metrics_with_a_zero_denominator_do_not_hide_the_others() -> None:
    no_breaches_nothing_flagged = Confusion(tp=0, fp=0, fn=0, tn=5)
    assert precision(no_breaches_nothing_flagged) is None
    assert recall(no_breaches_nothing_flagged) is None
    assert false_positive_rate(no_breaches_nothing_flagged) == 0.0


def test_needs_review_counts_as_a_flag() -> None:
    cases = [breach_case("b1"), clean_case("n1")]
    preds = [make_prediction("b1", "needs_review"), make_prediction("n1", "needs_review")]
    assert confusion(pair(cases, preds)) == Confusion(tp=1, fp=1, fn=0, tn=0)


def test_pair_rejects_a_missing_case() -> None:
    cases = [breach_case("b1"), clean_case("n1")]
    with pytest.raises(IncompleteRunError, match="missing: n1"):
        pair(cases, [make_prediction("b1")])


def test_pair_rejects_extra_and_duplicate_predictions() -> None:
    cases = [breach_case("b1")]
    with pytest.raises(IncompleteRunError, match="not in the eval set: x9"):
        pair(cases, [make_prediction("b1"), make_prediction("x9")])
    with pytest.raises(IncompleteRunError, match="predicted more than once: b1"):
        pair(cases, [make_prediction("b1"), make_prediction("b1")])


def test_recall_by_category_lists_every_category() -> None:
    cases = [breach_case("b1", "guarantee"), breach_case("b2", "guarantee"), clean_case("n1")]
    preds = [
        make_prediction("b1", "breach"),
        make_prediction("b2"),
        make_prediction("n1", "breach"),
    ]
    by_category = recall_by_category(pair(cases, preds))
    assert list(by_category) == list(CATEGORIES)
    assert by_category["guarantee"] == (1, 2)
    assert by_category["tax_legal"] == (0, 0)


def test_false_flags_by_trap_lists_every_trap() -> None:
    cases = [
        clean_case("n1", "budgeting_action"),
        clean_case("n2", "budgeting_action"),
        breach_case("b1"),
    ]
    preds = [
        make_prediction("n1", "needs_review"),
        make_prediction("n2"),
        make_prediction("b1", "breach"),
    ]
    by_trap = false_flags_by_trap(pair(cases, preds))
    assert list(by_trap) == list(TRAPS)
    assert by_trap["budgeting_action"] == (1, 2)
    assert by_trap["repeating_member"] == (0, 0)


def test_evidence_stats_count_failed_flags_and_false_flags_among_them() -> None:
    cases = [
        breach_case("b1"),
        breach_case("b2"),
        clean_case("n1"),
        clean_case("n2"),
        clean_case("n3"),
    ]
    preds = [
        make_prediction("b1", "needs_review", evidence_failed=True),
        make_prediction("b2", "breach"),
        make_prediction("n1", "needs_review", evidence_failed=True),
        make_prediction("n2", "breach"),
        make_prediction("n3"),
    ]
    assert evidence_stats(pair(cases, preds)) == EvidenceStats(
        flags=4, failed=2, failed_false_flags=1
    )


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate() -> None:
    pairs = list(zip(*_hand_counted_run(), strict=True)) * 4
    point = precision(confusion(pairs))
    interval = bootstrap_ci(pairs, precision)
    assert interval == bootstrap_ci(pairs, precision)
    assert point is not None
    assert interval is not None
    low, high = interval
    assert low < point < high


def test_bootstrap_ci_is_none_when_the_metric_is_never_defined() -> None:
    pairs = pair(
        [clean_case("n1"), clean_case("n2")], [make_prediction("n1"), make_prediction("n2")]
    )
    assert bootstrap_ci(pairs, recall) is None
    assert bootstrap_ci([], recall) is None


def test_evaluate_fills_every_field() -> None:
    cases, preds = _hand_counted_run()
    for i in (0, 4, 9):  # b0 caught, n0 falsely flagged, n5 correctly left alone
        cases[i] = replace(cases[i], borderline=True)
    result = evaluate(cases, preds)
    assert result.confusion == Confusion(tp=3, fp=2, fn=1, tn=4)
    assert result.precision == pytest.approx(0.6)
    assert result.recall_ci is not None
    assert result.borderline_correct == (2, 3)
    assert result.evidence == EvidenceStats(flags=5, failed=0, failed_false_flags=0)


def test_cohen_kappa_on_a_textbook_table() -> None:
    # 50 items: both yes 20, A yes / B no 5, A no / B yes 10, both no 15.
    # p_o = 0.70, p_e = 0.5 * 0.6 + 0.5 * 0.4 = 0.50, kappa = (0.70 - 0.50) / 0.50 = 0.40
    a = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_is_one_for_identical_labels() -> None:
    labels = ["breach", "no_breach", "no_breach", "breach"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)
    assert cohen_kappa(["no_breach"] * 3, ["no_breach"] * 3) == 1.0


def test_cohen_kappa_rejects_mismatched_lists() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohen_kappa(["breach"], ["breach", "no_breach"])


def test_agreement_reports_kappas_and_lists_disagreements() -> None:
    rows_a = [
        LabelRow("c1", "breach", "personal_advice"),
        LabelRow("c2", "breach", "guarantee"),
        LabelRow("c3", "no_breach", None),
        LabelRow("c4", "no_breach", None),
    ]
    rows_b = [
        LabelRow("c1", "breach", "personal_advice"),
        LabelRow("c2", "breach", "tax_legal"),
        LabelRow("c3", "breach", "guarantee"),
        LabelRow("c4", "no_breach", None),
    ]
    result = agreement(rows_a, rows_b)
    assert result.n == 4
    assert result.raw_agreement == pytest.approx(0.75)
    # Labels: p_o = 3/4, p_e = 0.5 * 0.75 + 0.5 * 0.25 = 0.5 -> kappa 0.5.
    assert result.kappa_label == pytest.approx(0.5)
    # Five classes: p_o = 2/4, p_e = 1/16 + 1/16 + 2/16 = 0.25 -> kappa 1/3.
    assert result.kappa_category == pytest.approx(1 / 3)
    assert result.disagreements == [
        ("c2", "guarantee", "tax_legal"),
        ("c3", "no_breach", "guarantee"),
    ]


def test_agreement_rejects_different_case_ids() -> None:
    with pytest.raises(ValueError, match="different cases: c2, c3"):
        agreement(
            [LabelRow("c1", "no_breach", None), LabelRow("c2", "no_breach", None)],
            [LabelRow("c1", "no_breach", None), LabelRow("c3", "no_breach", None)],
        )


def test_both_differ_from_draft_lists_cases_the_labellers_agree_on_against_the_draft() -> None:
    cases = [
        clean_case("c1"),
        breach_case("c2", "tax_legal"),
        breach_case("c3", "guarantee"),
        clean_case("c4"),
    ]
    rows_a = [
        LabelRow("c1", "breach", "guarantee"),  # both differ from the draft
        LabelRow("c2", "breach", "tax_legal"),  # A and B disagree: an ordinary disagreement
        LabelRow("c3", "breach", "personal_advice"),  # both differ on the breach type
        LabelRow("c4", "no_breach", None),  # both match the draft
    ]
    rows_b = [
        LabelRow("c1", "breach", "guarantee"),
        LabelRow("c2", "no_breach", None),
        LabelRow("c3", "breach", "personal_advice"),
        LabelRow("c4", "no_breach", None),
    ]
    assert both_differ_from_draft(cases, rows_a, rows_b) == [
        ("c1", "no_breach", "guarantee"),
        ("c3", "guarantee", "personal_advice"),
    ]


def test_both_differ_from_draft_rejects_labels_that_do_not_cover_the_set() -> None:
    rows = [LabelRow("c1", "no_breach", None), LabelRow("c9", "no_breach", None)]
    with pytest.raises(ValueError, match="missing: c2; not in the eval set: c9"):
        both_differ_from_draft([clean_case("c1"), clean_case("c2")], rows, rows)
