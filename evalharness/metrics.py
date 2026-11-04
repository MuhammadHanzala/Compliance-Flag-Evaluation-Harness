"""Metrics for one run against the gold labels, and agreement between two labellers."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from evalharness.schema import CATEGORIES, TRAPS, Case, LabelRow, Prediction

Pair = tuple[Case, Prediction]


class IncompleteRunError(ValueError):
    """The predictions do not cover the eval set one-to-one."""


@dataclass(frozen=True)
class Confusion:
    """Positive = gold breach. Predicted positive = flagged (breach or needs_review)."""

    tp: int
    fp: int
    fn: int
    tn: int


Metric = Callable[[Confusion], float | None]


@dataclass(frozen=True)
class EvidenceStats:
    flags: int
    failed: int
    failed_false_flags: int


@dataclass(frozen=True)
class RunMetrics:
    confusion: Confusion
    precision: float | None
    recall: float | None
    f1: float | None
    fpr: float | None
    precision_ci: tuple[float, float] | None
    recall_ci: tuple[float, float] | None
    by_category: dict[str, tuple[int, int]]
    by_trap: dict[str, tuple[int, int]]
    evidence: EvidenceStats
    borderline_correct: tuple[int, int]


def pair(cases: Sequence[Case], predictions: Sequence[Prediction]) -> list[Pair]:
    """Match each case to its one prediction; anything missing, extra or doubled is an error."""
    counts = Counter(p.case_id for p in predictions)
    case_ids = {c.id for c in cases}
    problems = {
        "missing": [c.id for c in cases if c.id not in counts],
        "not in the eval set": sorted(set(counts) - case_ids),
        "predicted more than once": sorted(i for i, n in counts.items() if n > 1),
    }
    errors = [f"{what}: {', '.join(ids)}" for what, ids in problems.items() if ids]
    if errors:
        raise IncompleteRunError("predictions do not match the eval set; " + "; ".join(errors))
    by_id = {p.case_id: p for p in predictions}
    return [(case, by_id[case.id]) for case in cases]


def confusion(pairs: Sequence[Pair]) -> Confusion:
    counts = Counter((case.is_breach, pred.flagged) for case, pred in pairs)
    return Confusion(
        tp=counts[True, True],
        fp=counts[False, True],
        fn=counts[True, False],
        tn=counts[False, False],
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def precision(c: Confusion) -> float | None:
    return _ratio(c.tp, c.tp + c.fp)


def recall(c: Confusion) -> float | None:
    return _ratio(c.tp, c.tp + c.fn)


def f1(c: Confusion) -> float | None:
    return _ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn)


def false_positive_rate(c: Confusion) -> float | None:
    return _ratio(c.fp, c.fp + c.tn)


def recall_by_category(pairs: Sequence[Pair]) -> dict[str, tuple[int, int]]:
    """(caught, total) breaches per category, every category listed even when it has none."""
    result: dict[str, tuple[int, int]] = {}
    for category in CATEGORIES:
        preds = [pred for case, pred in pairs if case.is_breach and case.category == category]
        result[category] = (sum(p.flagged for p in preds), len(preds))
    return result


def false_flags_by_trap(pairs: Sequence[Pair]) -> dict[str, tuple[int, int]]:
    """(false flags, total) non-breaches per trap, every trap listed even when it has none."""
    result: dict[str, tuple[int, int]] = {}
    for trap in TRAPS:
        preds = [pred for case, pred in pairs if not case.is_breach and case.trap == trap]
        result[trap] = (sum(p.flagged for p in preds), len(preds))
    return result


def evidence_stats(pairs: Sequence[Pair]) -> EvidenceStats:
    """How many flags failed the evidence check, and how many of those were false flags."""
    failed = [case for case, pred in pairs if pred.evidence_failed]
    return EvidenceStats(
        flags=sum(pred.flagged for _, pred in pairs),
        failed=len(failed),
        failed_false_flags=sum(not case.is_breach for case in failed),
    )


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation between the closest ranks of an ordered list."""
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_ci(
    pairs: Sequence[Pair], metric: Metric, *, n: int = 1000, seed: int = 20261103
) -> tuple[float, float] | None:
    """95% percentile bootstrap interval; resamples where the metric is undefined are skipped."""
    if not pairs:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n):
        value = metric(confusion(rng.choices(pairs, k=len(pairs))))
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return _percentile(values, 0.025), _percentile(values, 0.975)


def _borderline_correct(pairs: Sequence[Pair]) -> tuple[int, int]:
    borderline = [(case, pred) for case, pred in pairs if case.borderline]
    return sum(pred.flagged == case.is_breach for case, pred in borderline), len(borderline)


def evaluate(cases: Sequence[Case], predictions: Sequence[Prediction]) -> RunMetrics:
    pairs = pair(cases, predictions)
    c = confusion(pairs)
    return RunMetrics(
        confusion=c,
        precision=precision(c),
        recall=recall(c),
        f1=f1(c),
        fpr=false_positive_rate(c),
        precision_ci=bootstrap_ci(pairs, precision),
        recall_ci=bootstrap_ci(pairs, recall),
        by_category=recall_by_category(pairs),
        by_trap=false_flags_by_trap(pairs),
        evidence=evidence_stats(pairs),
        borderline_correct=_borderline_correct(pairs),
    )


@dataclass(frozen=True)
class Agreement:
    """Label agreement compares breach/no_breach; the category view uses five classes."""

    n: int
    raw_agreement: float
    kappa_label: float
    kappa_category: float
    disagreements: list[tuple[str, str, str]]  # (case id, labeller A, labeller B), five-class


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two labellers rating the same items in the same order.

    When both labellers give every item the same single class, chance agreement is 1 and the
    formula divides by zero; that is perfect agreement, so it returns 1.0.
    """
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two non-empty label lists of the same length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    counts_a, counts_b = Counter(a), Counter(b)
    chance = sum(counts_a[k] * counts_b[k] for k in counts_a) / (n * n)
    if chance == 1:
        return 1.0
    return (observed - chance) / (1 - chance)


def _five_class(item: Case | LabelRow) -> str:
    """The breach category for a breach, otherwise "no_breach"."""
    return item.category or item.label


def agreement(rows_a: Sequence[LabelRow], rows_b: Sequence[LabelRow]) -> Agreement:
    by_id_a, by_id_b = {r.id: r for r in rows_a}, {r.id: r for r in rows_b}
    if len(by_id_a) != len(rows_a) or len(by_id_b) != len(rows_b):
        raise ValueError("a labeller CSV lists the same case id twice")
    if by_id_a.keys() != by_id_b.keys():
        only = sorted(by_id_a.keys() ^ by_id_b.keys())
        raise ValueError(f"the two labellers labelled different cases: {', '.join(only)}")
    ids = sorted(by_id_a)
    if not ids:
        raise ValueError("there are no labels to compare")
    labels_a = [by_id_a[i].label for i in ids]
    labels_b = [by_id_b[i].label for i in ids]
    classes_a = [_five_class(by_id_a[i]) for i in ids]
    classes_b = [_five_class(by_id_b[i]) for i in ids]
    return Agreement(
        n=len(ids),
        raw_agreement=sum(x == y for x, y in zip(labels_a, labels_b, strict=True)) / len(ids),
        kappa_label=cohen_kappa(labels_a, labels_b),
        kappa_category=cohen_kappa(classes_a, classes_b),
        disagreements=[
            (i, x, y) for i, x, y in zip(ids, classes_a, classes_b, strict=True) if x != y
        ],
    )


def _check_covers(cases: Sequence[Case], rows: Sequence[LabelRow]) -> None:
    case_ids, label_ids = {c.id for c in cases}, {r.id for r in rows}
    problems = {
        "missing": [c.id for c in cases if c.id not in label_ids],
        "not in the eval set": sorted(label_ids - case_ids),
    }
    errors = [f"{what}: {', '.join(ids)}" for what, ids in problems.items() if ids]
    if errors:
        raise ValueError("the labels do not cover the eval set; " + "; ".join(errors))


def both_differ_from_draft(
    cases: Sequence[Case], rows_a: Sequence[LabelRow], rows_b: Sequence[LabelRow]
) -> list[tuple[str, str, str]]:
    """(case id, drafted class, both labellers' class) where A and B agree but not with the draft.

    These are not A-vs-B disagreements, yet they must be settled too, or the gold label stays one
    neither labeller chose. Both labellers must cover the eval set exactly (ValueError otherwise).
    """
    _check_covers(cases, rows_a)
    _check_covers(cases, rows_b)
    by_id_a, by_id_b = {r.id: r for r in rows_a}, {r.id: r for r in rows_b}
    found: list[tuple[str, str, str]] = []
    for case in cases:
        a, b = _five_class(by_id_a[case.id]), _five_class(by_id_b[case.id])
        if a == b != _five_class(case):
            found.append((case.id, _five_class(case), a))
    return found
