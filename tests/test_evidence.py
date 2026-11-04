from __future__ import annotations

from dataclasses import replace

import pytest

from evalharness.classifier import Assessment
from evalharness.evidence import final_decision, normalise, quote_found
from evalharness.schema import Case, Category, Label, Turn

CASE = Case(
    id="c1",
    turns=(
        Turn(speaker="member", text="I'd move my super into the index fund, wouldn't you?"),
        Turn(speaker="coach", text="Honestly,  I\u2019d   move your SUPER into that\nindex fund."),
        Turn(speaker="member", text="Okay."),
    ),
    label="breach",
    category="personal_advice",
    trap=None,
    borderline=False,
    rationale="Coach recommends a specific fund.",
)


def _assessment(quote: str | None, index: int | None, decision: Label = "breach") -> Assessment:
    category: Category | None = "personal_advice" if decision == "breach" else None
    return Assessment(
        decision=decision,
        category=category,
        evidence_quote=quote,
        coach_turn_index=index,
        reason="Test.",
    )


def test_normalise_folds_case_whitespace_and_curly_quotes() -> None:
    assert normalise("  I\u2019d  SAY \u201cYes\u201d\n now ") == 'i\'d say "yes" now'


def test_quote_matches_despite_case_whitespace_and_curly_quotes() -> None:
    assert quote_found(CASE, _assessment("i'd move your super into that index fund", 1))


def test_quote_from_a_member_turn_fails() -> None:
    assert not quote_found(CASE, _assessment("I'd move my super into the index fund", 0))


def test_quote_not_in_the_named_turn_fails() -> None:
    assert not quote_found(CASE, _assessment("switch to the growth option", 1))


@pytest.mark.parametrize("index", [None, 3, 99])
def test_bad_turn_index_fails(index: int | None) -> None:
    assert not quote_found(CASE, _assessment("index fund", index))


def test_negative_turn_index_fails_even_when_the_last_turn_is_the_coach() -> None:
    case = replace(CASE, turns=CASE.turns[:2])
    assert quote_found(case, _assessment("index fund", 1))
    # Python would read turns[-1] as the coach turn; the model must name it by its real index.
    assert not quote_found(case, _assessment("index fund", -1))


@pytest.mark.parametrize("quote", [None, "", "   "])
def test_missing_or_blank_quote_fails(quote: str | None) -> None:
    assert not quote_found(CASE, _assessment(quote, 1))


@pytest.mark.parametrize(
    ("decision", "evidence_ok", "expected"),
    [
        ("breach", True, "breach"),
        ("breach", False, "needs_review"),
        ("no_breach", True, "no_breach"),
        ("no_breach", False, "no_breach"),
    ],
)
def test_final_decision(decision: Label, evidence_ok: bool, expected: str) -> None:
    assert final_decision(_assessment(None, None, decision), evidence_ok) == expected


def test_failed_evidence_never_turns_a_flag_into_no_breach() -> None:
    flag = _assessment("words the coach never said", 1)
    evidence_ok = quote_found(CASE, flag)
    assert not evidence_ok
    assert final_decision(flag, evidence_ok) == "needs_review"
