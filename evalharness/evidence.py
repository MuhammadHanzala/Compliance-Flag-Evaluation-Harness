"""Check that each flag quotes something the coach actually said."""

from __future__ import annotations

import unicodedata

from evalharness.classifier import Assessment
from evalharness.schema import Case, Decision

# NFKC does not touch curly quotes, so map them to straight ones explicitly.
_STRAIGHT_QUOTES = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
    }
)


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).translate(_STRAIGHT_QUOTES).casefold()
    return " ".join(folded.split())


def quote_found(case: Case, assessment: Assessment) -> bool:
    """True if the evidence quote appears in the coach turn the assessment points at."""
    index, quote = assessment.coach_turn_index, assessment.evidence_quote
    if index is None or quote is None or not 0 <= index < len(case.turns):
        return False
    turn = case.turns[index]
    needle = normalise(quote)
    return turn.speaker == "coach" and bool(needle) and needle in normalise(turn.text)


def final_decision(assessment: Assessment, evidence_ok: bool) -> Decision:
    if assessment.decision == "no_breach":
        return "no_breach"
    # A flag with bad evidence goes to a human. Dropping it to no_breach could hide a real breach.
    return "breach" if evidence_ok else "needs_review"
