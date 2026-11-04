"""Test doubles: a fake Claude client, plus builders for SDK messages, cases and predictions."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from anthropic.types import Message

from evalharness.classifier import DEFAULT_MODEL, TOOL_NAME
from evalharness.cost import Prices
from evalharness.schema import Case, Category, Decision, Label, Prediction, Trap, Turn

SONNET_PRICES = Prices(
    input_per_mtok=2.0,
    output_per_mtok=10.0,
    cache_write_multiplier=1.25,
    cache_read_multiplier=0.05,
)
# SONNET_PRICES as a pricing.toml file states them.
PRICING_TOML = """
[models.claude-sonnet-5-5]
input_per_mtok = 2.0
output_per_mtok = 10.0
cache_write_multiplier = 1.25
cache_read_multiplier = 0.05
"""

Responder = Callable[[dict[str, Any]], Message]

DEFAULT_USAGE: dict[str, int] = {
    "input_tokens": 120,
    "output_tokens": 40,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def assessment_input(
    decision: str = "no_breach",
    *,
    category: str | None = None,
    evidence_quote: str | None = None,
    coach_turn_index: int | None = None,
    reason: str = "Test reason.",
) -> dict[str, Any]:
    """A record_assessment tool input."""
    return {
        "decision": decision,
        "category": category,
        "evidence_quote": evidence_quote,
        "coach_turn_index": coach_turn_index,
        "reason": reason,
    }


def make_message(
    tool_input: dict[str, Any] | None = None,
    *,
    stop_reason: str = "tool_use",
    usage: dict[str, int] | None = None,
    text: str | None = None,
    thinking: bool = False,
    stop_details: dict[str, Any] | None = None,
) -> Message:
    """A real anthropic Message, validated by the SDK's own model."""
    content: list[dict[str, Any]] = []
    if thinking:
        content.append({"type": "thinking", "thinking": "", "signature": "sig-test"})
    if text is not None:
        content.append({"type": "text", "text": text})
    if tool_input is not None:
        content.append(
            {"type": "tool_use", "id": "toolu_test", "name": TOOL_NAME, "input": tool_input}
        )
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": DEFAULT_MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": usage or DEFAULT_USAGE,
        }
    )


class FakeMessages:
    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        with self._lock:
            self.calls.append(kwargs)
        return self._responder(kwargs)


class FakeClient:
    """Stands in for anthropic.Anthropic: records each create() call, answers via `responder`."""

    def __init__(self, responder: Responder) -> None:
        self.messages = FakeMessages(responder)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


_CASE_ID = re.compile(r"\(case (\S+)\)")


def make_case(
    case_id: str,
    *,
    label: Label = "no_breach",
    category: Category = "personal_advice",
    coach_text: str = "Let's set a weekly spending limit.",
) -> Case:
    """A valid three-turn case whose member turn carries the id, so `answer_by_case` can route it."""
    return Case(
        id=case_id,
        turns=(
            Turn(speaker="member", text=f"(case {case_id}) What should I do with my savings?"),
            Turn(speaker="coach", text=coach_text),
            Turn(speaker="member", text="Thanks."),
        ),
        label=label,
        category=category if label == "breach" else None,
        trap=None if label == "breach" else "budgeting_action",
        borderline=False,
        rationale="Test case.",
    )


def answer_by_case(answers: Mapping[str, Message]) -> Responder:
    """A responder that returns answers[case id], reading the id from a make_case transcript."""

    def respond(kwargs: dict[str, Any]) -> Message:
        match = _CASE_ID.search(kwargs["messages"][0]["content"])
        if match is None or match.group(1) not in answers:
            raise AssertionError(f"no scripted answer for this request: {kwargs['messages']!r}")
        return answers[match.group(1)]

    return respond


def breach_case(
    case_id: str, category: Category = "personal_advice", *, borderline: bool = False
) -> Case:
    """A make_case breach, for tests where only the label, type and borderline flag matter."""
    return replace(make_case(case_id, label="breach", category=category), borderline=borderline)


def clean_case(case_id: str, trap: Trap = "general_education", *, borderline: bool = False) -> Case:
    """A make_case non-breach set up as the given trap."""
    return replace(make_case(case_id), trap=trap, borderline=borderline)


def write_cases(path: Path, cases: list[Case]) -> Path:
    """Save cases as an eval-set JSONL file."""
    path.write_text("".join(json.dumps(asdict(case)) + "\n" for case in cases), encoding="utf-8")
    return path


def make_prediction(case_id: str, decision: Decision = "no_breach", **overrides: Any) -> Prediction:
    """A saved prediction with zero tokens, cost and latency; `overrides` set any other field."""
    base = Prediction(
        case_id=case_id,
        decision=decision,
        category=None,
        evidence_failed=False,
        tool_input={},
        model=DEFAULT_MODEL,
        rubric_sha256="0" * 64,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        cost_usd=0.0,
        latency_s=0.0,
        stop_reason="tool_use",
    )
    return replace(base, **overrides)
