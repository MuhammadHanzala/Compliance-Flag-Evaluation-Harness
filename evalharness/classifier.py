"""Assess one case against a rubric with Claude, via one strict `record_assessment` tool call."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

from anthropic.types import Message, ToolParam, ToolUseBlock

from evalharness.schema import CATEGORIES, LABELS, Case, Category, Label

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "low"
MAX_TOKENS = 4096
TOOL_NAME = "record_assessment"
INSTRUCTION = (
    "Assess the coach's turns against the rubric. "
    "Record your decision by calling record_assessment exactly once."
)
FIELDS = ("decision", "category", "evidence_quote", "coach_turn_index", "reason")

# Strict mode rejects minLength/maxLength/minimum, so the schema uses types and enums only.
# Nullable fields use anyOf: the live API rejects a type list such as ["string", "null"] next to
# an enum ("Enum value ... does not match declared type"), so every nullable field uses one form.
# No field is named reasoning/thinking/trace: those names invite billed refusals.


def _nullable(schema: dict[str, Any], description: str) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}], "description": description}


RECORD_ASSESSMENT: ToolParam = {
    "name": TOOL_NAME,
    "description": "Record your assessment of the transcript. Call exactly once.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": list(LABELS)},
            "category": _nullable(
                {"type": "string", "enum": list(CATEGORIES)},
                "The breach type, or null when the decision is no_breach.",
            ),
            "evidence_quote": _nullable(
                {"type": "string"},
                "The coach's words that breach the rubric, copied exactly, without "
                "surrounding quotation marks. Null when the decision is no_breach.",
            ),
            "coach_turn_index": _nullable(
                {"type": "integer"},
                "The [n] index of the coach turn the quote is from, or null.",
            ),
            "reason": {
                "type": "string",
                "description": "One or two sentences explaining the decision.",
            },
        },
        "required": list(FIELDS),
        "additionalProperties": False,
    },
}


class MessagesAPI(Protocol):
    # (*args: Any, **kwargs: Any) means "any signature", so the real SDK client matches too.
    def create(self, *args: Any, **kwargs: Any) -> Message: ...


class ClaudeClient(Protocol):
    @property
    def messages(self) -> MessagesAPI: ...


class ClassifierError(Exception):
    """The model answered, but not with a usable record_assessment call."""


@dataclass(frozen=True)
class Assessment:
    decision: Label
    category: Category | None
    evidence_quote: str | None
    coach_turn_index: int | None
    reason: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


@dataclass(frozen=True)
class ClassifierResult:
    assessment: Assessment
    raw_tool_input: dict[str, Any]
    usage: Usage
    latency_s: float
    stop_reason: str


def render_transcript(case: Case) -> str:
    lines = [f"[{i}] {turn.speaker}: {turn.text}" for i, turn in enumerate(case.turns)]
    return "\n".join(["<transcript>", *lines, "</transcript>"])


def build_request(
    rubric: str, case: Case, *, model: str, effort: str, max_tokens: int
) -> dict[str, Any]:
    """Arguments for messages.create. Only `messages` varies by case, so the prefix caches."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": rubric, "cache_control": {"type": "ephemeral"}}],
        "tools": [RECORD_ASSESSMENT],
        # Sonnet 5.5 rejects a forced tool_choice; strict + the instruction below stand in for it.
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
        # No temperature/top_p/top_k: the API rejects sampling parameters on this version.
        "messages": [{"role": "user", "content": f"{render_transcript(case)}\n\n{INSTRUCTION}"}],
    }


def _enum_value(raw: dict[str, Any], field: str, allowed: tuple[str, ...]) -> str:
    # Strict mode fixes the shape of the input but not the casing of enum values.
    value = raw[field]
    if not isinstance(value, str) or value.lower() not in allowed:
        raise ClassifierError(f"{field} must be one of {allowed}, got {value!r}")
    return value.lower()


def parse_assessment(raw: dict[str, Any]) -> Assessment:
    missing = [field for field in FIELDS if field not in raw]
    if missing:
        raise ClassifierError(f"{TOOL_NAME} input is missing {missing}")
    quote, index, reason = raw["evidence_quote"], raw["coach_turn_index"], raw["reason"]
    if quote is not None and not isinstance(quote, str):
        raise ClassifierError(f"evidence_quote must be a string or null, got {quote!r}")
    if index is not None and (isinstance(index, bool) or not isinstance(index, int)):
        raise ClassifierError(f"coach_turn_index must be an integer or null, got {index!r}")
    if not isinstance(reason, str):
        raise ClassifierError(f"reason must be a string, got {reason!r}")
    category = None if raw["category"] is None else _enum_value(raw, "category", CATEGORIES)
    return Assessment(
        decision=cast("Label", _enum_value(raw, "decision", LABELS)),
        category=cast("Category | None", category),
        evidence_quote=quote,
        coach_turn_index=index,
        reason=reason,
    )


def _tool_input(message: Message) -> dict[str, Any]:
    if message.stop_reason == "refusal":
        category = message.stop_details.category if message.stop_details else None
        raise ClassifierError(f"model refused (category: {category or 'not given'})")
    if message.stop_reason in ("max_tokens", "model_context_window_exceeded"):
        raise ClassifierError(f"output cut off ({message.stop_reason}); tool input may be partial")
    # Find the call by type and name: a thinking block can come before it.
    for block in message.content:
        if isinstance(block, ToolUseBlock) and block.name == TOOL_NAME:
            return dict(block.input)
    raise ClassifierError(f"no {TOOL_NAME} call (stop_reason: {message.stop_reason})")


def _usage(message: Message) -> Usage:
    usage = message.usage
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
    )


def classify(
    client: ClaudeClient,
    rubric: str,
    case: Case,
    *,
    model: str,
    effort: str,
    max_tokens: int = MAX_TOKENS,
) -> ClassifierResult:
    """One call for one case. SDK errors propagate; the SDK has already retried them."""
    request = build_request(rubric, case, model=model, effort=effort, max_tokens=max_tokens)
    start = time.perf_counter()
    message = client.messages.create(**request)
    latency_s = time.perf_counter() - start
    raw = _tool_input(message)
    return ClassifierResult(
        assessment=parse_assessment(raw),
        raw_tool_input=raw,
        usage=_usage(message),
        latency_s=latency_s,
        stop_reason=message.stop_reason or "",
    )
