from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest

from evalharness.classifier import (
    DEFAULT_MODEL,
    INSTRUCTION,
    RECORD_ASSESSMENT,
    TOOL_NAME,
    ClassifierError,
    ClassifierResult,
    build_request,
    classify,
    parse_assessment,
    render_transcript,
)
from evalharness.schema import CATEGORIES, Case, validate_cases
from tests.fakes import (
    FakeClient,
    answer_by_case,
    assessment_input,
    make_case,
    make_message,
)

RUBRIC = "You review coaching transcripts for compliance breaches."
CASE = make_case("c1", label="breach", coach_text="Buy that ETF.")
BREACH = assessment_input(
    "breach", category="personal_advice", evidence_quote="Buy that ETF.", coach_turn_index=1
)


def _request(case: Case = CASE) -> dict[str, Any]:
    return build_request(RUBRIC, case, model=DEFAULT_MODEL, effort="low", max_tokens=4096)


def _classify_reply(message: anthropic.types.Message) -> ClassifierResult:
    client = FakeClient(lambda _kwargs: message)
    return classify(client, RUBRIC, CASE, model=DEFAULT_MODEL, effort="low")


def test_render_transcript_numbers_every_turn_from_zero() -> None:
    assert render_transcript(CASE) == (
        "<transcript>\n"
        "[0] member: (case c1) What should I do with my savings?\n"
        "[1] coach: Buy that ETF.\n"
        "[2] member: Thanks.\n"
        "</transcript>"
    )


def test_request_caches_the_rubric_in_system() -> None:
    assert _request()["system"] == [
        {"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}
    ]


def test_request_uses_one_strict_tool_with_auto_choice() -> None:
    request = _request()
    (tool,) = request["tools"]
    schema = tool["input_schema"]
    assert tool["name"] == TOOL_NAME
    assert tool["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_nullable_fields_use_any_of_not_a_type_list() -> None:
    # The live API rejected {"type": ["string", "null"], "enum": [..., None]} with a 400.
    properties: dict[str, Any] = RECORD_ASSESSMENT["input_schema"]["properties"]
    schemas = [*properties.values(), *(s for p in properties.values() for s in p.get("anyOf", []))]
    assert [s for s in schemas if isinstance(s.get("type"), list)] == []
    for field in ("category", "evidence_quote", "coach_turn_index"):
        assert {"type": "null"} in properties[field]["anyOf"]
    assert properties["category"]["anyOf"][0] == {"type": "string", "enum": list(CATEGORIES)}


def test_request_sets_thinking_and_effort_and_no_sampling_params() -> None:
    request = _request()
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k"} & request.keys()
    assert request["messages"] == [
        {"role": "user", "content": f"{render_transcript(CASE)}\n\n{INSTRUCTION}"}
    ]


def test_request_differs_between_cases_only_in_messages() -> None:
    first, second = _request(CASE), _request(make_case("c2", coach_text="Try a budget."))
    assert first.pop("messages") != second.pop("messages")
    assert json.dumps(first) == json.dumps(second)


def test_real_sdk_client_round_trip() -> None:
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        reply = make_message(BREACH, thinking=True)
        return httpx2.Response(200, json=reply.model_dump(mode="json"))

    client = anthropic.Anthropic(
        api_key="test-key",
        base_url="http://test.invalid",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    result = classify(client, RUBRIC, CASE, model=DEFAULT_MODEL, effort="low")

    (request,) = sent
    assert str(request.url) == "http://test.invalid/v1/messages"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == json.loads(json.dumps(_request()))
    assert result.assessment.decision == "breach"
    assert result.assessment.evidence_quote == "Buy that ETF."
    assert result.raw_tool_input == BREACH
    assert result.stop_reason == "tool_use"


def test_classify_sends_build_request_and_reports_usage() -> None:
    usage = {
        "input_tokens": 30,
        "output_tokens": 60,
        "cache_creation_input_tokens": 700,
        "cache_read_input_tokens": 0,
    }
    client = FakeClient(lambda _kwargs: make_message(BREACH, usage=usage))
    result = classify(client, RUBRIC, CASE, model=DEFAULT_MODEL, effort="low")
    assert client.calls == [_request()]
    assert (result.usage.input_tokens, result.usage.output_tokens) == (30, 60)
    assert result.usage.cache_creation_input_tokens == 700
    assert result.latency_s >= 0


def test_missing_cache_usage_counts_as_zero() -> None:
    result = _classify_reply(make_message(BREACH, usage={"input_tokens": 5, "output_tokens": 7}))
    assert result.usage.cache_creation_input_tokens == 0
    assert result.usage.cache_read_input_tokens == 0


def test_thinking_block_before_the_tool_call_is_skipped() -> None:
    result = _classify_reply(make_message(BREACH, thinking=True, text="Checking turn 1."))
    assert result.assessment.coach_turn_index == 1


def test_enum_casing_is_normalised() -> None:
    shouted = {**BREACH, "decision": "Breach", "category": "PERSONAL_ADVICE"}
    assessment = _classify_reply(make_message(shouted)).assessment
    assert (assessment.decision, assessment.category) == ("breach", "personal_advice")


def test_refusal_raises_with_its_category() -> None:
    refusal = make_message(
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "general_harms", "explanation": None},
    )
    with pytest.raises(ClassifierError, match="refused.*general_harms"):
        _classify_reply(refusal)


def test_max_tokens_raises_even_with_a_partial_tool_call() -> None:
    with pytest.raises(ClassifierError, match="max_tokens"):
        _classify_reply(make_message({"decision": "breach"}, stop_reason="max_tokens"))


def test_reply_without_a_tool_call_raises() -> None:
    with pytest.raises(ClassifierError, match=f"no {TOOL_NAME} call"):
        _classify_reply(make_message(text="No breach here.", stop_reason="end_turn"))


@pytest.mark.parametrize(
    "change",
    [
        {"decision": "maybe"},
        {"category": "insider_trading"},
        {"evidence_quote": 42},
        {"coach_turn_index": True},
        {"coach_turn_index": "1"},
        {"reason": None},
    ],
)
def test_malformed_tool_input_raises(change: dict[str, Any]) -> None:
    with pytest.raises(ClassifierError):
        parse_assessment({**BREACH, **change})


def test_missing_field_raises() -> None:
    raw = {key: value for key, value in BREACH.items() if key != "reason"}
    with pytest.raises(ClassifierError, match="missing"):
        parse_assessment(raw)


def test_answer_by_case_routes_each_case_to_its_reply() -> None:
    other = make_case("c2")
    assert validate_cases([CASE, other]) == []
    client = FakeClient(
        answer_by_case({"c1": make_message(BREACH), "c2": make_message(assessment_input())})
    )
    decisions = [
        classify(client, RUBRIC, case, model=DEFAULT_MODEL, effort="low").assessment.decision
        for case in (CASE, other)
    ]
    assert decisions == ["breach", "no_breach"]
    assert len(client.calls) == 2
