from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message

from evalharness.classifier import DEFAULT_MODEL
from evalharness.cost import call_cost
from evalharness.graph import Outcome, RunContext, assess_case, run_cases
from tests.fakes import (
    SONNET_PRICES,
    FakeClient,
    Responder,
    assessment_input,
    make_case,
    make_message,
)

NO_BREACH = make_message(assessment_input())
CASES = [make_case(f"c{n}") for n in range(10)]


def _context(client: FakeClient, on_outcome: Callable[[Outcome], None] | None = None) -> RunContext:
    return RunContext(
        client=client,
        rubric="Flag personal advice.",
        rubric_sha256="rubric-hash",
        model=DEFAULT_MODEL,
        effort="low",
        max_tokens=4096,
        prices=SONNET_PRICES,
        on_outcome=on_outcome or (lambda _outcome: None),
    )


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx2.Request("POST", "http://test.invalid"))


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "http://test.invalid/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls(f"Error code: {status}", response=response, body=None)


@pytest.mark.parametrize(
    ("quote", "decision", "evidence_failed"),
    [("Buy that ETF.", "breach", False), ("Sell your house.", "needs_review", True)],
)
def test_assess_case_checks_evidence_and_prices_the_call(
    quote: str, decision: str, evidence_failed: bool
) -> None:
    case = make_case("c1", label="breach", coach_text="Buy that ETF.")
    tool_input = assessment_input(
        "breach", category="personal_advice", evidence_quote=quote, coach_turn_index=1
    )
    usage = {
        "input_tokens": 30,
        "output_tokens": 60,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 700,
    }
    seen: list[Outcome] = []
    client = FakeClient(lambda _kwargs: make_message(tool_input, usage=usage))

    outcome = assess_case(case, _context(client, seen.append))

    prediction = outcome.prediction
    assert prediction is not None
    assert outcome.error is None
    assert (prediction.decision, prediction.evidence_failed) == (decision, evidence_failed)
    assert prediction.tool_input == tool_input
    assert (prediction.model, prediction.rubric_sha256) == (DEFAULT_MODEL, "rubric-hash")
    assert prediction.cache_read_input_tokens == 700
    assert prediction.cost_usd == pytest.approx(call_cost(SONNET_PRICES, **usage))
    assert seen == [outcome]


def _no_tool_call(_kwargs: dict[str, Any]) -> Message:
    return make_message(text="No breach.", stop_reason="end_turn")


def _connection_down(_kwargs: dict[str, Any]) -> Message:
    raise _connection_error()


def _rejected(_kwargs: dict[str, Any]) -> Message:
    raise _status_error(anthropic.BadRequestError, 400)


def _rate_limited(_kwargs: dict[str, Any]) -> Message:
    raise _status_error(anthropic.RateLimitError, 429)


def _only(case_id: str, respond: Responder) -> Responder:
    """`respond` answers case_id; every other case gets a no-breach reply."""

    def route(kwargs: dict[str, Any]) -> Message:
        if f"(case {case_id})" in kwargs["messages"][0]["content"]:
            return respond(kwargs)
        return NO_BREACH

    return route


@pytest.mark.parametrize(
    ("respond", "error"),
    [(_no_tool_call, "ClassifierError"), (_connection_down, "APIConnectionError")],
)
def test_assess_case_turns_errors_into_a_failed_outcome(respond: Responder, error: str) -> None:
    seen: list[Outcome] = []
    outcome = assess_case(make_case("c1"), _context(FakeClient(respond), seen.append))
    assert outcome.prediction is None
    assert outcome.error is not None
    assert outcome.error.startswith(error)
    assert not outcome.fatal
    assert seen == [outcome]


@pytest.mark.parametrize(
    ("cls", "status", "fatal"),
    [
        (anthropic.BadRequestError, 400, True),
        (anthropic.AuthenticationError, 401, True),
        (anthropic.PermissionDeniedError, 403, True),
        (anthropic.NotFoundError, 404, True),
        (anthropic.RateLimitError, 429, False),
        (anthropic.InternalServerError, 500, False),
        (anthropic.OverloadedError, 529, False),
    ],
)
def test_only_a_rejected_request_is_fatal(
    cls: type[anthropic.APIStatusError], status: int, fatal: bool
) -> None:
    def respond(_kwargs: dict[str, Any]) -> Message:
        raise _status_error(cls, status)

    outcome = assess_case(make_case("c1"), _context(FakeClient(respond)))
    assert outcome.error is not None
    assert outcome.error.startswith(cls.__name__)
    assert outcome.fatal is fatal


def test_first_case_is_classified_alone_before_any_other_call_starts() -> None:
    lock = threading.Lock()
    events: list[str] = []

    def respond(_kwargs: dict[str, Any]) -> Message:
        with lock:
            events.append("start")
        time.sleep(0.02)
        with lock:
            events.append("end")
        return NO_BREACH

    client = FakeClient(respond)
    outcomes = run_cases(CASES, _context(client), workers=4)

    assert "(case c0)" in client.calls[0]["messages"][0]["content"]
    assert events[:2] == ["start", "end"]
    assert len(events) == 2 * len(CASES)
    assert outcomes[0].case_id == "c0"  # the runner reads the warm-up call from outcomes[0]


def test_parallel_calls_never_exceed_max_concurrency() -> None:
    lock = threading.Lock()
    in_flight = peak = 0

    def respond(_kwargs: dict[str, Any]) -> Message:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.03)
        with lock:
            in_flight -= 1
        return NO_BREACH

    run_cases(CASES, _context(FakeClient(respond)), workers=3)
    assert 1 < peak <= 3


@pytest.mark.parametrize("respond", [_connection_down, _rejected])
def test_every_case_gets_one_outcome_and_a_failing_branch_keeps_the_rest(
    respond: Responder,
) -> None:
    # A rejected request after the warm-up call stops nothing: the other cases are already sent.
    client = FakeClient(_only("c5", respond))
    outcomes = run_cases(CASES, _context(client), workers=4)

    assert len(client.calls) == len(CASES)
    assert sorted(o.case_id for o in outcomes) == sorted(case.id for case in CASES)
    assert [o.case_id for o in outcomes if o.error] == ["c5"]


def test_a_rejected_warm_up_request_stops_the_run_after_one_call() -> None:
    client = FakeClient(_only("c0", _rejected))
    outcomes = run_cases(CASES, _context(client), workers=4)

    assert len(client.calls) == 1
    [outcome] = outcomes
    assert (outcome.case_id, outcome.fatal) == ("c0", True)
    assert outcome.error is not None
    assert outcome.error.startswith("BadRequestError")


@pytest.mark.parametrize("respond", [_rate_limited, _no_tool_call])
def test_other_warm_up_failures_still_fan_out_to_every_case(respond: Responder) -> None:
    client = FakeClient(_only("c0", respond))
    outcomes = run_cases(CASES, _context(client), workers=4)

    assert len(client.calls) == len(CASES)
    assert sorted(o.case_id for o in outcomes) == sorted(case.id for case in CASES)
    assert [o.case_id for o in outcomes if o.error] == ["c0"]


@pytest.mark.parametrize("count", [0, 1])
def test_zero_or_one_case_needs_no_fan_out(count: int) -> None:
    client = FakeClient(lambda _kwargs: NO_BREACH)
    outcomes = run_cases(CASES[:count], _context(client), workers=4)
    assert [o.case_id for o in outcomes] == [case.id for case in CASES[:count]]
    assert len(client.calls) == count
