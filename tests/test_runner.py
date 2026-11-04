from __future__ import annotations

import fcntl
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message

from evalharness.classifier import DEFAULT_MODEL, ClassifierError, build_request
from evalharness.cost import call_cost
from evalharness.runner import RunConfig, RunError, run
from evalharness.schema import load_predictions, sha256_file
from tests.fakes import (
    PRICING_TOML,
    SONNET_PRICES,
    FakeClient,
    answer_by_case,
    assessment_input,
    make_case,
    make_message,
    write_cases,
)

PRICING = f"""{PRICING_TOML}
[models.claude-opus-5-5]
input_per_mtok = 4.0
output_per_mtok = 20.0
cache_write_multiplier = 1.25
cache_read_multiplier = 0.05
"""
CASES = [make_case(f"c{n}") for n in range(1, 6)]


def _usage(input_tokens: int, *, cache_write: int = 0, cache_read: int = 0) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": 40,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
    }


WARM_USAGE = _usage(50, cache_write=800)
READ_USAGE = _usage(50, cache_read=800)
MISS_USAGE = _usage(850)


@pytest.fixture
def config(tmp_path: Path) -> RunConfig:
    write_cases(tmp_path / "eval_set.jsonl", CASES)
    (tmp_path / "rubric.md").write_text("Flag personal advice.", encoding="utf-8")
    (tmp_path / "pricing.toml").write_text(PRICING, encoding="utf-8")
    return RunConfig(
        rubric_path=tmp_path / "rubric.md",
        name="v1",
        model=DEFAULT_MODEL,
        effort="low",
        workers=2,
        eval_path=tmp_path / "eval_set.jsonl",
        runs_dir=tmp_path / "runs",
        pricing_path=tmp_path / "pricing.toml",
    )


def _responder(*, failing: str | None = None) -> Callable[[dict[str, Any]], Message]:
    """c1 writes the cache, c5 misses it, the rest read it. `failing` gets no tool call."""
    answers = {case.id: make_message(assessment_input(), usage=READ_USAGE) for case in CASES}
    answers["c1"] = make_message(assessment_input(), usage=WARM_USAGE)
    answers["c5"] = make_message(assessment_input(), usage=MISS_USAGE)
    if failing:
        answers[failing] = make_message(text="No breach.", stop_reason="end_turn")
    respond = answer_by_case(answers)

    def slow_warm_call(kwargs: dict[str, Any]) -> Message:
        # A slow first call would finish last if the others ran alongside it.
        if "(case c1)" in kwargs["messages"][0]["content"]:
            time.sleep(0.05)
        return respond(kwargs)

    return slow_warm_call


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "http://test.invalid/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls(f"Error code: {status}", response=response, body=None)


def _warm_call_raises(error: Exception) -> Callable[[dict[str, Any]], Message]:
    """c1, the warm-up call, raises `error`; the other cases answer as in _responder()."""
    respond = _responder()

    def route(kwargs: dict[str, Any]) -> Message:
        if "(case c1)" in kwargs["messages"][0]["content"]:
            raise error
        return respond(kwargs)

    return route


def _saved_ids(config: RunConfig) -> list[str]:
    return [p.case_id for p in load_predictions(config.runs_dir / "v1" / "predictions.jsonl")]


def test_run_saves_one_line_per_case_with_the_warm_call_first(config: RunConfig) -> None:
    summary = run(config, FakeClient(_responder()))

    predictions = load_predictions(config.runs_dir / "v1" / "predictions.jsonl")
    assert predictions[0].case_id == "c1"
    assert sorted(p.case_id for p in predictions) == [case.id for case in CASES]
    assert {p.rubric_sha256 for p in predictions} == {sha256_file(config.rubric_path)}
    assert {p.model for p in predictions} == {DEFAULT_MODEL}
    assert (summary.total_cases, summary.skipped, summary.completed) == (5, 0, 5)
    assert summary.failures == []


def test_rerun_with_the_same_settings_makes_zero_calls(config: RunConfig) -> None:
    first = run(config, FakeClient(_responder()))
    client = FakeClient(_responder())

    again = run(config, client)

    assert client.calls == []
    assert (again.skipped, again.completed, again.not_attempted) == (5, 0, 0)
    assert again.cost == first.cost
    assert (again.first_call_cache_write, again.later_calls) == (0, 0)
    assert len(_saved_ids(config)) == 5


def _new_rubric(config: RunConfig) -> RunConfig:
    path = config.rubric_path.with_name("rubric_v2.md")
    path.write_text("Flag personal advice. Explaining a product is not advice.", encoding="utf-8")
    return replace(config, rubric_path=path)


def _new_eval_set(config: RunConfig) -> RunConfig:
    write_cases(config.eval_path, [*CASES, make_case("c6")])
    return config


def _new_model(config: RunConfig) -> RunConfig:
    return replace(config, model="claude-opus-5-5")


def _new_effort(config: RunConfig) -> RunConfig:
    return replace(config, effort="high")


@pytest.mark.parametrize("change", [_new_rubric, _new_eval_set, _new_model, _new_effort])
def test_changed_setting_in_an_existing_run_dir_raises(
    config: RunConfig, change: Callable[[RunConfig], RunConfig]
) -> None:
    run(config, FakeClient(_responder()))
    client = FakeClient(_responder())

    with pytest.raises(RunError, match="Pick a new --name"):
        run(change(config), client)
    assert client.calls == []


def test_failed_case_is_reported_not_saved_and_retried_alone(config: RunConfig) -> None:
    summary = run(config, FakeClient(_responder(failing="c3")))

    [(case_id, error)] = summary.failures
    assert case_id == "c3"
    assert error.startswith("ClassifierError")
    assert summary.completed == 4
    assert "c3" not in _saved_ids(config)

    client = FakeClient(_responder())
    retry = run(config, client)

    assert len(client.calls) == 1
    assert "(case c3)" in client.calls[0]["messages"][0]["content"]
    assert (retry.skipped, retry.completed, retry.failures) == (4, 1, [])
    assert sorted(_saved_ids(config)) == [case.id for case in CASES]


def test_a_rejected_warm_up_request_stops_the_run_and_saves_nothing(config: RunConfig) -> None:
    client = FakeClient(_warm_call_raises(_status_error(anthropic.BadRequestError, 400)))

    summary = run(config, client)

    assert len(client.calls) == 1
    [(case_id, error)] = summary.failures
    assert (case_id, error) == ("c1", "BadRequestError: Error code: 400")
    assert (summary.completed, summary.not_attempted) == (0, summary.total_cases - 1)
    assert _saved_ids(config) == []

    retry = run(config, FakeClient(_responder()))  # once the request is fixed, nothing is lost
    assert (retry.skipped, retry.completed, retry.not_attempted) == (0, 5, 0)


@pytest.mark.parametrize(
    "error",
    [_status_error(anthropic.RateLimitError, 429), ClassifierError("no record_assessment call")],
)
def test_other_warm_up_failures_still_run_every_case(config: RunConfig, error: Exception) -> None:
    client = FakeClient(_warm_call_raises(error))

    summary = run(config, client)

    assert len(client.calls) == 5
    assert [case_id for case_id, _ in summary.failures] == ["c1"]
    assert (summary.completed, summary.not_attempted) == (4, 0)
    assert sorted(_saved_ids(config)) == ["c2", "c3", "c4", "c5"]


def test_summary_counts_cache_use_and_cost(config: RunConfig) -> None:
    summary = run(config, FakeClient(_responder()))

    assert summary.first_call_cache_write == 800
    assert (summary.later_calls, summary.later_calls_with_cache_read) == (4, 3)
    assert summary.cost.calls == 5
    usages = [WARM_USAGE, READ_USAGE, READ_USAGE, READ_USAGE, MISS_USAGE]
    expected = sum(call_cost(SONNET_PRICES, **usage) for usage in usages)
    assert summary.cost.total_usd == pytest.approx(expected)


def test_cache_figures_come_from_this_invocations_warm_call(config: RunConfig) -> None:
    every_call_writes = answer_by_case(
        {case.id: make_message(assessment_input(), usage=WARM_USAGE) for case in CASES}
    )

    def warm_call_fails(kwargs: dict[str, Any]) -> Message:
        if "(case c1)" in kwargs["messages"][0]["content"]:
            return make_message(text="No breach.", stop_reason="end_turn")
        return every_call_writes(kwargs)

    first = run(config, FakeClient(warm_call_fails))
    # The warm-up call failed, so it wrote nothing, even though line 1 of the file wrote 800.
    assert (first.first_call_cache_write, first.later_calls) == (0, 4)
    assert first.later_calls_with_cache_read == 0

    resume = run(config, FakeClient(_responder()))
    # The resume's only call is c1 retried, and it is this invocation's warm-up call.
    assert (resume.first_call_cache_write, resume.later_calls) == (800, 0)


def test_a_torn_last_line_is_dropped_and_its_case_retried(
    config: RunConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    run(config, FakeClient(_responder()))
    path = config.runs_dir / "v1" / "predictions.jsonl"
    torn_id = _saved_ids(config)[-1]
    path.write_bytes(path.read_bytes()[:-20])  # a crash partway through the last write
    client = FakeClient(_responder())

    resume = run(config, client)

    assert str(path) in capsys.readouterr().err
    assert len(client.calls) == 1
    assert f"(case {torn_id})" in client.calls[0]["messages"][0]["content"]
    assert (resume.skipped, resume.completed) == (4, 1)
    assert sorted(_saved_ids(config)) == [case.id for case in CASES]
    assert run(config, FakeClient(_responder())).completed == 0


def test_a_second_run_on_the_same_name_is_refused_while_the_first_holds_it(
    config: RunConfig,
) -> None:
    run_dir = config.runs_dir / "v1"
    run_dir.mkdir(parents=True)
    client = FakeClient(_responder())

    # flock locks belong to each open file, so this stands in for another process's run.
    with (run_dir / ".lock").open("w", encoding="utf-8") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(RunError, match="in use by another run"):
            run(config, client)

    assert client.calls == []
    assert not (run_dir / "manifest.json").exists()


def test_manifest_records_the_run_settings(config: RunConfig) -> None:
    run(config, FakeClient(_responder()))

    manifest = json.loads((config.runs_dir / "v1" / "manifest.json").read_text(encoding="utf-8"))
    request = build_request("", CASES[0], model=DEFAULT_MODEL, effort="low", max_tokens=4096)
    assert manifest["rubric_sha256"] == sha256_file(config.rubric_path)
    assert manifest["eval_set_sha256"] == sha256_file(config.eval_path)
    assert (manifest["model"], manifest["effort"]) == (DEFAULT_MODEL, "low")
    assert manifest["thinking"] == request["thinking"]["type"]
    assert manifest["max_tokens"] == config.max_tokens
