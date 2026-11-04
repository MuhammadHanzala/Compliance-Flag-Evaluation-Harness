"""The run as a LangGraph: warm the prompt cache on one case, then assess the rest in parallel.

    START -> warm_cache -> fan_out(Send) -> assess (one branch per remaining case) -> END

warm_cache classifies the first case alone, so its call writes the rubric to the prompt cache
and the parallel calls after it read the cache instead of each paying for a write. fan_out
goes straight to END when no case is left, or when the API rejected the warm-up request itself.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from evalharness.classifier import ClassifierError, ClassifierResult, ClaudeClient, classify
from evalharness.cost import Prices, call_cost
from evalharness.evidence import final_decision, quote_found
from evalharness.schema import Case, Prediction

# Supersteps, not cases: warm_cache is one step and all the Sends run in the next.
RECURSION_LIMIT = 10
# The request or the key is wrong (400, 401, 403, 404), so every call would get the same answer.
# 429 and 5xx are not here: they are about load, and the SDK has already retried them.
REJECTED = (
    anthropic.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
)


@dataclass(frozen=True)
class Outcome:
    """What happened to one case: a prediction, or the error that stopped it."""

    case_id: str
    prediction: Prediction | None
    error: str | None
    # The request itself was rejected, so every other case would fail the same way.
    fatal: bool = False


@dataclass(frozen=True)
class RunContext:
    """Run-wide settings and dependencies, shared read-only by every node."""

    client: ClaudeClient
    rubric: str
    rubric_sha256: str
    model: str
    effort: str
    max_tokens: int
    prices: Prices
    on_outcome: Callable[[Outcome], None]


class State(TypedDict):
    cases: list[Case]
    outcomes: Annotated[list[Outcome], operator.add]


class CaseState(TypedDict):
    case: Case


def _prediction(case: Case, result: ClassifierResult, ctx: RunContext) -> Prediction:
    assessment, usage = result.assessment, result.usage
    decision = final_decision(assessment, quote_found(case, assessment))
    return Prediction(
        case_id=case.id,
        decision=decision,
        category=assessment.category,
        evidence_failed=decision == "needs_review",
        tool_input=result.raw_tool_input,
        model=ctx.model,
        rubric_sha256=ctx.rubric_sha256,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cost_usd=call_cost(ctx.prices, **asdict(usage)),
        latency_s=result.latency_s,
        stop_reason=result.stop_reason,
    )


def assess_case(case: Case, ctx: RunContext) -> Outcome:
    """Classify, check the evidence, and hand the outcome to ctx.on_outcome straight away.

    Errors become an Outcome instead of an exception: if one parallel branch raises, LangGraph
    drops the results of every branch in that step, including calls that were already paid for.
    """
    try:
        result = classify(
            ctx.client,
            ctx.rubric,
            case,
            model=ctx.model,
            effort=ctx.effort,
            max_tokens=ctx.max_tokens,
        )
    except (ClassifierError, anthropic.APIError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        fatal = isinstance(exc, REJECTED)
        outcome = Outcome(case_id=case.id, prediction=None, error=error, fatal=fatal)
    else:
        outcome = Outcome(case_id=case.id, prediction=_prediction(case, result, ctx), error=None)
    ctx.on_outcome(outcome)
    return outcome


def warm_cache(state: State, runtime: Runtime[RunContext]) -> dict[str, list[Outcome]]:
    return {"outcomes": [assess_case(state["cases"][0], runtime.context)]}


def fan_out(state: State) -> list[Send] | str:
    rest = state["cases"][1:]
    if not rest or state["outcomes"][0].fatal:  # outcomes holds only the warm-up call's so far
        return END
    return [Send("assess", CaseState(case=case)) for case in rest]


def assess(state: CaseState, runtime: Runtime[RunContext]) -> dict[str, list[Outcome]]:
    return {"outcomes": [assess_case(state["case"], runtime.context)]}


def build_graph() -> CompiledStateGraph[State, RunContext, State, State]:
    builder = StateGraph(State, context_schema=RunContext)
    builder.add_node("warm_cache", warm_cache)
    builder.add_node("assess", assess)
    builder.add_edge(START, "warm_cache")
    builder.add_conditional_edges("warm_cache", fan_out, ["assess", END])
    builder.add_edge("assess", END)
    # No RetryPolicy: its default retries 4xx errors too. The SDK client does the retrying.
    return builder.compile()


def run_cases(cases: Sequence[Case], ctx: RunContext, *, workers: int) -> list[Outcome]:
    """Assess every case, at most `workers` calls at a time after the first.

    The first outcome is always warm_cache's: it is merged one superstep before the others.
    """
    if not cases:
        return []
    final: dict[str, Any] = build_graph().invoke(
        {"cases": list(cases)},
        config={"max_concurrency": workers, "recursion_limit": RECURSION_LIMIT},
        context=ctx,
    )
    return final["outcomes"]
