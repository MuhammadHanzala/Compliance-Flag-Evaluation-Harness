from __future__ import annotations

from pathlib import Path

import pytest

from evalharness.cost import call_cost, load_prices, summarise_cost
from evalharness.schema import Prediction
from tests.fakes import SONNET_PRICES, make_prediction

PRICING = Path(__file__).resolve().parent.parent / "config" / "pricing.toml"


def _prediction(case_id: str, cost_usd: float, cache_read: int) -> Prediction:
    return make_prediction(
        case_id,
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=cache_read,
        cost_usd=cost_usd,
    )


def test_load_prices_reads_the_committed_config() -> None:
    assert load_prices(PRICING, "claude-sonnet-5-5") == SONNET_PRICES


def test_missing_model_names_the_file_and_the_model(tmp_path: Path) -> None:
    pricing = tmp_path / "pricing.toml"
    pricing.write_text(PRICING.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="no prices for model 'claude-unknown'") as error:
        load_prices(pricing, "claude-unknown")
    assert str(pricing) in str(error.value)


def test_call_cost_matches_a_hand_calculation() -> None:
    cost = call_cost(
        SONNET_PRICES,
        input_tokens=1_000,  # 1,000 x $2.00/M          = $0.002
        cache_creation_input_tokens=2_000,  # 2,000 x $2.50/M  = $0.005
        cache_read_input_tokens=30_000,  # 30,000 x $0.10/M   = $0.003
        output_tokens=500,  # 500 x $10.00/M                  = $0.005
    )
    assert cost == pytest.approx(0.015)


def test_summarise_cost_sums_tokens_and_stored_cost() -> None:
    run = summarise_cost([_prediction("c1", 0.004, 0), _prediction("c2", 0.002, 900)])
    assert (run.calls, run.input_tokens, run.output_tokens) == (2, 200, 100)
    assert run.cache_read_input_tokens == 900
    assert run.total_usd == pytest.approx(0.006)
    assert run.usd_per_100_sessions == pytest.approx(0.3)


def test_summarise_cost_counts_later_calls_that_read_the_cache() -> None:
    # Line 1 is the cache-warming call, so its own cache read does not count.
    run = summarise_cost(
        [_prediction("c1", 0.004, 900), _prediction("c2", 0.002, 900), _prediction("c3", 0.002, 0)]
    )
    assert (run.later_calls_with_cache_read, run.later_calls) == (1, 2)


def test_empty_run_costs_nothing() -> None:
    run = summarise_cost([])
    assert (run.calls, run.total_usd, run.usd_per_100_sessions) == (0, 0.0, 0.0)
    assert (run.later_calls_with_cache_read, run.later_calls) == (0, 0)
