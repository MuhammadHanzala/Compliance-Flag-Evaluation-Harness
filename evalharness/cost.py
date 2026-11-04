"""Token prices from config/pricing.toml, and the cost of one call and of a whole run."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evalharness.schema import Prediction

TOKENS_PER_MTOK = 1_000_000


@dataclass(frozen=True)
class Prices:
    """USD per million tokens; cache prices are multipliers of the input price."""

    input_per_mtok: float
    output_per_mtok: float
    cache_write_multiplier: float
    cache_read_multiplier: float


@dataclass(frozen=True)
class RunCost:
    calls: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_usd: float
    later_calls_with_cache_read: int

    @property
    def usd_per_100_sessions(self) -> float:
        return self.total_usd / self.calls * 100 if self.calls else 0.0

    @property
    def later_calls(self) -> int:
        """Every call after the first one, which warmed the cache."""
        return max(self.calls - 1, 0)


def load_prices(path: Path, model: str) -> Prices:
    with path.open("rb") as f:
        models = tomllib.load(f).get("models", {})
    if model not in models:
        known = ", ".join(sorted(models)) or "none"
        raise ValueError(f"{path} has no prices for model {model!r} (it has: {known})")
    entry = models[model]
    return Prices(
        input_per_mtok=float(entry["input_per_mtok"]),
        output_per_mtok=float(entry["output_per_mtok"]),
        cache_write_multiplier=float(entry["cache_write_multiplier"]),
        cache_read_multiplier=float(entry["cache_read_multiplier"]),
    )


def call_cost(
    prices: Prices,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
) -> float:
    """USD for one call. `input_tokens` is only the uncached part of the prompt."""
    cache_write_per_mtok = prices.input_per_mtok * prices.cache_write_multiplier
    cache_read_per_mtok = prices.input_per_mtok * prices.cache_read_multiplier
    return (
        input_tokens * prices.input_per_mtok
        + cache_creation_input_tokens * cache_write_per_mtok
        + cache_read_input_tokens * cache_read_per_mtok
        + output_tokens * prices.output_per_mtok
    ) / TOKENS_PER_MTOK


def summarise_cost(predictions: Sequence[Prediction]) -> RunCost:
    """Totals over saved predictions, using the cost stored with each call."""
    return RunCost(
        calls=len(predictions),
        input_tokens=sum(p.input_tokens for p in predictions),
        output_tokens=sum(p.output_tokens for p in predictions),
        cache_creation_input_tokens=sum(p.cache_creation_input_tokens for p in predictions),
        cache_read_input_tokens=sum(p.cache_read_input_tokens for p in predictions),
        total_usd=sum(p.cost_usd for p in predictions),
        # The file is in completion order, so line 1 is the call that warmed the cache.
        later_calls_with_cache_read=sum(p.cache_read_input_tokens > 0 for p in predictions[1:]),
    )
