"""Compare two runs on the same eval set, case by case, and gate the change."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalharness.metrics import RunMetrics, pair
from evalharness.schema import Case, Prediction, load_predictions, sha256_file


@dataclass(frozen=True)
class Comparison:
    """Case ids that changed between run A and run B, sorted."""

    false_flags_fixed: list[str]
    new_false_flags: list[str]
    new_misses: list[str]
    misses_fixed: list[str]
    discordant_b: int  # A right, B wrong
    discordant_c: int  # A wrong, B right
    mcnemar_p: float
    reasons_b: dict[str, str]  # run B's stated reason for each changed case


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]  # why it failed; empty when it passed


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test: under no real difference, b ~ Binomial(b + c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(
    cases: Sequence[Case], preds_a: Sequence[Prediction], preds_b: Sequence[Prediction]
) -> Comparison:
    """Pair both runs with the eval set (raises IncompleteRunError) and list what changed."""
    false_flags_fixed: list[str] = []
    new_false_flags: list[str] = []
    new_misses: list[str] = []
    misses_fixed: list[str] = []
    reasons_b: dict[str, str] = {}
    for (case, a), (_, b) in zip(pair(cases, preds_a), pair(cases, preds_b), strict=True):
        if a.flagged == b.flagged:
            continue
        reasons_b[case.id] = str(b.tool_input.get("reason", ""))
        if case.is_breach and b.flagged:
            misses_fixed.append(case.id)
        elif case.is_breach:
            new_misses.append(case.id)
        elif b.flagged:
            new_false_flags.append(case.id)
        else:
            false_flags_fixed.append(case.id)
    discordant_b = len(new_false_flags) + len(new_misses)
    discordant_c = len(false_flags_fixed) + len(misses_fixed)
    return Comparison(
        false_flags_fixed=sorted(false_flags_fixed),
        new_false_flags=sorted(new_false_flags),
        new_misses=sorted(new_misses),
        misses_fixed=sorted(misses_fixed),
        discordant_b=discordant_b,
        discordant_c=discordant_c,
        mcnemar_p=mcnemar_exact(discordant_b, discordant_c),
        reasons_b=reasons_b,
    )


def gate(
    comparison: Comparison,
    metrics_b: RunMetrics,
    *,
    max_new_misses: int = 0,
    recall_floor: float = 0.95,
) -> GateResult:
    """Block run B if it newly misses breaches run A caught, or if its recall is too low."""
    reasons: list[str] = []
    misses = comparison.new_misses
    if len(misses) > max_new_misses:
        reasons.append(
            f"new misses (breaches caught before, missed now): {', '.join(misses)}; "
            f"{len(misses)} found, {max_new_misses} allowed"
        )
    if metrics_b.recall is None:
        reasons.append("recall is undefined because the eval set has no breaches")
    elif metrics_b.recall < recall_floor:
        reasons.append(f"recall {metrics_b.recall:.1%} is below the floor of {recall_floor:.1%}")
    return GateResult(passed=not reasons, reasons=reasons)


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[Prediction]]:
    """Read runs/<name>/manifest.json and runs/<name>/predictions.jsonl."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{run_dir} is not a run directory: it has no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest, load_predictions(run_dir / "predictions.jsonl")


def check_same_set(manifest_a: dict[str, Any], manifest_b: dict[str, Any], eval_path: Path) -> None:
    """Both runs must have used the same eval set, and that file must not have changed since."""
    sha_a, sha_b = manifest_a["eval_set_sha256"], manifest_b["eval_set_sha256"]
    if sha_a != sha_b:
        raise ValueError(
            f"the runs used different eval sets (sha256 {sha_a[:12]} vs {sha_b[:12]}); "
            "rerun one of them on the same set"
        )
    current = sha256_file(eval_path)
    if current != sha_a:
        raise ValueError(
            f"{eval_path} has changed since the runs were made (sha256 now {current[:12]}, "
            f"runs used {sha_a[:12]}); rerun both on the current set"
        )
