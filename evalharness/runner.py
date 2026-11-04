"""Run one rubric over the eval set, saving each prediction the moment its call returns."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalharness import __version__
from evalharness.classifier import MAX_TOKENS, ClaudeClient
from evalharness.cost import RunCost, load_prices, summarise_cost
from evalharness.graph import Outcome, RunContext, run_cases
from evalharness.schema import Case, Prediction, load_cases, load_predictions, sha256_file

# Settings that change what the model is asked. A run directory holds one combination only.
LOCKED_SETTINGS = ("rubric_sha256", "model", "effort", "eval_set_sha256")


class RunError(Exception):
    """The run cannot go ahead as configured."""


@dataclass(frozen=True)
class RunConfig:
    rubric_path: Path
    name: str
    model: str
    effort: str
    workers: int = 4
    eval_path: Path = Path("data/eval_set.jsonl")
    runs_dir: Path = Path("runs")
    pricing_path: Path = Path("config/pricing.toml")
    max_tokens: int = MAX_TOKENS


@dataclass(frozen=True)
class RunSummary:
    """What one invocation did. Cost covers every prediction in the file; the cache figures cover
    only this invocation's calls (first_call_cache_write is 0 if its warm-up call failed).
    """

    name: str
    total_cases: int
    skipped: int
    completed: int
    failures: list[tuple[str, str]]
    # Cases not sent: the API rejected the warm-up request, so the run stopped after it.
    not_attempted: int
    cost: RunCost
    first_call_cache_write: int
    later_calls: int
    later_calls_with_cache_read: int


def _manifest(config: RunConfig, rubric_sha256: str, eval_set_sha256: str) -> dict[str, Any]:
    return {
        "name": config.name,
        "rubric_path": config.rubric_path.as_posix(),
        "rubric_sha256": rubric_sha256,
        "model": config.model,
        "effort": config.effort,
        "thinking": "adaptive",
        "max_tokens": config.max_tokens,
        "eval_set_path": config.eval_path.as_posix(),
        "eval_set_sha256": eval_set_sha256,
        "harness_version": __version__,
    }


def _check_or_write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest for a new run; refuse to mix settings in an existing one."""
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    changed = [key for key in LOCKED_SETTINGS if saved.get(key) != manifest[key]]
    if changed:
        raise RunError(
            f"{path.parent} was run with a different {', '.join(changed)}. "
            "Pick a new --name for this configuration."
        )


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    """Hold runs/<name>/.lock for the whole run, so two runs never write one directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    # flock is Unix-only, which is fine: the harness runs on Linux (Docker and CI). The kernel
    # drops the lock if the process dies, so a killed run never blocks the rerun that resumes it.
    with (run_dir / ".lock").open("w", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunError(
                f"{run_dir} is in use by another run. Wait for it to finish, or pick a new --name."
            ) from None
        yield


def _drop_torn_last_line(path: Path) -> None:
    """Cut a last line that has no newline (a crash mid-write), so its case is retried."""
    data = path.read_bytes() if path.exists() else b""
    if data and not data.endswith(b"\n"):
        os.truncate(path, data.rfind(b"\n") + 1)
        print(f"warning: dropped an incomplete last line from {path}", file=sys.stderr)


def _saver(path: Path) -> Callable[[Outcome], None]:
    """Append each prediction as one line. Failures are not saved, so a rerun retries them."""
    lock = threading.Lock()

    def save(outcome: Outcome) -> None:
        if outcome.prediction is None:
            return
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(outcome.prediction.to_json() + "\n")

    return save


def _summary(
    name: str, *, total: int, skipped: int, outcomes: list[Outcome], saved: list[Prediction]
) -> RunSummary:
    # outcomes[0] is this invocation's warm_cache call; the parallel calls come after it.
    warm = outcomes[0].prediction if outcomes else None
    later = [o.prediction for o in outcomes[1:] if o.prediction is not None]
    return RunSummary(
        name=name,
        total_cases=total,
        skipped=skipped,
        completed=sum(o.prediction is not None for o in outcomes),
        failures=[(o.case_id, o.error) for o in outcomes if o.error is not None],
        not_attempted=total - skipped - len(outcomes),
        cost=summarise_cost(saved),
        first_call_cache_write=warm.cache_creation_input_tokens if warm else 0,
        later_calls=len(later),
        later_calls_with_cache_read=sum(p.cache_read_input_tokens > 0 for p in later),
    )


def _context(
    config: RunConfig, client: ClaudeClient, rubric_sha256: str, predictions_path: Path
) -> RunContext:
    """Everything the graph nodes need, with a saver that appends to predictions.jsonl."""
    return RunContext(
        client=client,
        rubric=config.rubric_path.read_text(encoding="utf-8"),
        rubric_sha256=rubric_sha256,
        model=config.model,
        effort=config.effort,
        max_tokens=config.max_tokens,
        prices=load_prices(config.pricing_path, config.model),
        on_outcome=_saver(predictions_path),
    )


def _pending(cases: list[Case], predictions_path: Path) -> list[Case]:
    """The cases with no saved prediction yet."""
    done = {p.case_id for p in load_predictions(predictions_path)}
    return [case for case in cases if case.id not in done]


def run(config: RunConfig, client: ClaudeClient) -> RunSummary:
    """Classify every case not yet saved in runs/<name>/. A finished run makes no calls."""
    cases = load_cases(config.eval_path)
    rubric_sha256 = sha256_file(config.rubric_path)
    manifest = _manifest(config, rubric_sha256, sha256_file(config.eval_path))
    run_dir = config.runs_dir / config.name
    predictions_path = run_dir / "predictions.jsonl"
    ctx = _context(config, client, rubric_sha256, predictions_path)
    with _run_lock(run_dir):
        _check_or_write_manifest(run_dir / "manifest.json", manifest)
        _drop_torn_last_line(predictions_path)
        pending = _pending(cases, predictions_path)
        outcomes = run_cases(pending, ctx, workers=config.workers)
        saved = load_predictions(predictions_path)
    return _summary(
        config.name,
        total=len(cases),
        skipped=len(cases) - len(pending),
        outcomes=outcomes,
        saved=saved,
    )
