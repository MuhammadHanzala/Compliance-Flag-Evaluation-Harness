"""Command line: validate the set, check labeller agreement, print the blind sheet, run, compare."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import anthropic

from evalharness.classifier import DEFAULT_EFFORT, DEFAULT_MODEL, ClaudeClient
from evalharness.compare import check_same_set, compare, gate, load_run
from evalharness.cost import summarise_cost
from evalharness.metrics import (
    IncompleteRunError,
    RunMetrics,
    agreement,
    both_differ_from_draft,
    evaluate,
)
from evalharness.report import render_report
from evalharness.runner import RunConfig, RunError, RunSummary, run
from evalharness.schema import (
    CATEGORIES,
    TRAPS,
    Case,
    LabelRow,
    Prediction,
    load_cases,
    load_label_csv,
    validate_cases,
)

OK, FAILED, INPUT_ERROR = 0, 1, 2
DEFAULT_EVAL = Path("data/eval_set.jsonl")
DEFAULT_REPORTS = Path("reports")
LABELLING_SHEET = Path("data/labelling_sheet.md")
EFFORTS = ("low", "medium", "high")
NO_KEY = (
    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env, add your key, then load it, "
    "for example: uv run --env-file .env python -m evalharness run ... "
    "(docker compose reads .env by itself)."
)
SHEET_INTRO = (
    "# Labelling sheet\n\n"
    "Transcripts only, in eval-set order. Label each case on your own with the test in "
    "data/LABELLING.md, without looking at the other labeller's file.\n"
)


def make_client() -> ClaudeClient | None:
    """The real SDK client, or None when there is no key. Tests replace this function."""
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return None
    # The SDK retries connection errors, 408, 409, 429 and 5xx itself, with backoff.
    return anthropic.Anthropic(max_retries=4, timeout=120.0)


def _set_counts(cases: Sequence[Case]) -> list[str]:
    categories = Counter(case.category for case in cases if case.is_breach)
    traps = Counter(case.trap for case in cases if not case.is_breach)
    return [
        f"breach: {categories.total()}",
        *(f"  {category}: {categories[category]}" for category in CATEGORIES),
        f"no_breach: {traps.total()}",
        *(f"  {trap}: {traps[trap]}" for trap in TRAPS),
        f"borderline: {sum(case.borderline for case in cases)}",
    ]


def _validate(args: argparse.Namespace) -> int:
    cases = load_cases(args.eval_set)
    errors = validate_cases(cases)
    if errors:
        print(*errors, sep="\n", file=sys.stderr)
        print(f"{args.eval_set}: {len(errors)} problem(s) found", file=sys.stderr)
        return FAILED
    print(f"{args.eval_set}: {len(cases)} cases, all valid")
    print(*_set_counts(cases), sep="\n")
    return OK


def _load_labels(path: Path) -> list[LabelRow]:
    """A labeller CSV, with a plain message when it is still (partly) the blank template."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    blank = sum(not (row.get("label") or "").strip() for row in rows)
    if blank:
        raise ValueError(
            f"{path} has {blank} of {len(rows)} rows still unlabelled; "
            f"fill it in from {LABELLING_SHEET} (see data/LABELLING.md)"
        )
    return load_label_csv(path)


def _agreement(args: argparse.Namespace) -> int:
    rows_a, rows_b = _load_labels(args.labels_a), _load_labels(args.labels_b)
    result = agreement(rows_a, rows_b)
    against_draft = both_differ_from_draft(load_cases(args.eval), rows_a, rows_b)
    print(f"Cases labelled by both: {result.n}")
    print(f"Raw agreement, breach vs no_breach: {result.raw_agreement:.1%}")
    print(f"Cohen's kappa, breach vs no_breach: {result.kappa_label:.2f}")
    print(f"Cohen's kappa, five classes (no_breach or breach type): {result.kappa_category:.2f}")
    print(f"Disagreements on the five classes: {len(result.disagreements)}")
    for case_id, label_a, label_b in result.disagreements:
        print(f"  {case_id}: A {label_a}, B {label_b}")
    print(f"Both labellers differ from the drafted label: {len(against_draft)}")
    for case_id, drafted, both in against_draft:
        print(f"  {case_id}: drafted {drafted}, both {both}")
    return OK


def _sheet_markdown(cases: Sequence[Case]) -> str:
    """Case ids and numbered turns only: no label, category, trap, borderline flag or rationale."""
    parts = [SHEET_INTRO]
    for case in cases:
        turns = (
            f"{n}. **{turn.speaker}:** {turn.text}" for n, turn in enumerate(case.turns, start=1)
        )
        parts.append(f"## {case.id}\n\n" + "\n".join(turns) + "\n")
    return "\n".join(parts)


def _sheet(args: argparse.Namespace) -> int:
    print(_sheet_markdown(load_cases(args.eval_set)), end="")
    return OK


def _print_summary(summary: RunSummary) -> None:
    cost = summary.cost
    print(
        f"Run {summary.name}: {summary.total_cases} cases, {summary.skipped} already saved, "
        f"{summary.completed} completed, {len(summary.failures)} failed"
    )
    print(
        f"Cost over {cost.calls} saved calls: ${cost.total_usd:.4f} total, "
        f"${cost.usd_per_100_sessions:.4f} per 100 sessions"
    )
    print(
        f"Tokens: {cost.input_tokens:,} input (uncached), "
        f"{cost.cache_creation_input_tokens:,} cache write, "
        f"{cost.cache_read_input_tokens:,} cache read, {cost.output_tokens:,} output"
    )
    if summary.completed or summary.failures:  # a rerun that made no calls has nothing to show
        print(
            f"Prompt cache: this run's first call wrote {summary.first_call_cache_write:,} tokens; "
            f"{summary.later_calls_with_cache_read} of {summary.later_calls} later calls read it"
        )
    if summary.not_attempted:
        # The warm-up call is the only failure; this names its error in place of the list.
        _, error = summary.failures[0]
        print(
            f"Stopped after the first call: the API rejected the request ({error}), "
            f"so the other {summary.not_attempted} cases were not sent. "
            "Fix the cause and rerun the same command.",
            file=sys.stderr,
        )
    elif summary.failures:
        print("Failed cases, not saved (rerun the same command to retry them):", file=sys.stderr)
        for case_id, error in summary.failures:
            print(f"  {case_id}: {error}", file=sys.stderr)


def _run(args: argparse.Namespace) -> int:
    client = make_client()
    if client is None:
        print(NO_KEY, file=sys.stderr)
        return INPUT_ERROR
    config = RunConfig(
        rubric_path=args.rubric,
        name=args.name,
        model=args.model,
        effort=args.effort,
        workers=args.workers,
        eval_path=args.eval,
    )
    summary = run(config, client)
    _print_summary(summary)
    return FAILED if summary.failures else OK


def _evaluate(
    run_dir: Path, cases: Sequence[Case], predictions: Sequence[Prediction]
) -> RunMetrics:
    try:
        return evaluate(cases, predictions)
    except IncompleteRunError as exc:
        message = f"{run_dir} is incomplete: {exc}. Rerun its run command to finish it."
        raise IncompleteRunError(message) from exc


def _compare(args: argparse.Namespace) -> int:
    run_a, run_b = args.run_a, args.run_b
    manifest_a, preds_a = load_run(run_a)
    manifest_b, preds_b = load_run(run_b)
    check_same_set(manifest_a, manifest_b, args.eval)
    cases = load_cases(args.eval)
    metrics_a = _evaluate(run_a, cases, preds_a)
    metrics_b = _evaluate(run_b, cases, preds_b)
    comparison = compare(cases, preds_a, preds_b)
    result = gate(
        comparison, metrics_b, max_new_misses=args.max_new_misses, recall_floor=args.recall_floor
    )
    report = render_report(
        name_a=run_a.name,
        name_b=run_b.name,
        cases=cases,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        comparison=comparison,
        gate_result=result,
        cost_a=summarise_cost(preds_a),
        cost_b=summarise_cost(preds_b),
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        max_new_misses=args.max_new_misses,
        recall_floor=args.recall_floor,
    )
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.reports_dir / f"report_{run_a.name}_vs_{run_b.name}.md"
    report_path.write_text(report, encoding="utf-8")
    print(
        f"{run_a.name} -> {run_b.name}: "
        f"false flags fixed {len(comparison.false_flags_fixed)}, "
        f"new false flags {len(comparison.new_false_flags)}, "
        f"new misses {len(comparison.new_misses)}, misses fixed {len(comparison.misses_fixed)}"
    )
    print(f"Gate: {'PASS' if result.passed else 'FAIL'}")
    for reason in result.reasons:
        print(f"  - {reason}")
    print(f"Report: {report_path}")
    return OK if result.passed else FAILED


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _fraction(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalharness", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="check the eval set and print its counts")
    validate.add_argument("eval_set", type=Path, help="eval set JSONL file")
    validate.set_defaults(handler=_validate)

    agree = commands.add_parser("agreement", help="Cohen's kappa between two labellers")
    agree.add_argument("labels_a", type=Path, help="labeller A's CSV")
    agree.add_argument("labels_b", type=Path, help="labeller B's CSV")
    agree.add_argument(
        "--eval", type=Path, default=DEFAULT_EVAL, help="eval set (default: %(default)s)"
    )
    agree.set_defaults(handler=_agreement)

    sheet = commands.add_parser("sheet", help="print the blind labelling sheet (Markdown)")
    sheet.add_argument("eval_set", type=Path, help="eval set JSONL file")
    sheet.set_defaults(handler=_sheet)

    run_cmd = commands.add_parser("run", help="classify every case with one rubric")
    run_cmd.add_argument("--rubric", type=Path, required=True, help="rubric prompt file")
    run_cmd.add_argument("--name", required=True, help="run directory name under runs/")
    run_cmd.add_argument("--model", default=DEFAULT_MODEL, help="model id (default: %(default)s)")
    run_cmd.add_argument(
        "--effort",
        choices=EFFORTS,
        default=DEFAULT_EFFORT,
        help="thinking effort (default: %(default)s)",
    )
    run_cmd.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="parallel calls after the cache-warming call (default: %(default)s)",
    )
    run_cmd.add_argument(
        "--eval", type=Path, default=DEFAULT_EVAL, help="eval set (default: %(default)s)"
    )
    run_cmd.set_defaults(handler=_run)

    compare_cmd = commands.add_parser(
        "compare", help="compare run_a with run_b, write a report, gate run_b"
    )
    compare_cmd.add_argument("run_a", type=Path, help="baseline run directory, e.g. runs/v1")
    compare_cmd.add_argument("run_b", type=Path, help="candidate run directory (the one gated)")
    compare_cmd.add_argument(
        "--max-new-misses",
        type=_non_negative_int,
        default=0,
        help="most new misses allowed: breaches run_a caught that run_b misses (default: %(default)s)",
    )
    compare_cmd.add_argument(
        "--recall-floor",
        type=_fraction,
        default=0.95,
        help="fail if run_b's breach recall is below this (default: %(default)s)",
    )
    compare_cmd.add_argument(
        "--eval", type=Path, default=DEFAULT_EVAL, help="eval set (default: %(default)s)"
    )
    compare_cmd.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS,
        help="where the Markdown report is written (default: %(default)s)",
    )
    compare_cmd.set_defaults(handler=_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 ok; 1 validation, gate or run failed; 2 bad input (argparse also uses 2)."""
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, KeyError, RunError) as exc:
        detail = f"missing field {exc}" if isinstance(exc, KeyError) else str(exc)
        print(f"error: {detail}", file=sys.stderr)
        return INPUT_ERROR
