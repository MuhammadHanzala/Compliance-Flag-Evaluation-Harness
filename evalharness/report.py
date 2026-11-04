"""Render the comparison of two runs as a Markdown report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evalharness.compare import Comparison, GateResult
from evalharness.cost import RunCost
from evalharness.metrics import RunMetrics
from evalharness.schema import Case

REASON_CHARS = 160


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _with_ci(value: float | None, ci: tuple[float, float] | None) -> str:
    interval = "n/a" if ci is None else f"{_pct(ci[0])} to {_pct(ci[1])}"
    return f"{_pct(value)} [{interval}]"


def _p_value(p: float) -> str:
    """The comparison operator and the value, e.g. "= 0.0312" or "< 0.0001"."""
    return f"= {p:.4f}" if p >= 0.0001 else "< 0.0001"


def _of(counts: tuple[int, int]) -> str:
    return f"{counts[0]} of {counts[1]}"


def _cell(text: str) -> str:
    """One line, short enough for a table, with pipes escaped."""
    flat = " ".join(text.split())
    if len(flat) > REASON_CHARS:
        flat = flat[: REASON_CHARS - 1].rstrip() + "\N{HORIZONTAL ELLIPSIS}"
    return flat.replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def _metric_rows(m: RunMetrics) -> list[tuple[str, str]]:
    c, e = m.confusion, m.evidence
    return [
        ("Precision [95% CI]", _with_ci(m.precision, m.precision_ci)),
        ("Recall [95% CI]", _with_ci(m.recall, m.recall_ci)),
        ("F1", _pct(m.f1)),
        ("False positive rate", _pct(m.fpr)),
        ("TP / FP / FN / TN", f"{c.tp} / {c.fp} / {c.fn} / {c.tn}"),
        (
            "Flags failing the evidence check (of which false flags)",
            f"{e.failed} of {e.flags} ({e.failed_false_flags})",
        ),
        ("Borderline cases correct", _of(m.borderline_correct)),
    ]


def _headline(name_a: str, name_b: str, a: RunMetrics, b: RunMetrics) -> str:
    rows = [
        [label, x, y] for (label, x), (_, y) in zip(_metric_rows(a), _metric_rows(b), strict=True)
    ]
    return "## Headline\n\n" + _table(["Metric", name_a, name_b], rows)


def _count_rows(
    a: Mapping[str, tuple[int, int]], b: Mapping[str, tuple[int, int]]
) -> list[list[str]]:
    return [[key, _of(a[key]), _of(b[key])] for key in a]


def _breakdowns(name_a: str, name_b: str, a: RunMetrics, b: RunMetrics) -> str:
    return (
        "## Recall by breach type\n\nBreaches caught, out of the breaches of that type.\n\n"
        + _table(["Breach type", name_a, name_b], _count_rows(a.by_category, b.by_category))
        + "\n\n## False flags by trap\n\n"
        + "Non-breaches flagged, out of the non-breaches built on that trap.\n\n"
        + _table(["Trap", name_a, name_b], _count_rows(a.by_trap, b.by_trap))
    )


def _gold(case: Case) -> str:
    return f"{case.label} ({case.category or case.trap})"


def _changed_cases(cases: Sequence[Case], comparison: Comparison, name_b: str) -> str:
    by_id = {case.id: case for case in cases}
    groups = [
        ("False flags fixed", comparison.false_flags_fixed),
        ("New false flags", comparison.new_false_flags),
        ("New misses", comparison.new_misses),
        ("Misses fixed", comparison.misses_fixed),
    ]
    parts = ["## Changed cases"]
    for title, ids in groups:
        rows = [[i, _gold(by_id[i]), _cell(comparison.reasons_b.get(i, ""))] for i in ids]
        body = _table(["Case", "Gold label", f"{name_b} reason"], rows) if ids else "None."
        parts.append(f"### {title} ({len(ids)})\n\n{body}")
    return "\n\n".join(parts)


def _mcnemar(comparison: Comparison, name_a: str, name_b: str) -> str:
    return (
        "## McNemar exact test (reported, not gated)\n\n"
        f"Cases {name_a} got right and {name_b} got wrong: {comparison.discordant_b}. "
        f"Cases {name_a} got wrong and {name_b} got right: {comparison.discordant_c}. "
        f"Two-sided exact p {_p_value(comparison.mcnemar_p)}.\n\n"
        "The p-value is the chance of a split at least this lopsided if both rubrics were "
        "equally accurate; a small value means the difference is unlikely to be chance, but it "
        "does not weigh a miss against a false flag, so the gate does not use it."
    )


def _gate(
    result: GateResult,
    comparison: Comparison,
    metrics_b: RunMetrics,
    name_b: str,
    max_new_misses: int,
    recall_floor: float,
) -> str:
    rows = [
        ["New misses", f"at most {max_new_misses}", str(len(comparison.new_misses))],
        ["Recall", f"at least {_pct(recall_floor)}", _pct(metrics_b.recall)],
    ]
    text = f"## Gate: {'PASS' if result.passed else 'FAIL'}\n\n"
    text += _table(["Check", "Threshold", name_b], rows)
    if result.reasons:
        text += "\n\nWhy it failed:\n\n" + "\n".join(f"- {reason}" for reason in result.reasons)
    return text


def _cost(name_a: str, name_b: str, a: RunCost, b: RunCost) -> str:
    rows = [
        ["Calls", str(a.calls), str(b.calls)],
        ["Input tokens (uncached)", f"{a.input_tokens:,}", f"{b.input_tokens:,}"],
        [
            "Cache write tokens",
            f"{a.cache_creation_input_tokens:,}",
            f"{b.cache_creation_input_tokens:,}",
        ],
        ["Cache read tokens", f"{a.cache_read_input_tokens:,}", f"{b.cache_read_input_tokens:,}"],
        [
            "Later calls that read the cache",
            _of((a.later_calls_with_cache_read, a.later_calls)),
            _of((b.later_calls_with_cache_read, b.later_calls)),
        ],
        ["Output tokens", f"{a.output_tokens:,}", f"{b.output_tokens:,}"],
        ["Total cost (USD)", f"${a.total_usd:.4f}", f"${b.total_usd:.4f}"],
        [
            "Cost per 100 sessions (USD)",
            f"${a.usd_per_100_sessions:.4f}",
            f"${b.usd_per_100_sessions:.4f}",
        ],
    ]
    return "## Cost\n\n" + _table(["", name_a, name_b], rows)


def _manifest_value(manifest: Mapping[str, Any], key: str, chars: int | None = None) -> str:
    found = manifest.get(key)
    return "n/a" if found is None else str(found)[:chars]


def _provenance(name_a: str, name_b: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    def row(label: str, key: str, chars: int | None = None) -> list[str]:
        return [label, _manifest_value(a, key, chars), _manifest_value(b, key, chars)]

    rows = [
        row("Rubric", "rubric_path"),
        row("Rubric sha256", "rubric_sha256", 12),
        row("Model", "model"),
        row("Effort", "effort"),
        row("Eval set sha256", "eval_set_sha256", 12),
    ]
    return "## Run provenance\n\n" + _table(["", name_a, name_b], rows)


def render_report(
    *,
    name_a: str,
    name_b: str,
    cases: Sequence[Case],
    metrics_a: RunMetrics,
    metrics_b: RunMetrics,
    comparison: Comparison,
    gate_result: GateResult,
    cost_a: RunCost,
    cost_b: RunCost,
    manifest_a: Mapping[str, Any],
    manifest_b: Mapping[str, Any],
    max_new_misses: int,
    recall_floor: float,
) -> str:
    """Markdown with no timestamps, so the same inputs always give the same bytes."""
    breaches = sum(case.is_breach for case in cases)
    intro = (
        f"# {name_a} vs {name_b}\n\n"
        f"Eval set: {len(cases)} cases, {breaches} breaches and {len(cases) - breaches} "
        "non-breaches. A flag is a `breach` or `needs_review` decision."
    )
    sections = [
        intro,
        _headline(name_a, name_b, metrics_a, metrics_b),
        _breakdowns(name_a, name_b, metrics_a, metrics_b),
        _changed_cases(cases, comparison, name_b),
        _mcnemar(comparison, name_a, name_b),
        _gate(gate_result, comparison, metrics_b, name_b, max_new_misses, recall_floor),
        _cost(name_a, name_b, cost_a, cost_b),
        _provenance(name_a, name_b, manifest_a, manifest_b),
    ]
    return "\n\n".join(sections) + "\n"
