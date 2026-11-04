"""Data types for the eval set, predictions and labeller CSVs, plus loading and validation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, get_args

Speaker = Literal["coach", "member"]
Label = Literal["breach", "no_breach"]
Category = Literal["personal_advice", "credit_assistance", "guarantee", "tax_legal"]
Trap = Literal[
    "general_education",
    "member_mentions_product",
    "referral_to_adviser",
    "budgeting_action",
    "debt_strategy_general",
    "repeating_member",
]
# A flag whose evidence quote failed the check stays a flag, marked for a human.
Decision = Literal["breach", "no_breach", "needs_review"]

SPEAKERS: tuple[str, ...] = get_args(Speaker)
LABELS: tuple[str, ...] = get_args(Label)
CATEGORIES: tuple[str, ...] = get_args(Category)
TRAPS: tuple[str, ...] = get_args(Trap)

MIN_TURNS, MAX_TURNS = 3, 8


@dataclass(frozen=True)
class Turn:
    speaker: Speaker
    text: str


@dataclass(frozen=True)
class Case:
    id: str
    turns: tuple[Turn, ...]
    label: Label
    category: Category | None
    trap: Trap | None
    borderline: bool
    rationale: str

    @property
    def is_breach(self) -> bool:
        return self.label == "breach"


@dataclass(frozen=True)
class Prediction:
    """One line of runs/<name>/predictions.jsonl."""

    case_id: str
    decision: Decision
    category: Category | None
    evidence_failed: bool
    tool_input: dict[str, Any]
    model: str
    rubric_sha256: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float
    latency_s: float
    stop_reason: str

    @property
    def flagged(self) -> bool:
        return self.decision != "no_breach"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(frozen=True)
class LabelRow:
    """One labeller's call on one case, read from labels_a.csv / labels_b.csv."""

    id: str
    label: Label
    category: Category | None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_from_dict(raw: dict[str, Any]) -> Case:
    return Case(
        id=raw["id"],
        turns=tuple(Turn(speaker=t["speaker"], text=t["text"]) for t in raw["turns"]),
        label=raw["label"],
        category=raw.get("category"),
        trap=raw.get("trap"),
        borderline=bool(raw.get("borderline", False)),
        rationale=raw.get("rationale", ""),
    )


def _jsonl_lines(path: Path) -> list[tuple[int, str]]:
    """The non-blank lines of a JSONL file, each with its 1-based line number."""
    # Split on "\n" only: splitlines() also breaks on U+2028/U+0085, which json.dumps leaves raw.
    lines = path.read_text(encoding="utf-8").split("\n")
    return [(n, line) for n, line in enumerate(lines, start=1) if line.strip()]


def load_cases(path: Path) -> list[Case]:
    return [_case_from_dict(json.loads(line)) for _, line in _jsonl_lines(path)]


def _case_errors(case: Case) -> list[str]:
    errors: list[str] = []
    if case.label not in LABELS:
        errors.append(f"label {case.label!r} is not one of {LABELS}")
    if case.label == "breach" and case.category not in CATEGORIES:
        errors.append("breach needs a category")
    if case.label == "breach" and case.trap is not None:
        errors.append("breach must not have a trap")
    if case.label == "no_breach" and case.category is not None:
        errors.append("no_breach must have category null")
    if case.label == "no_breach" and case.trap not in TRAPS:
        errors.append(f"no_breach needs a trap from {TRAPS}")
    if not MIN_TURNS <= len(case.turns) <= MAX_TURNS:
        errors.append(f"has {len(case.turns)} turns, expected {MIN_TURNS}-{MAX_TURNS}")
    if any(t.speaker not in SPEAKERS for t in case.turns):
        errors.append(f"speaker must be one of {SPEAKERS}")
    if not any(t.speaker == "coach" for t in case.turns):
        errors.append("has no coach turn")
    if not case.rationale.strip():
        errors.append("needs a rationale")
    return errors


def validate_cases(cases: Iterable[Case]) -> list[str]:
    """Return one message per problem; an empty list means the set is valid."""
    errors: list[str] = []
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            errors.append(f"{case.id}: duplicate id")
        seen.add(case.id)
        errors.extend(f"{case.id}: {e}" for e in _case_errors(case))
    return errors


def load_predictions(path: Path) -> list[Prediction]:
    if not path.exists():
        return []
    predictions: list[Prediction] = []
    for n, line in _jsonl_lines(path):
        try:
            predictions.append(Prediction(**json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{n}: {exc}. If this is a torn last line from a crash, delete it and rerun."
            ) from exc
    return predictions


def load_label_csv(path: Path) -> list[LabelRow]:
    """Read a labeller CSV with columns id,label,category (category blank for no_breach)."""
    # utf-8-sig: spreadsheets often save "CSV UTF-8" with a BOM, which would hide the id column.
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    labels: list[LabelRow] = []
    for row in rows:
        label = (row.get("label") or "").strip()
        category = (row.get("category") or "").strip() or None
        if label not in LABELS:
            raise ValueError(f"{path}: {row.get('id')}: label {label!r} is not one of {LABELS}")
        if label == "breach" and category not in CATEGORIES:
            raise ValueError(f"{path}: {row['id']}: breach needs a category from {CATEGORIES}")
        category = category if label == "breach" else None
        labels.append(LabelRow(id=row["id"].strip(), label=label, category=category))
    return labels
