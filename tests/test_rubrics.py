"""Checks on the rubric files themselves: shared output contract, cacheable length, no leaked cases."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUBRIC_V1 = ROOT / "prompts" / "rubric_v1.md"
RUBRIC_V2 = ROOT / "prompts" / "rubric_v2.md"
EVAL_SET = ROOT / "data" / "eval_set.jsonl"

# Common words that make unrelated sentences look alike.
STOPWORDS = frozenset(
    {"and", "can", "for", "just", "should", "that", "the", "this", "with", "you", "your"}
)
# Shorter fragments ("can't lose", "Pretty much.") overlap by chance, not by paraphrase.
MIN_WORDS = 4
# The original v2 examples scored up to 0.455 (an avalanche example against c049);
# the rewritten ones score at most 0.222.
MAX_SIMILARITY = 0.3


def _output_section(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text[text.index("## Output") :]


def _words(text: str) -> frozenset[str]:
    text = re.sub("['\u2019]", "", text.lower())  # "you're" -> "youre"
    return frozenset(w for w in re.findall(r"[a-z0-9]+", text) if len(w) >= 3) - STOPWORDS


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b)


def _eval_sentences() -> list[tuple[str, str]]:
    sentences = []
    for line in EVAL_SET.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        for turn in case["turns"]:
            for sentence in re.split(r"(?<=[.!?])\s+", turn["text"]):
                sentences.append((case["id"], sentence))
    return sentences


def test_output_sections_are_identical() -> None:
    assert _output_section(RUBRIC_V1) == _output_section(RUBRIC_V2)


def test_rubric_v2_clears_the_prompt_cache_minimum() -> None:
    # Sonnet 5.5 only caches a prefix of at least 512 tokens, silently. At the usual
    # ~4 characters per English token, 4,000 characters is ~1,000 tokens: a safe margin.
    assert len(RUBRIC_V2.read_text(encoding="utf-8")) > 4_000


def test_rubric_v2_examples_do_not_paraphrase_eval_cases() -> None:
    quotes = re.findall(r'"([^"\n]+)"', RUBRIC_V2.read_text(encoding="utf-8"))
    examples = [q for q in quotes if len(_words(q)) >= MIN_WORDS]
    sentences = [(i, s) for i, s in _eval_sentences() if len(_words(s)) >= MIN_WORDS]
    assert examples and sentences

    score, example, (case_id, sentence) = max(
        (_jaccard(_words(e), _words(s)), e, (i, s)) for e in examples for i, s in sentences
    )

    assert score < MAX_SIMILARITY, (
        f"{score:.3f}: rubric example {example!r} is close to {case_id}: {sentence!r}"
    )
