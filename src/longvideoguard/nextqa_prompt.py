from __future__ import annotations

import re
from typing import Mapping, Sequence

OPTION_LABELS = ("A", "B", "C", "D", "E")


def format_nextqa_prompt(
    question: str,
    options: Sequence[str],
) -> str:
    """Build a strict multiple-choice prompt without exposing the gold label."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question must be non-empty")
    if len(options) != len(OPTION_LABELS):
        raise ValueError("NExT-QA multiple-choice samples must contain five options")

    clean_options = [str(option).strip() for option in options]
    if not all(clean_options):
        raise ValueError("all options must be non-empty")

    option_block = "\n".join(
        f"{label}. {option}"
        for label, option in zip(OPTION_LABELS, clean_options, strict=True)
    )
    return (
        "Watch the video carefully and answer the multiple-choice question.\n"
        "Choose the single best option using visual evidence from the video.\n"
        "Return exactly one uppercase letter: A, B, C, D, or E. "
        "Do not include an explanation.\n\n"
        f"Question: {clean_question}\n\n"
        f"Options:\n{option_block}\n\n"
        "Answer:"
    )


def parse_option_letter(text: str) -> str | None:
    """Extract one answer label from common model response formats.

    The parser is deliberately conservative. Ambiguous responses containing
    multiple distinct option labels are treated as invalid.
    """
    stripped = text.strip()
    normalized = stripped.upper()
    if not normalized:
        return None

    explicit_uppercase_labels = sorted(set(re.findall(r"\b[A-E]\b", stripped)))
    if len(explicit_uppercase_labels) > 1:
        return None

    if normalized in OPTION_LABELS:
        return normalized

    patterns = (
        r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*[:\-]?\s*\(?([A-E])\)?\b",
        r"^\s*\(?([A-E])\)?[\.\):]?\s*$",
        r"^\s*([A-E])[\.\):]\s+",
    )

    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, normalized))

    unique = sorted(set(matches))
    return unique[0] if len(unique) == 1 else None


def answer_index_to_letter(answer_index: int) -> str:
    if answer_index not in range(len(OPTION_LABELS)):
        raise ValueError(f"answer_index must be from 0 to 4, got {answer_index}")
    return OPTION_LABELS[answer_index]


def validate_manifest_row(row: Mapping[str, object]) -> None:
    """Validate fields required by zero-shot NExT-QA inference."""
    required = {
        "sample_id",
        "video_id",
        "video_relpath",
        "question",
        "options",
        "answer_index",
        "question_category",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"manifest row is missing required fields: {missing}")

    options = row["options"]
    if not isinstance(options, list) or len(options) != 5:
        raise ValueError("manifest row must contain exactly five options")

    try:
        answer_index = int(row["answer_index"])
    except (TypeError, ValueError) as exc:
        raise ValueError("answer_index must be an integer") from exc
    answer_index_to_letter(answer_index)
