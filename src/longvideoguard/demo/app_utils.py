from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

DISPLAY_METHODS = {
    "uniform_8": "Uniform-8",
    "scene_aware_8": "Scene-aware-8",
    "query_aware_8": "Query-aware-8",
    "uniform_16": "Uniform-16",
}


def normalize_options(options: Sequence[str]) -> list[str]:
    cleaned = [str(option).strip() for option in options]
    if len(cleaned) != 5:
        raise ValueError("Exactly five answer options are required.")
    if any(not option for option in cleaned):
        raise ValueError("Answer options cannot be empty.")
    return cleaned


def build_multiple_choice_prompt(
    question: str,
    options: Sequence[str],
) -> str:
    question_text = str(question).strip()
    if not question_text:
        raise ValueError("Question cannot be empty.")

    normalized = normalize_options(options)
    option_block = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(
            ("A", "B", "C", "D", "E"),
            normalized,
            strict=True,
        )
    )
    return (
        "Watch the video carefully and answer the multiple-choice question.\n"
        "Choose the single best option using visual evidence from the video.\n"
        "Return exactly one uppercase letter: A, B, C, D, or E. "
        "Do not include an explanation.\n\n"
        f"Question: {question_text}\n\n"
        f"Options:\n{option_block}\n\n"
        "Answer:"
    )


def parse_answer(raw_output: str) -> str | None:
    text = str(raw_output).strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text

    match = re.search(r"(?<![A-Z])([A-E])(?![A-Z])", text)
    return match.group(1) if match else None


def load_json(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSON file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {source}.")
    return payload


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSONL file not found: {source}")
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def accuracy_card_payload(
    final_metrics: Mapping[str, object],
) -> list[dict[str, object]]:
    frozen = final_metrics["frozen"]
    rows = []
    for method in ("uniform_8", "scene_aware_8", "uniform_16"):
        metrics = frozen[method]
        rows.append(
            {
                "method": DISPLAY_METHODS[method],
                "accuracy": float(metrics["accuracy"]),
                "correct": int(metrics["correct"]),
                "count": int(metrics["count"]),
                "mean_input_tokens": float(
                    metrics["input_token_count"]["mean"]
                ),
                "mean_latency_seconds": float(
                    metrics["latency_seconds"]["mean"]
                ),
            }
        )
    return rows


def resolve_project_file(
    project_root: str | Path,
    *relative_parts: str,
) -> Path:
    return Path(project_root).expanduser().resolve().joinpath(
        *relative_parts
    )
