from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

OPTION_LABELS = ("A", "B", "C", "D", "E")
METHODS = ("uniform", "scene_aware", "query_aware")


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSONL file not found: {source}")

    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} of {source} must be a JSON object."
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {source}.")
    return rows


def answer_letter(row: Mapping[str, object]) -> str:
    for field in (
        "answer_letter",
        "gold_answer_letter",
        "assistant_target",
    ):
        value = row.get(field)
        if value is None:
            continue
        letter = str(value).strip().upper()
        if letter in OPTION_LABELS:
            return letter

    value = row.get("answer_index")
    if value is not None:
        index = int(value)
        if index in range(5):
            return OPTION_LABELS[index]

    raise ValueError(
        f"Could not resolve answer for sample {row.get('sample_id')!r}."
    )


def question_category(row: Mapping[str, object]) -> str:
    return str(
        row.get("question_category")
        or row.get("category")
        or "unknown"
    )


def format_prompt(row: Mapping[str, object]) -> str:
    question = str(row.get("question", "")).strip()
    options = row.get("options")
    if not question:
        raise ValueError(f"Missing question in {row.get('sample_id')!r}.")
    if not isinstance(options, list) or len(options) != 5:
        raise ValueError(
            f"Sample {row.get('sample_id')!r} must have five options."
        )

    option_block = "\n".join(
        f"{letter}. {str(option).strip()}"
        for letter, option in zip(
            OPTION_LABELS,
            options,
            strict=True,
        )
    )

    return (
        "Watch the video carefully and answer the multiple-choice question.\n"
        "Choose the single best option using visual evidence from the video.\n"
        "Return exactly one uppercase letter: A, B, C, D, or E. "
        "Do not include an explanation.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_block}\n\n"
        "Answer:"
    )


def parse_answer(raw_output: str) -> str | None:
    stripped = str(raw_output).strip().upper()
    if stripped in OPTION_LABELS:
        return stripped

    match = re.search(r"(?<![A-Z])([A-E])(?![A-Z])", stripped)
    if match:
        return match.group(1)
    return None


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def method_summary(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not predictions:
        raise ValueError("predictions must be non-empty")

    count = len(predictions)
    correct = sum(bool(row["is_correct"]) for row in predictions)
    valid = sum(row.get("prediction") in OPTION_LABELS for row in predictions)

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["question_category"])].append(row)

    by_category = {}
    for category, rows in sorted(grouped.items()):
        category_correct = sum(bool(row["is_correct"]) for row in rows)
        by_category[category] = {
            "count": len(rows),
            "correct": category_correct,
            "accuracy": category_correct / len(rows),
        }

    latencies = [float(row["latency_seconds"]) for row in predictions]
    input_tokens = [float(row["input_token_count"]) for row in predictions]
    output_tokens = [float(row["generated_token_count"]) for row in predictions]
    peak_memory = [
        float(row["peak_gpu_memory_mb"])
        for row in predictions
        if row.get("peak_gpu_memory_mb") is not None
    ]

    return {
        "count": count,
        "correct": correct,
        "accuracy": correct / count,
        "valid_predictions": valid,
        "valid_rate": valid / count,
        "by_question_category": by_category,
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
        },
        "input_token_count": {
            "mean": statistics.fmean(input_tokens),
            "median": statistics.median(input_tokens),
        },
        "generated_token_count": {
            "mean": statistics.fmean(output_tokens),
            "median": statistics.median(output_tokens),
        },
        "peak_gpu_memory_mb": (
            {
                "mean": statistics.fmean(peak_memory),
                "max": max(peak_memory),
            }
            if peak_memory
            else None
        ),
    }


def exact_mcnemar_p_value(discordant_left: int, discordant_right: int) -> float:
    """
    Exact two-sided McNemar p-value using a Binomial(n, 0.5) test.

    The two inputs are the discordant counts: left-only correct and
    right-only correct.
    """
    if discordant_left < 0 or discordant_right < 0:
        raise ValueError("Discordant counts must be non-negative.")

    total = discordant_left + discordant_right
    if total == 0:
        return 1.0

    smaller = min(discordant_left, discordant_right)
    lower_tail = sum(
        math.comb(total, index)
        for index in range(smaller + 1)
    ) / (2**total)
    return min(1.0, 2.0 * lower_tail)


def paired_comparison(
    baseline_predictions: Sequence[Mapping[str, object]],
    candidate_predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline = {
        str(row["sample_id"]): row for row in baseline_predictions
    }
    candidate = {
        str(row["sample_id"]): row for row in candidate_predictions
    }
    if set(baseline) != set(candidate):
        raise ValueError("Paired predictions must have identical sample IDs.")

    both_correct = 0
    baseline_only = 0
    candidate_only = 0
    both_wrong = 0
    changed_prediction = 0

    for sample_id in sorted(baseline):
        left = baseline[sample_id]
        right = candidate[sample_id]
        left_correct = bool(left["is_correct"])
        right_correct = bool(right["is_correct"])

        if left_correct and right_correct:
            both_correct += 1
        elif left_correct:
            baseline_only += 1
        elif right_correct:
            candidate_only += 1
        else:
            both_wrong += 1

        if left.get("prediction") != right.get("prediction"):
            changed_prediction += 1

    count = len(baseline)
    baseline_accuracy = (
        both_correct + baseline_only
    ) / count
    candidate_accuracy = (
        both_correct + candidate_only
    ) / count

    return {
        "count": count,
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "absolute_accuracy_delta": candidate_accuracy - baseline_accuracy,
        "percentage_point_delta": 100
        * (candidate_accuracy - baseline_accuracy),
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "both_wrong": both_wrong,
        "prediction_change_count": changed_prediction,
        "prediction_change_rate": changed_prediction / count,
        "mcnemar_exact_p_value": exact_mcnemar_p_value(
            baseline_only,
            candidate_only,
        ),
    }


def write_jsonl(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return destination


def write_json(
    payload: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
