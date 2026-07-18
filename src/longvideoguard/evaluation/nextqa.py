from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

OPTION_LABELS = ("A", "B", "C", "D", "E")
INVALID_LABEL = "INVALID"

REQUIRED_FIELDS = {
    "sample_id",
    "video_id",
    "question_category",
    "gold_answer_letter",
    "predicted_letter",
    "raw_output",
    "latency_seconds",
    "input_token_count",
    "generated_token_count",
    "peak_gpu_memory_mb",
    "error",
}


def load_prediction_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load a prediction JSONL file and reject malformed or duplicate rows."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Prediction file not found: {source}")

    rows: list[dict[str, object]] = []
    seen_sample_ids: set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source}: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Line {line_number} of {source} must contain a JSON object."
                )

            missing = sorted(REQUIRED_FIELDS - set(payload))
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing required fields: {missing}"
                )

            sample_id = str(payload["sample_id"]).strip()
            if not sample_id:
                raise ValueError(f"Line {line_number}: sample_id is empty.")
            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"Line {line_number}: duplicate sample_id {sample_id!r}."
                )
            seen_sample_ids.add(sample_id)

            gold = str(payload["gold_answer_letter"]).strip().upper()
            if gold not in OPTION_LABELS:
                raise ValueError(
                    f"Line {line_number}: invalid gold answer letter {gold!r}."
                )

            predicted = payload["predicted_letter"]
            if predicted is not None:
                predicted = str(predicted).strip().upper()
                if predicted not in OPTION_LABELS:
                    predicted = None
                payload["predicted_letter"] = predicted

            rows.append(payload)

    if not rows:
        raise ValueError(f"Prediction file is empty: {source}")
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float | None:
    """Return a nearest-rank percentile for a non-empty numeric sequence."""
    if not values:
        return None
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> dict[str, float] | None:
    """Compute a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")

    proportion = successes / total
    z_squared = z**2
    denominator = 1 + z_squared / total
    center = (
        proportion
        + z_squared / (2 * total)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total**2)
        )
        / denominator
    )
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _numeric_values(
    rows: Iterable[Mapping[str, object]],
    field: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _group_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(rows)
    valid = sum(row.get("predicted_letter") in OPTION_LABELS for row in rows)
    correct = sum(
        row.get("predicted_letter") == row.get("gold_answer_letter")
        for row in rows
    )
    errors = sum(row.get("error") not in (None, "") for row in rows)

    return {
        "count": total,
        "correct": correct,
        "accuracy": _rate(correct, total),
        "accuracy_wilson_95": wilson_interval(correct, total),
        "valid_predictions": valid,
        "valid_rate": _rate(valid, total),
        "invalid_predictions": total - valid,
        "invalid_rate": _rate(total - valid, total),
        "runtime_errors": errors,
        "runtime_error_rate": _rate(errors, total),
    }


def _breakdown(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field, "unknown") or "unknown")
        grouped[key].append(row)
    return {
        key: _group_metrics(group)
        for key, group in sorted(grouped.items())
    }


def _confusion_matrix(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int]]:
    columns = (*OPTION_LABELS, INVALID_LABEL)
    matrix = {
        gold: {predicted: 0 for predicted in columns}
        for gold in OPTION_LABELS
    }

    for row in rows:
        gold = str(row["gold_answer_letter"])
        predicted = row.get("predicted_letter")
        column = str(predicted) if predicted in OPTION_LABELS else INVALID_LABEL
        matrix[gold][column] += 1
    return matrix


def _per_video_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(row)
    return {
        video_id: _group_metrics(group)
        for video_id, group in sorted(grouped.items())
    }


def evaluate_nextqa_predictions(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Calculate NExT-QA pilot quality, validity, and efficiency metrics."""
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction sequence.")

    overall = _group_metrics(rows)
    total = int(overall["count"])

    gold_distribution = Counter(
        str(row["gold_answer_letter"])
        for row in rows
    )
    predicted_distribution = Counter(
        (
            str(row["predicted_letter"])
            if row.get("predicted_letter") in OPTION_LABELS
            else INVALID_LABEL
        )
        for row in rows
    )

    majority_count = max(gold_distribution.values())
    category_breakdown = _breakdown(rows, "question_category")
    category_accuracies = [
        float(metrics["accuracy"])
        for metrics in category_breakdown.values()
        if metrics["accuracy"] is not None
    ]

    latencies = _numeric_values(rows, "latency_seconds")
    input_tokens = _numeric_values(rows, "input_token_count")
    generated_tokens = _numeric_values(rows, "generated_token_count")
    peak_memory = _numeric_values(rows, "peak_gpu_memory_mb")

    experiment_configs = {
        json.dumps(row.get("experiment", {}), sort_keys=True, ensure_ascii=False)
        for row in rows
    }
    environments = {
        json.dumps(row.get("environment", {}), sort_keys=True, ensure_ascii=False)
        for row in rows
    }

    return {
        "schema_version": "1.0",
        "overall": overall,
        "baselines": {
            "random_choice_accuracy": 1 / len(OPTION_LABELS),
            "majority_gold_label": gold_distribution.most_common(1)[0][0],
            "majority_gold_label_accuracy": majority_count / total,
        },
        "macro_category_accuracy": (
            statistics.fmean(category_accuracies)
            if category_accuracies
            else None
        ),
        "by_question_category": category_breakdown,
        "by_question_type": _breakdown(rows, "question_type"),
        "by_video": _per_video_metrics(rows),
        "gold_label_distribution": {
            label: gold_distribution.get(label, 0)
            for label in OPTION_LABELS
        },
        "predicted_label_distribution": {
            label: predicted_distribution.get(label, 0)
            for label in (*OPTION_LABELS, INVALID_LABEL)
        },
        "confusion_matrix": _confusion_matrix(rows),
        "efficiency": {
            "latency_seconds": {
                "count": len(latencies),
                "mean": _mean(latencies),
                "median": _median(latencies),
                "p95_nearest_rank": _nearest_rank_percentile(latencies, 0.95),
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
            "input_token_count": {
                "count": len(input_tokens),
                "mean": _mean(input_tokens),
                "min": min(input_tokens) if input_tokens else None,
                "max": max(input_tokens) if input_tokens else None,
            },
            "generated_token_count": {
                "count": len(generated_tokens),
                "mean": _mean(generated_tokens),
                "min": min(generated_tokens) if generated_tokens else None,
                "max": max(generated_tokens) if generated_tokens else None,
            },
            "peak_gpu_memory_mb": {
                "count": len(peak_memory),
                "mean": _mean(peak_memory),
                "max": max(peak_memory) if peak_memory else None,
            },
        },
        "reproducibility": {
            "num_distinct_experiment_configs": len(experiment_configs),
            "experiment_configs": [
                json.loads(item)
                for item in sorted(experiment_configs)
            ],
            "num_distinct_environments": len(environments),
            "environments": [
                json.loads(item)
                for item in sorted(environments)
            ],
        },
    }


def error_cases(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return incorrect, invalid, or runtime-error rows for manual review."""
    cases: list[dict[str, object]] = []
    for row in rows:
        is_correct = (
            row.get("predicted_letter") == row.get("gold_answer_letter")
        )
        is_valid = row.get("predicted_letter") in OPTION_LABELS
        has_error = row.get("error") not in (None, "")
        if is_correct and is_valid and not has_error:
            continue

        reason: list[str] = []
        if has_error:
            reason.append("runtime_error")
        if not is_valid:
            reason.append("invalid_prediction")
        elif not is_correct:
            reason.append("wrong_answer")

        case = dict(row)
        case["failure_tags"] = reason
        cases.append(case)
    return cases


def write_metrics(
    metrics: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def write_error_cases(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    cases = error_cases(rows)
    with destination.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return destination


def _format_percent(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{100 * float(value):.2f}%"


def write_markdown_report(
    metrics: Mapping[str, object],
    output_path: str | Path,
    *,
    prediction_path: str | Path | None = None,
) -> Path:
    """Write a compact human-readable report without hiding raw metrics."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    overall = metrics["overall"]
    baselines = metrics["baselines"]
    efficiency = metrics["efficiency"]
    by_category = metrics["by_question_category"]

    assert isinstance(overall, Mapping)
    assert isinstance(baselines, Mapping)
    assert isinstance(efficiency, Mapping)
    assert isinstance(by_category, Mapping)

    latency = efficiency["latency_seconds"]
    memory = efficiency["peak_gpu_memory_mb"]
    assert isinstance(latency, Mapping)
    assert isinstance(memory, Mapping)

    lines = [
        "# NExT-QA Zero-shot Pilot Evaluation",
        "",
        "> Preliminary pilot result. Do not present this as a full benchmark score.",
        "",
    ]
    if prediction_path is not None:
        lines.extend(
            [
                f"- Prediction file: `{Path(prediction_path)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall",
            "",
            f"- Samples: **{overall['count']}**",
            f"- Accuracy: **{_format_percent(overall['accuracy'])}**",
            f"- Valid output rate: **{_format_percent(overall['valid_rate'])}**",
            f"- Runtime error rate: **{_format_percent(overall['runtime_error_rate'])}**",
            f"- Random-choice baseline: **{_format_percent(baselines['random_choice_accuracy'])}**",
            f"- Majority-label baseline: **{_format_percent(baselines['majority_gold_label_accuracy'])}**",
            f"- Macro category accuracy: **{_format_percent(metrics['macro_category_accuracy'])}**",
            "",
            "## Accuracy by question category",
            "",
            "| Category | Count | Correct | Accuracy | Valid rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for category, category_metrics in by_category.items():
        assert isinstance(category_metrics, Mapping)
        lines.append(
            "| "
            f"{category} | "
            f"{category_metrics['count']} | "
            f"{category_metrics['correct']} | "
            f"{_format_percent(category_metrics['accuracy'])} | "
            f"{_format_percent(category_metrics['valid_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            f"- Mean latency: **{latency['mean'] if latency['mean'] is not None else 'N/A'} s/sample**",
            f"- Median latency: **{latency['median'] if latency['median'] is not None else 'N/A'} s/sample**",
            f"- P95 latency: **{latency['p95_nearest_rank'] if latency['p95_nearest_rank'] is not None else 'N/A'} s/sample**",
            f"- Maximum recorded peak GPU memory: **{memory['max'] if memory['max'] is not None else 'N/A'} MB**",
            "",
            "## Interpretation constraints",
            "",
            "- The pilot is small, so a few questions can noticeably change accuracy.",
            "- The development pilot has already been inspected and must not be treated as a pristine final test set.",
            "- Report raw predictions and failure cases together with aggregate metrics.",
            "- A full benchmark claim requires a larger frozen evaluation set and controlled comparisons.",
            "",
        ]
    )

    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
