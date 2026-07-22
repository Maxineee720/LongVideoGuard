from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

METHODS = ("uniform_8", "scene_aware_8", "uniform_16")
LETTERS = ("A", "B", "C", "D", "E")


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
                    f"Line {line_number} of {source} must contain an object."
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {source}.")
    return rows


def load_prediction_directory(
    directory: str | Path,
) -> dict[str, list[dict[str, object]]]:
    root = Path(directory).expanduser().resolve()
    predictions: dict[str, list[dict[str, object]]] = {}
    for method in METHODS:
        path = root / f"{method}_predictions.jsonl"
        predictions[method] = load_jsonl(path)
    return predictions


def answer_letter(row: Mapping[str, object]) -> str:
    for field in (
        "gold_answer_letter",
        "answer_letter",
        "assistant_target",
    ):
        value = row.get(field)
        if value is None:
            continue
        letter = str(value).strip().upper()
        if letter in LETTERS:
            return letter

    answer_index = row.get("answer_index")
    if answer_index is not None:
        index = int(answer_index)
        if index in range(5):
            return LETTERS[index]

    raise ValueError(
        f"Could not resolve gold answer for {row.get('sample_id')!r}."
    )


def question_category(row: Mapping[str, object]) -> str:
    return str(
        row.get("question_category")
        or row.get("category")
        or "unknown"
    )


def build_case_table(
    manifest_rows: Sequence[Mapping[str, object]],
    predictions_by_method: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
) -> list[dict[str, object]]:
    manifest_index = {
        str(row["sample_id"]): row for row in manifest_rows
    }
    prediction_indices = {
        method: {
            str(row["sample_id"]): row for row in rows
        }
        for method, rows in predictions_by_method.items()
    }

    reference_ids = set(manifest_index)
    for method in METHODS:
        ids = set(prediction_indices[method])
        if ids != reference_ids:
            missing = sorted(reference_ids - ids)
            extra = sorted(ids - reference_ids)
            raise ValueError(
                f"{method} IDs do not match manifest. "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    cases: list[dict[str, object]] = []

    for sample_id in sorted(reference_ids):
        manifest = manifest_index[sample_id]
        gold = answer_letter(manifest)

        method_predictions = {
            method: prediction_indices[method][sample_id]
            for method in METHODS
        }
        predicted_letters = {
            method: method_predictions[method].get("prediction")
            for method in METHODS
        }
        correct_flags = {
            method: bool(method_predictions[method]["is_correct"])
            for method in METHODS
        }

        unique_predictions = {
            prediction
            for prediction in predicted_letters.values()
            if prediction is not None
        }

        if all(correct_flags.values()):
            pattern = "all_correct"
        elif not any(correct_flags.values()):
            pattern = "all_wrong"
        elif (
            correct_flags["uniform_8"]
            and not correct_flags["scene_aware_8"]
        ):
            pattern = "uniform8_only_vs_scene"
        elif (
            correct_flags["scene_aware_8"]
            and not correct_flags["uniform_8"]
        ):
            pattern = "scene_only_vs_uniform8"
        else:
            pattern = "mixed_correctness"

        options = manifest.get("options")
        if not isinstance(options, list):
            options = []

        cases.append(
            {
                "sample_id": sample_id,
                "video_id": str(manifest.get("video_id", "")),
                "question_category": question_category(manifest),
                "question": str(manifest.get("question", "")),
                "options": [str(option) for option in options],
                "gold_answer_letter": gold,
                "uniform_8_prediction": predicted_letters["uniform_8"],
                "scene_aware_8_prediction": predicted_letters[
                    "scene_aware_8"
                ],
                "uniform_16_prediction": predicted_letters["uniform_16"],
                "uniform_8_correct": correct_flags["uniform_8"],
                "scene_aware_8_correct": correct_flags["scene_aware_8"],
                "uniform_16_correct": correct_flags["uniform_16"],
                "prediction_agreement_count": len(unique_predictions),
                "all_predictions_same": len(unique_predictions) <= 1,
                "error_pattern": pattern,
            }
        )

    return cases


def bucket_cases(
    cases: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    buckets: dict[str, list[dict[str, object]]] = {
        "all_three_correct": [],
        "all_three_wrong": [],
        "uniform8_correct_uniform16_wrong": [],
        "uniform16_correct_uniform8_wrong": [],
        "uniform8_correct_scene_wrong": [],
        "scene_correct_uniform8_wrong": [],
        "all_predictions_same_wrong": [],
        "prediction_disagreement": [],
    }

    for case in cases:
        u8 = bool(case["uniform_8_correct"])
        scene = bool(case["scene_aware_8_correct"])
        u16 = bool(case["uniform_16_correct"])

        if u8 and scene and u16:
            buckets["all_three_correct"].append(dict(case))
        if not u8 and not scene and not u16:
            buckets["all_three_wrong"].append(dict(case))
        if u8 and not u16:
            buckets["uniform8_correct_uniform16_wrong"].append(dict(case))
        if u16 and not u8:
            buckets["uniform16_correct_uniform8_wrong"].append(dict(case))
        if u8 and not scene:
            buckets["uniform8_correct_scene_wrong"].append(dict(case))
        if scene and not u8:
            buckets["scene_correct_uniform8_wrong"].append(dict(case))
        if (
            not u8
            and not scene
            and not u16
            and bool(case["all_predictions_same"])
        ):
            buckets["all_predictions_same_wrong"].append(dict(case))
        if not bool(case["all_predictions_same"]):
            buckets["prediction_disagreement"].append(dict(case))

    return buckets


def _accuracy(
    cases: Sequence[Mapping[str, object]],
    method: str,
) -> float:
    if not cases:
        return 0.0
    field = f"{method}_correct"
    return sum(bool(case[field]) for case in cases) / len(cases)


def category_summary(
    cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["question_category"])].append(case)

    summary = {}
    for category, rows in sorted(grouped.items()):
        summary[category] = {
            "count": len(rows),
            "uniform_8_accuracy": _accuracy(rows, "uniform_8"),
            "scene_aware_8_accuracy": _accuracy(
                rows,
                "scene_aware_8",
            ),
            "uniform_16_accuracy": _accuracy(rows, "uniform_16"),
            "all_three_wrong_count": sum(
                not bool(row["uniform_8_correct"])
                and not bool(row["scene_aware_8_correct"])
                and not bool(row["uniform_16_correct"])
                for row in rows
            ),
            "prediction_disagreement_count": sum(
                not bool(row["all_predictions_same"])
                for row in rows
            ),
        }
    return summary


def majority_vote_prediction(
    case: Mapping[str, object],
) -> str | None:
    predictions = [
        case["uniform_8_prediction"],
        case["scene_aware_8_prediction"],
        case["uniform_16_prediction"],
    ]
    valid = [str(value) for value in predictions if value in LETTERS]
    if not valid:
        return None

    counts = Counter(valid)
    most_common = counts.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    if most_common[0][1] > most_common[1][1]:
        return most_common[0][0]

    # Tie-break only for post-hoc diagnostics.
    return str(case["uniform_8_prediction"])


def majority_vote_summary(
    cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    correct = 0
    changed_from_uniform8 = 0
    rows = []

    for case in cases:
        prediction = majority_vote_prediction(case)
        is_correct = prediction == case["gold_answer_letter"]
        correct += int(is_correct)
        changed_from_uniform8 += int(
            prediction != case["uniform_8_prediction"]
        )
        rows.append(
            {
                "sample_id": case["sample_id"],
                "prediction": prediction,
                "gold_answer_letter": case["gold_answer_letter"],
                "is_correct": is_correct,
            }
        )

    return {
        "count": len(cases),
        "correct": correct,
        "accuracy": correct / len(cases),
        "changed_from_uniform8_count": changed_from_uniform8,
        "changed_from_uniform8_rate": (
            changed_from_uniform8 / len(cases)
        ),
        "warning": (
            "Majority vote is post-hoc frozen-set analysis and must not "
            "replace the preregistered policy."
        ),
        "rows": rows,
    }


def build_summary(
    cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    buckets = bucket_cases(cases)
    majority = majority_vote_summary(cases)

    return {
        "count": len(cases),
        "method_accuracy": {
            "uniform_8": _accuracy(cases, "uniform_8"),
            "scene_aware_8": _accuracy(cases, "scene_aware_8"),
            "uniform_16": _accuracy(cases, "uniform_16"),
        },
        "bucket_counts": {
            name: len(rows) for name, rows in buckets.items()
        },
        "category_summary": category_summary(cases),
        "majority_vote_diagnostic": {
            key: value
            for key, value in majority.items()
            if key != "rows"
        },
        "interpretation": {
            "persistent_errors": (
                "all_three_wrong cases are failures not rescued by either "
                "more frames or scene-aware sampling."
            ),
            "frame_budget_sensitivity": (
                "uniform16_correct_uniform8_wrong cases benefit from more "
                "frames; the reverse bucket shows extra frames can also hurt."
            ),
            "sampling_sensitivity": (
                "uniform8/scene-aware disagreement cases isolate errors "
                "caused by frame selection rather than model capacity alone."
            ),
        },
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


def write_csv(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        "sample_id",
        "video_id",
        "question_category",
        "question",
        "options",
        "gold_answer_letter",
        "uniform_8_prediction",
        "scene_aware_8_prediction",
        "uniform_16_prediction",
        "uniform_8_correct",
        "scene_aware_8_correct",
        "uniform_16_correct",
        "all_predictions_same",
        "error_pattern",
    )

    with destination.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in rows:
            output = {
                field: row.get(field)
                for field in fieldnames
            }
            output["options"] = json.dumps(
                row.get("options", []),
                ensure_ascii=False,
            )
            writer.writerow(output)

    return destination


def format_options(options: Sequence[object]) -> str:
    return "\n".join(
        f"  {letter}. {option}"
        for letter, option in zip(
            LETTERS,
            options,
            strict=False,
        )
    )


def write_markdown_report(
    summary: Mapping[str, object],
    buckets: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
    output_path: str | Path,
    *,
    max_cases_per_bucket: int = 20,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Stage 9A — Frozen-set final error analysis",
        "",
        "## Method accuracy",
        "",
        "| Method | Accuracy |",
        "|---|---:|",
    ]
    for method, value in summary["method_accuracy"].items():
        lines.append(f"| {method} | {100 * float(value):.2f}% |")

    lines.extend(
        [
            "",
            "## Error buckets",
            "",
            "| Bucket | Count |",
            "|---|---:|",
        ]
    )
    for name, count in summary["bucket_counts"].items():
        lines.append(f"| {name} | {count} |")

    lines.extend(
        [
            "",
            "## Category summary",
            "",
            "| Category | Count | Uniform-8 | Scene-aware-8 | Uniform-16 | "
            "All-three wrong | Prediction disagreement |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for category, payload in summary["category_summary"].items():
        lines.append(
            f"| {category} | {payload['count']} | "
            f"{100 * payload['uniform_8_accuracy']:.2f}% | "
            f"{100 * payload['scene_aware_8_accuracy']:.2f}% | "
            f"{100 * payload['uniform_16_accuracy']:.2f}% | "
            f"{payload['all_three_wrong_count']} | "
            f"{payload['prediction_disagreement_count']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Persistent errors:** all three methods fail; changing frame "
            "count or scene selection does not rescue the answer.",
            "- **Frame-budget sensitive:** Uniform-8 and Uniform-16 disagree "
            "on correctness.",
            "- **Sampling sensitive:** Uniform-8 and Scene-aware-8 disagree "
            "on correctness.",
            "- Majority vote is reported only as post-hoc diagnostics and "
            "must not replace the preregistered policy.",
            "",
        ]
    )

    selected_buckets = (
        "uniform16_correct_uniform8_wrong",
        "uniform8_correct_uniform16_wrong",
        "scene_correct_uniform8_wrong",
        "uniform8_correct_scene_wrong",
        "all_three_wrong",
    )

    for bucket_name in selected_buckets:
        rows = list(buckets[bucket_name])
        lines.extend(
            [
                f"## {bucket_name}",
                "",
                f"Count: **{len(rows)}**",
                "",
            ]
        )

        for index, case in enumerate(
            rows[:max_cases_per_bucket],
            start=1,
        ):
            lines.extend(
                [
                    f"### {index}. `{case['sample_id']}` "
                    f"({case['question_category']})",
                    "",
                    f"**Question:** {case['question']}",
                    "",
                    "```text",
                    format_options(case.get("options", [])),
                    "```",
                    "",
                    f"- Gold: `{case['gold_answer_letter']}`",
                    f"- Uniform-8: `{case['uniform_8_prediction']}`",
                    f"- Scene-aware-8: "
                    f"`{case['scene_aware_8_prediction']}`",
                    f"- Uniform-16: `{case['uniform_16_prediction']}`",
                    "",
                ]
            )

    destination.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return destination
