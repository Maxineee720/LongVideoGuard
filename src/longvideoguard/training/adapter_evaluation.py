from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

OPTION_LABELS = ("A", "B", "C", "D", "E")


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load a non-empty JSONL file and reject duplicate sample IDs."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSONL file not found: {source}")

    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} of {source} must contain a JSON object."
                )

            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"Line {line_number}: sample_id is missing.")
            if sample_id in seen_ids:
                raise ValueError(
                    f"Line {line_number}: duplicate sample_id {sample_id!r}."
                )
            seen_ids.add(sample_id)
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {source}.")
    return rows


def answer_letter(row: Mapping[str, object]) -> str:
    """Resolve a gold answer letter from supported LongVideoGuard schemas."""
    for field in (
        "gold_answer_letter",
        "answer_letter",
        "assistant_target",
    ):
        value = row.get(field)
        if value is None:
            continue
        letter = str(value).strip().upper()
        if letter in OPTION_LABELS:
            return letter

    index = row.get("answer_index")
    if index is not None:
        try:
            integer = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid answer_index for {row.get('sample_id')}: {index!r}"
            ) from exc
        if integer in range(len(OPTION_LABELS)):
            return OPTION_LABELS[integer]

    raise ValueError(
        f"Could not resolve gold answer for sample {row.get('sample_id')!r}."
    )


def question_category(row: Mapping[str, object]) -> str:
    return str(
        row.get("question_category")
        or row.get("category")
        or "unknown"
    )


def video_filename(row: Mapping[str, object]) -> str:
    """Resolve the portable video path stored in a manifest row."""
    for field in ("video_relpath", "video"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(
        f"Sample {row.get('sample_id')!r} has no video_relpath/video field."
    )


def format_prompt(row: Mapping[str, object]) -> str:
    """Return an existing prompt or build the standard NExT-QA prompt."""
    existing = row.get("prompt")
    if existing is not None and str(existing).strip():
        return str(existing).strip()

    question = str(row.get("question", "")).strip()
    options = row.get("options")
    if not question:
        raise ValueError(
            f"Sample {row.get('sample_id')!r} has no question."
        )
    if not isinstance(options, list) or len(options) != 5:
        raise ValueError(
            f"Sample {row.get('sample_id')!r} must contain five options."
        )

    option_block = "\n".join(
        f"{label}. {str(option).strip()}"
        for label, option in zip(OPTION_LABELS, options, strict=True)
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


def cyclic_swap_video_ids(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Map each unique video ID to a different video using a stable cycle."""
    video_ids = sorted(
        {str(row.get("video_id", "")).strip() for row in rows}
    )
    if any(not video_id for video_id in video_ids):
        raise ValueError("Every row must contain a non-empty video_id.")
    if len(video_ids) < 2:
        raise ValueError("A swap-video test requires at least two videos.")

    return {
        video_id: video_ids[(index + 1) % len(video_ids)]
        for index, video_id in enumerate(video_ids)
    }


def video_path_by_id(
    rows: Sequence[Mapping[str, object]],
    *,
    video_root: str | Path,
) -> dict[str, Path]:
    """Resolve exactly one existing video path for each video ID."""
    root = Path(video_root).expanduser().resolve()
    mapping: dict[str, Path] = {}

    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            raise ValueError("A row has no video_id.")
        candidate = root / video_filename(row)

        previous = mapping.get(video_id)
        if previous is not None and previous != candidate:
            raise ValueError(
                f"Video {video_id!r} maps to multiple paths: "
                f"{previous} and {candidate}."
            )
        mapping[video_id] = candidate

    missing = [str(path) for path in mapping.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing video files: " + ", ".join(missing[:10])
        )
    return mapping


def classification_summary(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Calculate accuracy, validity, and category breakdown."""
    if not predictions:
        raise ValueError("predictions must be non-empty")

    total = len(predictions)
    correct = sum(bool(row.get("is_correct")) for row in predictions)
    valid = sum(
        row.get("predicted_letter") in OPTION_LABELS
        for row in predictions
    )

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("question_category", "unknown"))].append(row)

    by_category: dict[str, dict[str, object]] = {}
    for category, category_rows in sorted(grouped.items()):
        category_total = len(category_rows)
        category_correct = sum(
            bool(row.get("is_correct")) for row in category_rows
        )
        by_category[category] = {
            "count": category_total,
            "correct": category_correct,
            "accuracy": category_correct / category_total,
        }

    predicted_distribution = Counter(
        (
            str(row["predicted_letter"])
            if row.get("predicted_letter") in OPTION_LABELS
            else "INVALID"
        )
        for row in predictions
    )

    return {
        "count": total,
        "correct": correct,
        "accuracy": correct / total,
        "valid_predictions": valid,
        "valid_rate": valid / total,
        "by_question_category": by_category,
        "predicted_label_distribution": {
            label: predicted_distribution.get(label, 0)
            for label in (*OPTION_LABELS, "INVALID")
        },
    }


def swap_video_summary(
    correct_predictions: Sequence[Mapping[str, object]],
    swapped_predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare predictions under correct and deliberately mismatched videos."""
    if len(correct_predictions) != len(swapped_predictions):
        raise ValueError(
            "Correct-video and swapped-video predictions must align."
        )

    correct_by_id = {
        str(row["sample_id"]): row for row in correct_predictions
    }
    swapped_by_id = {
        str(row["sample_id"]): row for row in swapped_predictions
    }
    if set(correct_by_id) != set(swapped_by_id):
        raise ValueError("Prediction sample IDs do not match.")

    changed = 0
    same = 0
    correct_with_real = 0
    correct_with_swap = 0
    memorisation_warning = 0
    rows: list[dict[str, object]] = []

    for sample_id in sorted(correct_by_id):
        real = correct_by_id[sample_id]
        swap = swapped_by_id[sample_id]
        real_letter = real.get("predicted_letter")
        swap_letter = swap.get("predicted_letter")
        did_change = real_letter != swap_letter
        is_real_correct = bool(real.get("is_correct"))
        is_swap_correct = bool(swap.get("is_correct"))

        changed += int(did_change)
        same += int(not did_change)
        correct_with_real += int(is_real_correct)
        correct_with_swap += int(is_swap_correct)

        # If an overfit sample stays correct after replacing the video,
        # question-text memorisation is a plausible explanation.
        warning = (
            is_real_correct
            and is_swap_correct
            and not did_change
        )
        memorisation_warning += int(warning)

        rows.append(
            {
                "sample_id": sample_id,
                "gold_answer_letter": real.get("gold_answer_letter"),
                "correct_video_id": real.get("video_id_used"),
                "swapped_video_id": swap.get("video_id_used"),
                "correct_video_prediction": real_letter,
                "swapped_video_prediction": swap_letter,
                "prediction_changed": did_change,
                "correct_with_real_video": is_real_correct,
                "correct_with_swapped_video": is_swap_correct,
                "text_memorisation_warning": warning,
            }
        )

    total = len(rows)
    return {
        "count": total,
        "prediction_changed": changed,
        "prediction_change_rate": changed / total,
        "prediction_unchanged": same,
        "correct_video_accuracy": correct_with_real / total,
        "swapped_video_accuracy": correct_with_swap / total,
        "accuracy_drop_after_swap": (
            correct_with_real - correct_with_swap
        ) / total,
        "text_memorisation_warning_count": memorisation_warning,
        "text_memorisation_warning_rate": memorisation_warning / total,
        "interpretation": (
            "A low prediction-change rate or many samples that remain correct "
            "after a mismatched video are warnings of question-text "
            "memorisation. This is a diagnostic, not a formal causal proof."
        ),
        "comparisons": rows,
    }


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


def delta_from_baseline(
    adapter_summary: Mapping[str, object],
    baseline_summary: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if baseline_summary is None:
        return None
    adapter_accuracy = float(adapter_summary["accuracy"])
    baseline_accuracy = float(baseline_summary["accuracy"])
    return {
        "adapter_accuracy": adapter_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "absolute_accuracy_delta": adapter_accuracy - baseline_accuracy,
        "percentage_point_delta": 100 * (
            adapter_accuracy - baseline_accuracy
        ),
    }
