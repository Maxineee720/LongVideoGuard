from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from longvideoguard.training.stage6b_data import (
    ANSWERABLE,
    OPTION_LABELS,
    UNANSWERABLE,
    answer_letter,
    answerability_prompt,
    structured_target,
)


def load_training_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load Stage 6B rows containing prompt and structured assistant targets."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training JSONL file not found: {source}")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()

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

            required = {
                "sample_id",
                "video_id",
                "video_relpath",
                "prompt",
                "assistant_target",
                "gold_status",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {missing}"
                )

            sample_id = str(row["sample_id"]).strip()
            if not sample_id:
                raise ValueError(f"Line {line_number}: empty sample_id.")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample_id!r}")
            seen.add(sample_id)

            expected = expected_target(row)
            if str(row["assistant_target"]) != expected:
                raise ValueError(
                    f"Line {line_number}: assistant_target does not match "
                    f"gold labels. Expected {expected!r}, got "
                    f"{row['assistant_target']!r}."
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {source}.")
    return rows


def load_flexible_qa_jsonl(path: str | Path) -> list[dict[str, object]]:
    """
    Load Stage 4/6A QA rows for positive-only development evaluation.

    Both `video_relpath` and the older `video` field are accepted.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"QA JSONL file not found: {source}")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Line {line_number} of {source} must be a JSON object."
                )

            row = dict(raw)
            video_relpath = row.get("video_relpath") or row.get("video")
            if video_relpath is None or not str(video_relpath).strip():
                raise ValueError(
                    f"Line {line_number}: missing video_relpath/video."
                )
            row["video_relpath"] = str(video_relpath).strip()

            required = {
                "sample_id",
                "video_id",
                "question",
                "options",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {missing}"
                )

            sample_id = str(row["sample_id"]).strip()
            if not sample_id:
                raise ValueError(f"Line {line_number}: empty sample_id.")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample_id!r}")
            seen.add(sample_id)

            options = row["options"]
            if not isinstance(options, list) or len(options) != 5:
                raise ValueError(
                    f"Line {line_number}: expected exactly five options."
                )
            _ = answer_letter(row)
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {source}.")
    return rows


def build_positive_eval_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    role: str,
) -> list[dict[str, object]]:
    """Convert ordinary QA rows into structured answerable evaluation rows."""
    converted: list[dict[str, object]] = []

    for row in rows:
        letter = answer_letter(row)
        payload = dict(row)
        payload.update(
            {
                "task": "videoqa_answerability",
                "stage6b_role": role,
                "sample_id": f"{role}::{row['sample_id']}",
                "video_id": str(row["video_id"]),
                "video_relpath": str(row["video_relpath"]),
                "prompt": answerability_prompt(row),
                "assistant_target": structured_target(
                    status=ANSWERABLE,
                    answer=letter,
                ),
                "gold_status": ANSWERABLE,
                "gold_answer_letter": letter,
                "is_answerable": True,
                "source_question_sample_id": str(row["sample_id"]),
            }
        )
        converted.append(payload)

    return converted


def expected_target(row: Mapping[str, object]) -> str:
    status = str(row["gold_status"])
    answer_value = row.get("gold_answer_letter")
    answer = (
        str(answer_value).strip().upper()
        if answer_value is not None
        else None
    )
    return structured_target(status=status, answer=answer)


def parse_structured_output(raw_output: str) -> dict[str, object]:
    """
    Parse a generated answerability JSON object.

    The parser records both semantic validity and whether the raw output was
    already the exact compact canonical JSON expected by the task.
    """
    stripped = str(raw_output).strip()
    candidate = stripped

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return {
                "valid_structure": False,
                "exact_canonical_format": False,
                "predicted_status": None,
                "predicted_answer_letter": None,
                "canonical_output": None,
                "parse_error": "no_json_object",
            }
        candidate = stripped[start : end + 1]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return {
                "valid_structure": False,
                "exact_canonical_format": False,
                "predicted_status": None,
                "predicted_answer_letter": None,
                "canonical_output": None,
                "parse_error": "invalid_json",
            }

    if not isinstance(payload, dict):
        return {
            "valid_structure": False,
            "exact_canonical_format": False,
            "predicted_status": None,
            "predicted_answer_letter": None,
            "canonical_output": None,
            "parse_error": "json_not_object",
        }

    if set(payload) != {"status", "answer"}:
        return {
            "valid_structure": False,
            "exact_canonical_format": False,
            "predicted_status": None,
            "predicted_answer_letter": None,
            "canonical_output": None,
            "parse_error": "wrong_keys",
        }

    status = payload.get("status")
    answer = payload.get("answer")

    if status == ANSWERABLE:
        if not isinstance(answer, str):
            return {
                "valid_structure": False,
                "exact_canonical_format": False,
                "predicted_status": None,
                "predicted_answer_letter": None,
                "canonical_output": None,
                "parse_error": "answerable_requires_string_answer",
            }
        letter = answer.strip().upper()
        if letter not in OPTION_LABELS:
            return {
                "valid_structure": False,
                "exact_canonical_format": False,
                "predicted_status": None,
                "predicted_answer_letter": None,
                "canonical_output": None,
                "parse_error": "invalid_answer_letter",
            }
        canonical = structured_target(status=ANSWERABLE, answer=letter)
        return {
            "valid_structure": True,
            "exact_canonical_format": stripped == canonical,
            "predicted_status": ANSWERABLE,
            "predicted_answer_letter": letter,
            "canonical_output": canonical,
            "parse_error": None,
        }

    if status == UNANSWERABLE:
        if answer is not None:
            return {
                "valid_structure": False,
                "exact_canonical_format": False,
                "predicted_status": None,
                "predicted_answer_letter": None,
                "canonical_output": None,
                "parse_error": "unanswerable_requires_null",
            }
        canonical = structured_target(status=UNANSWERABLE, answer=None)
        return {
            "valid_structure": True,
            "exact_canonical_format": stripped == canonical,
            "predicted_status": UNANSWERABLE,
            "predicted_answer_letter": None,
            "canonical_output": canonical,
            "parse_error": None,
        }

    return {
        "valid_structure": False,
        "exact_canonical_format": False,
        "predicted_status": None,
        "predicted_answer_letter": None,
        "canonical_output": None,
        "parse_error": "invalid_status",
    }


def prediction_is_correct(
    *,
    gold_status: str,
    gold_answer_letter: str | None,
    parsed: Mapping[str, object],
) -> bool:
    if not bool(parsed["valid_structure"]):
        return False
    if parsed["predicted_status"] != gold_status:
        return False
    if gold_status == UNANSWERABLE:
        return parsed["predicted_answer_letter"] is None
    return parsed["predicted_answer_letter"] == gold_answer_letter


def evaluation_summary(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Calculate answerability, QA, refusal, and format metrics."""
    if not predictions:
        raise ValueError("predictions must be non-empty")

    positives = [
        row for row in predictions if row["gold_status"] == ANSWERABLE
    ]
    negatives = [
        row for row in predictions if row["gold_status"] == UNANSWERABLE
    ]

    total = len(predictions)
    exact_correct = sum(bool(row["is_correct"]) for row in predictions)
    valid = sum(bool(row["valid_structure"]) for row in predictions)
    exact_format = sum(
        bool(row["exact_canonical_format"]) for row in predictions
    )

    positive_correct = sum(bool(row["is_correct"]) for row in positives)
    positive_status_correct = sum(
        row["predicted_status"] == ANSWERABLE for row in positives
    )
    false_refusals = sum(
        row["predicted_status"] == UNANSWERABLE for row in positives
    )

    negative_status_correct = sum(
        row["predicted_status"] == UNANSWERABLE for row in negatives
    )
    false_answers = sum(
        row["predicted_status"] == ANSWERABLE for row in negatives
    )

    answerable_accuracy = (
        positive_correct / len(positives)
        if positives
        else None
    )
    unanswerable_recall = (
        negative_status_correct / len(negatives)
        if negatives
        else None
    )
    balanced_task_score = (
        0.5 * (answerable_accuracy + unanswerable_recall)
        if answerable_accuracy is not None
        and unanswerable_recall is not None
        else None
    )

    category_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in positives:
        category_rows[str(row.get("question_category", "unknown"))].append(row)

    by_category: dict[str, dict[str, object]] = {}
    for category, rows in sorted(category_rows.items()):
        correct = sum(bool(row["is_correct"]) for row in rows)
        by_category[category] = {
            "count": len(rows),
            "correct": correct,
            "answerable_accuracy": correct / len(rows),
        }

    status_distribution = Counter(
        str(row["predicted_status"])
        if row["predicted_status"] is not None
        else "invalid"
        for row in predictions
    )

    return {
        "count": total,
        "answerable_count": len(positives),
        "unanswerable_count": len(negatives),
        "overall_exact_correct": exact_correct,
        "overall_exact_accuracy": exact_correct / total,
        "balanced_task_score": balanced_task_score,
        "answerable_exact_correct": positive_correct,
        "answerable_exact_accuracy": answerable_accuracy,
        "answerable_status_accuracy": (
            positive_status_correct / len(positives)
            if positives
            else None
        ),
        "unanswerable_recall": unanswerable_recall,
        "false_refusal_count": false_refusals,
        "false_refusal_rate": (
            false_refusals / len(positives)
            if positives
            else None
        ),
        "false_answer_count": false_answers,
        "false_answer_rate": (
            false_answers / len(negatives)
            if negatives
            else None
        ),
        "valid_structure_count": valid,
        "valid_structure_rate": valid / total,
        "exact_canonical_format_count": exact_format,
        "exact_canonical_format_rate": exact_format / total,
        "answerable_by_question_category": by_category,
        "predicted_status_distribution": {
            ANSWERABLE: status_distribution.get(ANSWERABLE, 0),
            UNANSWERABLE: status_distribution.get(UNANSWERABLE, 0),
            "invalid": status_distribution.get("invalid", 0),
        },
    }


def checkpoint_key(
    metrics: Mapping[str, object],
    *,
    teacher_forced_loss: float,
) -> tuple[float, float, float, float, float]:
    """
    Rank checkpoints without letting an always-answer or always-refuse model win.

    Higher values are better. Loss is negated so lower loss wins the last
    tie-break.
    """
    if not math.isfinite(teacher_forced_loss):
        raise ValueError("teacher_forced_loss must be finite")

    values = (
        float(metrics["balanced_task_score"]),
        float(metrics["answerable_exact_accuracy"]),
        float(metrics["unanswerable_recall"]),
        float(metrics["overall_exact_accuracy"]),
        -teacher_forced_loss,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Checkpoint metrics must be finite")
    return values


def is_better_checkpoint(
    candidate_metrics: Mapping[str, object],
    *,
    candidate_loss: float,
    best_metrics: Mapping[str, object],
    best_loss: float,
    tolerance: float = 1e-12,
) -> bool:
    candidate = checkpoint_key(
        candidate_metrics,
        teacher_forced_loss=candidate_loss,
    )
    best = checkpoint_key(
        best_metrics,
        teacher_forced_loss=best_loss,
    )

    for candidate_value, best_value in zip(candidate, best, strict=True):
        if candidate_value > best_value + tolerance:
            return True
        if candidate_value < best_value - tolerance:
            return False
    return False


def shuffled_epoch_indices(
    *,
    num_samples: int,
    epochs: int,
    seed: int,
) -> list[list[int]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    schedules: list[list[int]] = []
    for epoch_index in range(epochs):
        indices = list(range(num_samples))
        random.Random(seed + epoch_index).shuffle(indices)
        schedules.append(indices)
    return schedules


def finite_mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("Cannot calculate the mean of an empty sequence.")
    if not all(math.isfinite(value) for value in materialized):
        raise ValueError(f"Non-finite values encountered: {materialized}")
    return sum(materialized) / len(materialized)


def move_batch_to_device(
    batch: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def resolve_video_path(
    row: Mapping[str, object],
    *,
    video_root: str | Path,
) -> Path:
    path = Path(video_root).expanduser().resolve() / str(row["video_relpath"])
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    return path


def build_prompt_inputs(
    processor: Any,
    row: Mapping[str, object],
    *,
    video_root: str | Path,
    num_frames: int,
) -> Any:
    video_path = resolve_video_path(row, video_root=video_root)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": str(video_path),
                },
                {
                    "type": "text",
                    "text": str(row["prompt"]),
                },
            ],
        }
    ]

    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = None

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs={
            "text_kwargs": {
                "return_tensors": "pt",
            },
            "videos_kwargs": {
                "do_sample_frames": True,
                "num_frames": num_frames,
            },
        },
    )
    inputs.pop("token_type_ids", None)
    return inputs


def evaluate_generation(
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    video_root: str | Path,
    num_frames: int,
    max_new_tokens: int = 32,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for evaluation.") from exc

    was_training = bool(model.training)
    model.eval()
    device = next(model.parameters()).device
    predictions: list[dict[str, object]] = []

    for position, row in enumerate(rows, start=1):
        inputs = build_prompt_inputs(
            processor,
            row,
            video_root=video_root,
            num_frames=num_frames,
        )
        inputs = inputs.to(device)
        input_length = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        new_tokens = generated[:, input_length:]
        raw_output = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        parsed = parse_structured_output(raw_output)

        gold_status = str(row["gold_status"])
        gold_answer_value = row.get("gold_answer_letter")
        gold_answer = (
            str(gold_answer_value).strip().upper()
            if gold_answer_value is not None
            else None
        )
        correct = prediction_is_correct(
            gold_status=gold_status,
            gold_answer_letter=gold_answer,
            parsed=parsed,
        )

        prediction = {
            "sample_id": str(row["sample_id"]),
            "video_id": str(row["video_id"]),
            "stage6b_role": row.get("stage6b_role"),
            "question_category": str(
                row.get("question_category")
                or row.get("category")
                or "unknown"
            ),
            "gold_status": gold_status,
            "gold_answer_letter": gold_answer,
            "raw_output": raw_output,
            **parsed,
            "is_correct": correct,
            "input_token_count": input_length,
            "generated_token_count": int(new_tokens.shape[-1]),
        }
        predictions.append(prediction)
        print(
            f"[eval {position}/{len(rows)}] {row['sample_id']} "
            f"gold={gold_status}/{gold_answer} "
            f"pred={parsed['predicted_status']}/"
            f"{parsed['predicted_answer_letter']} "
            f"correct={correct}"
        )

    if was_training:
        model.train()

    return evaluation_summary(predictions), predictions


def evaluate_teacher_forced_loss(
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    video_root: str | Path,
    num_frames: int,
) -> float:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for evaluation.") from exc

    from longvideoguard.training.sft_batch import build_qwen3vl_sft_batch

    was_training = bool(model.training)
    model.eval()
    device = next(model.parameters()).device
    losses: list[float] = []

    with torch.inference_mode():
        for position, row in enumerate(rows, start=1):
            print(f"[loss {position}/{len(rows)}] {row['sample_id']}")
            batch, _ = build_qwen3vl_sft_batch(
                processor,
                row,
                video_root=video_root,
                num_frames=num_frames,
            )
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            losses.append(
                float(outputs.loss.detach().float().cpu().item())
            )

    if was_training:
        model.train()
    return finite_mean(losses)


def append_jsonl(
    row: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
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
