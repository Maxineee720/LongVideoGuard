from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

OPTION_LABELS = ("A", "B", "C", "D", "E")


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load a non-empty JSONL file and reject duplicate sample IDs."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSONL file not found: {source}")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()

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

            required = {
                "sample_id",
                "video_id",
                "video_relpath",
                "prompt",
                "assistant_target",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing required fields: {missing}"
                )

            sample_id = str(row["sample_id"]).strip()
            if not sample_id:
                raise ValueError(f"Line {line_number}: sample_id is empty.")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample_id!r}")
            seen.add(sample_id)

            target = str(row["assistant_target"]).strip().upper()
            if target not in OPTION_LABELS:
                raise ValueError(
                    f"Line {line_number}: invalid assistant target {target!r}."
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {source}.")
    return rows


def shuffled_epoch_indices(
    *,
    num_samples: int,
    epochs: int,
    seed: int,
) -> list[list[int]]:
    """Return a deterministic but independently shuffled order per epoch."""
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


def deterministic_subset(
    rows: Sequence[Mapping[str, object]],
    *,
    max_samples: int | None,
    seed: int,
) -> list[Mapping[str, object]]:
    """Select a stable diagnostic subset without changing the source order."""
    if max_samples is None or max_samples >= len(rows):
        return list(rows)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(rows)), max_samples))
    return [rows[index] for index in selected_indices]


def is_better_checkpoint(
    *,
    accuracy: float,
    loss: float,
    best_accuracy: float | None,
    best_loss: float | None,
    tolerance: float = 1e-12,
) -> bool:
    """Prefer higher holdout accuracy; break ties using lower holdout loss."""
    if not math.isfinite(accuracy) or not math.isfinite(loss):
        raise ValueError("Checkpoint metrics must be finite.")

    if best_accuracy is None or best_loss is None:
        return True
    if accuracy > best_accuracy + tolerance:
        return True
    if abs(accuracy - best_accuracy) <= tolerance and loss < best_loss:
        return True
    return False


def update_patience(
    *,
    improved: bool,
    bad_epochs: int,
) -> int:
    if bad_epochs < 0:
        raise ValueError("bad_epochs must be non-negative")
    return 0 if improved else bad_epochs + 1


def finite_mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("Cannot calculate the mean of an empty sequence.")
    if not all(math.isfinite(value) for value in materialized):
        raise ValueError(f"Non-finite values encountered: {materialized}")
    return sum(materialized) / len(materialized)


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


def resolve_video_path(
    row: Mapping[str, object],
    *,
    video_root: str | Path,
) -> Path:
    root = Path(video_root).expanduser().resolve()
    path = root / str(row["video_relpath"])
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
    """Build prompt-only video inputs for deterministic generation."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

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


def move_batch_to_device(
    batch: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def evaluate_generation(
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    video_root: str | Path,
    num_frames: int,
    max_new_tokens: int = 4,
    video_override_by_sample: Mapping[str, tuple[str, Path]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Generate answer letters and calculate accuracy with category breakdown."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for evaluation.") from exc

    from longvideoguard.nextqa_prompt import parse_option_letter

    was_training = bool(model.training)
    model.eval()
    device = next(model.parameters()).device
    predictions: list[dict[str, object]] = []

    for position, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        original_video_id = str(row["video_id"])
        if video_override_by_sample is None:
            used_video_id = original_video_id
            prompt_row = row
            inputs = build_prompt_inputs(
                processor,
                prompt_row,
                video_root=video_root,
                num_frames=num_frames,
            )
        else:
            used_video_id, replacement_path = video_override_by_sample[sample_id]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": str(replacement_path),
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
        predicted = parse_option_letter(raw_output)
        gold = str(row["assistant_target"]).strip().upper()
        category = str(
            row.get("question_category")
            or row.get("category")
            or "unknown"
        )

        prediction = {
            "sample_id": sample_id,
            "source_video_id": original_video_id,
            "video_id_used": used_video_id,
            "question_category": category,
            "gold_answer_letter": gold,
            "raw_output": raw_output,
            "predicted_letter": predicted,
            "is_valid_prediction": predicted in OPTION_LABELS,
            "is_correct": predicted == gold,
            "input_token_count": input_length,
            "generated_token_count": int(new_tokens.shape[-1]),
        }
        predictions.append(prediction)
        print(
            f"[eval {position}/{len(rows)}] {sample_id} "
            f"pred={predicted} gold={gold} correct={prediction['is_correct']}"
        )

    if was_training:
        model.train()

    total = len(predictions)
    correct = sum(bool(row["is_correct"]) for row in predictions)
    valid = sum(bool(row["is_valid_prediction"]) for row in predictions)

    categories = sorted(
        {str(row["question_category"]) for row in predictions}
    )
    by_category: dict[str, dict[str, object]] = {}
    for category in categories:
        category_rows = [
            row
            for row in predictions
            if row["question_category"] == category
        ]
        category_correct = sum(
            bool(row["is_correct"]) for row in category_rows
        )
        by_category[category] = {
            "count": len(category_rows),
            "correct": category_correct,
            "accuracy": category_correct / len(category_rows),
        }

    summary = {
        "count": total,
        "correct": correct,
        "accuracy": correct / total,
        "valid_predictions": valid,
        "valid_rate": valid / total,
        "by_question_category": by_category,
    }
    return summary, predictions


def evaluate_teacher_forced_loss(
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    video_root: str | Path,
    num_frames: int,
) -> float:
    """Measure assistant-only loss on a video-disjoint holdout."""
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
            print(
                f"[holdout loss {position}/{len(rows)}] {row['sample_id']}"
            )
            batch, _ = build_qwen3vl_sft_batch(
                processor,
                row,
                video_root=video_root,
                num_frames=num_frames,
            )
            batch = batch.to(device)
            outputs = model(
                **batch,
                use_cache=False,
            )
            losses.append(
                float(outputs.loss.detach().float().cpu().item())
            )

    if was_training:
        model.train()
    return finite_mean(losses)


def cyclic_video_override(
    rows: Sequence[Mapping[str, object]],
    *,
    video_root: str | Path,
) -> dict[str, tuple[str, Path]]:
    """Assign every sample a deterministic video belonging to another ID."""
    unique_video_ids = sorted({str(row["video_id"]) for row in rows})
    if len(unique_video_ids) < 2:
        raise ValueError("Swap-video evaluation needs at least two videos.")

    root = Path(video_root).expanduser().resolve()
    path_by_video: dict[str, Path] = {}
    for row in rows:
        video_id = str(row["video_id"])
        path = root / str(row["video_relpath"])
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        previous = path_by_video.get(video_id)
        if previous is not None and previous != path:
            raise ValueError(
                f"Video ID {video_id!r} maps to multiple physical files."
            )
        path_by_video[video_id] = path

    swap_id = {
        video_id: unique_video_ids[(index + 1) % len(unique_video_ids)]
        for index, video_id in enumerate(unique_video_ids)
    }
    return {
        str(row["sample_id"]): (
            swap_id[str(row["video_id"])],
            path_by_video[swap_id[str(row["video_id"])]],
        )
        for row in rows
    }


def swap_summary(
    correct_predictions: Sequence[Mapping[str, object]],
    swapped_predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize prediction sensitivity to deliberately mismatched videos."""
    if len(correct_predictions) != len(swapped_predictions):
        raise ValueError("Prediction sequences must have equal length.")

    correct_by_id = {
        str(row["sample_id"]): row for row in correct_predictions
    }
    swapped_by_id = {
        str(row["sample_id"]): row for row in swapped_predictions
    }
    if set(correct_by_id) != set(swapped_by_id):
        raise ValueError("Prediction sample IDs do not align.")

    changed = 0
    warning = 0
    real_correct = 0
    swap_correct = 0

    for sample_id in correct_by_id:
        real = correct_by_id[sample_id]
        swap = swapped_by_id[sample_id]
        did_change = (
            real.get("predicted_letter")
            != swap.get("predicted_letter")
        )
        changed += int(did_change)
        real_is_correct = bool(real.get("is_correct"))
        swap_is_correct = bool(swap.get("is_correct"))
        real_correct += int(real_is_correct)
        swap_correct += int(swap_is_correct)
        warning += int(
            real_is_correct
            and swap_is_correct
            and not did_change
        )

    total = len(correct_by_id)
    return {
        "count": total,
        "prediction_change_rate": changed / total,
        "correct_video_accuracy": real_correct / total,
        "swapped_video_accuracy": swap_correct / total,
        "accuracy_drop_after_swap": (
            real_correct - swap_correct
        ) / total,
        "text_memorisation_warning_rate": warning / total,
        "text_memorisation_warning_count": warning,
    }
