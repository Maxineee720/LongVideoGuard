from __future__ import annotations

import gc
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

OPTION_LABELS = ("A", "B", "C", "D", "E")
DEFAULT_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
VISION_NAME_MARKERS = ("visual", "vision", "merger")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_sft_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load a non-empty LongVideoGuard SFT JSONL file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SFT JSONL file not found: {source}")

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
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing required fields: {missing}"
                )

            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample_id!r}")
            seen.add(sample_id)

            target = str(row["assistant_target"]).strip()
            if target not in OPTION_LABELS:
                raise ValueError(
                    f"Line {line_number}: invalid assistant target {target!r}."
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"SFT file is empty: {source}")
    return rows


def optimizer_step_sample_indices(
    *,
    num_samples: int,
    optimizer_steps: int,
    gradient_accumulation_steps: int,
    seed: int,
) -> list[list[int]]:
    """Return deterministic shuffled sample indices for each optimizer step."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if optimizer_steps <= 0:
        raise ValueError("optimizer_steps must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    rng = random.Random(seed)
    pool: list[int] = []
    schedule: list[list[int]] = []

    for _ in range(optimizer_steps):
        while len(pool) < gradient_accumulation_steps:
            epoch = list(range(num_samples))
            rng.shuffle(epoch)
            pool.extend(epoch)

        current = pool[:gradient_accumulation_steps]
        del pool[:gradient_accumulation_steps]
        schedule.append(current)

    return schedule


def find_language_lora_targets(
    model: Any,
    *,
    suffixes: Sequence[str] = DEFAULT_TARGET_SUFFIXES,
) -> list[str]:
    """Find exact LLM linear-module names while excluding the visual tower."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect model modules.") from exc

    normalized_suffixes = tuple(str(item) for item in suffixes)
    if not normalized_suffixes:
        raise ValueError("At least one LoRA target suffix is required.")

    targets: list[str] = []
    visual_matches: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.endswith(normalized_suffixes):
            continue

        lowered = name.lower()
        if any(marker in lowered for marker in VISION_NAME_MARKERS):
            visual_matches.append(name)
            continue
        targets.append(name)

    if not targets:
        raise ValueError(
            "No language-model LoRA targets were found for suffixes "
            f"{normalized_suffixes}."
        )

    # Exact full module names are returned, so PEFT does not have to apply
    # suffix matching to unrelated modules elsewhere in the architecture.
    return sorted(set(targets))


def count_parameters(model: Any) -> dict[str, int | float]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": (
            100.0 * trainable / total if total else 0.0
        ),
    }


def assert_only_lora_trainable(model: Any) -> list[str]:
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names:
        raise ValueError("The model has no trainable parameters.")
    unexpected = [
        name
        for name in trainable_names
        if "lora_" not in name
    ]
    if unexpected:
        raise ValueError(
            "Non-LoRA parameters are unexpectedly trainable: "
            f"{unexpected[:10]}"
        )
    return trainable_names


def snapshot_trainable_parameters(model: Any) -> dict[str, Any]:
    """Copy all trainable tensors to CPU for an exact update check."""
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameter_delta_summary(
    model: Any,
    before: Mapping[str, Any],
) -> dict[str, float | int]:
    """Calculate the L2 parameter change after optimisation."""
    squared_delta = 0.0
    squared_before = 0.0
    changed_tensors = 0
    compared_tensors = 0

    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if set(current) != set(before):
        missing = sorted(set(before) - set(current))
        added = sorted(set(current) - set(before))
        raise ValueError(
            "Trainable parameter names changed during training. "
            f"Missing={missing[:5]}, added={added[:5]}"
        )

    for name, parameter in current.items():
        compared_tensors += 1
        old = before[name]
        new = parameter.detach().float().cpu()
        delta = new - old
        delta_squared = float(delta.pow(2).sum().item())
        squared_delta += delta_squared
        squared_before += float(old.pow(2).sum().item())
        if delta_squared > 0:
            changed_tensors += 1

    delta_l2 = math.sqrt(squared_delta)
    before_l2 = math.sqrt(squared_before)
    return {
        "compared_tensors": compared_tensors,
        "changed_tensors": changed_tensors,
        "delta_l2": delta_l2,
        "initial_l2": before_l2,
        "relative_delta": delta_l2 / before_l2 if before_l2 else 0.0,
    }


def finite_mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("Cannot calculate the mean of an empty sequence.")
    if not all(math.isfinite(value) for value in materialized):
        raise ValueError(f"Non-finite value encountered: {materialized}")
    return sum(materialized) / len(materialized)


def write_jsonl(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return destination


def move_batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def precompute_teacher_forcing_batches(
    processor: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    video_root: str | Path,
    num_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    """Decode/process each small overfit sample once and retain CPU tensors."""
    from longvideoguard.training.sft_batch import build_qwen3vl_sft_batch

    batches: list[dict[str, Any]] = []
    reports: list[dict[str, object]] = []

    for position, row in enumerate(rows, start=1):
        print(
            f"[teacher batch {position}/{len(rows)}] "
            f"{row['sample_id']}"
        )
        batch, report = build_qwen3vl_sft_batch(
            processor,
            row,
            video_root=video_root,
            num_frames=num_frames,
        )
        batches.append(dict(batch))
        reports.append(report)

    return batches, reports


def precompute_generation_batches(
    processor: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    video_root: str | Path,
    num_frames: int,
) -> list[dict[str, Any]]:
    """Build prompt-only multimodal inputs used for deterministic evaluation."""
    from longvideoguard.training.sft_batch import build_qwen3vl_sft_messages

    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = None

    batches: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        print(
            f"[generation batch {position}/{len(rows)}] "
            f"{row['sample_id']}"
        )
        prompt_messages, _, _ = build_qwen3vl_sft_messages(
            row,
            video_root=video_root,
        )
        inputs = processor.apply_chat_template(
            prompt_messages,
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
        batches.append(dict(inputs))
    return batches


def evaluate_generation(
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, object]],
    generation_batches: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int = 4,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Generate A-E predictions and return aggregate accuracy plus raw rows."""
    from longvideoguard.nextqa_prompt import parse_option_letter

    if len(rows) != len(generation_batches):
        raise ValueError("rows and generation_batches must have equal length")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for generation evaluation.") from exc

    was_training = bool(model.training)
    model.eval()
    device = next(model.parameters()).device
    predictions: list[dict[str, object]] = []

    for row, cpu_batch in zip(rows, generation_batches, strict=True):
        batch = move_batch_to_device(cpu_batch, device)
        input_length = int(batch["input_ids"].shape[-1])

        with torch.inference_mode():
            generated = model.generate(
                **batch,
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
        gold = str(row["assistant_target"]).strip()

        predictions.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "gold_answer_letter": gold,
                "raw_output": raw_output,
                "predicted_letter": predicted,
                "is_valid_prediction": predicted in OPTION_LABELS,
                "is_correct": predicted == gold,
            }
        )

    if was_training:
        model.train()

    correct = sum(bool(row["is_correct"]) for row in predictions)
    valid = sum(bool(row["is_valid_prediction"]) for row in predictions)
    total = len(predictions)

    summary = {
        "count": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "valid_predictions": valid,
        "valid_rate": valid / total if total else None,
    }
    return summary, predictions


def mean_teacher_forced_loss(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
) -> float:
    """Measure mean assistant-only loss without updating parameters."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for loss evaluation.") from exc

    was_training = bool(model.training)
    model.eval()
    device = next(model.parameters()).device
    losses: list[float] = []

    with torch.inference_mode():
        for cpu_batch in batches:
            batch = move_batch_to_device(cpu_batch, device)
            outputs = model(
                **batch,
                use_cache=False,
            )
            losses.append(float(outputs.loss.detach().float().cpu().item()))

    if was_training:
        model.train()
    return finite_mean(losses)


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
