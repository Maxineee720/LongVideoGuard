from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


IGNORE_INDEX = -100


def load_sft_jsonl_row(
    path: str | Path,
    *,
    index: int = 0,
) -> dict[str, object]:
    """Load one auditable LongVideoGuard SFT row by zero-based index."""
    if index < 0:
        raise ValueError("index must be non-negative")

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SFT JSONL file not found: {source}")

    current_index = -1
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            current_index += 1
            if current_index != index:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} of {source} must contain a JSON object."
                )
            validate_sft_row(row)
            return row

    raise IndexError(
        f"Requested SFT row index {index}, but {source} contains "
        f"{current_index + 1} non-empty rows."
    )


def validate_sft_row(row: Mapping[str, object]) -> None:
    """Validate the fields required to construct a video SFT batch."""
    required = {
        "sample_id",
        "video_id",
        "video_relpath",
        "prompt",
        "assistant_target",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"SFT row is missing required fields: {missing}")

    for field in required:
        if not str(row[field]).strip():
            raise ValueError(f"SFT row field {field!r} must be non-empty.")

    target = str(row["assistant_target"]).strip()

    # Stage 1-6A ordinary NExT-QA rows supervise a single A-E answer letter.
    # Stage 6B answerability rows supervise an exact compact JSON object:
    # {"status":"answerable","answer":"D"} or
    # {"status":"unanswerable","answer":null}.
    is_answerability_task = (
        str(row.get("task", "")).strip() == "videoqa_answerability"
        or str(row.get("schema_version", "")).strip() == "2.0"
        or row.get("gold_status") in {"answerable", "unanswerable"}
    )

    if not is_answerability_task:
        if target not in {"A", "B", "C", "D", "E"}:
            raise ValueError(
                f"assistant_target must be one of A-E, got {target!r}."
            )
        return

    try:
        payload = json.loads(target)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Stage 6B assistant_target must be valid compact JSON, "
            f"got {target!r}."
        ) from exc

    if not isinstance(payload, dict) or set(payload) != {"status", "answer"}:
        raise ValueError(
            "Stage 6B assistant_target must contain exactly the keys "
            "'status' and 'answer'."
        )

    status = payload["status"]
    answer = payload["answer"]
    if status == "answerable":
        if answer not in {"A", "B", "C", "D", "E"}:
            raise ValueError(
                "An answerable Stage 6B target requires answer A-E."
            )
    elif status == "unanswerable":
        if answer is not None:
            raise ValueError(
                "An unanswerable Stage 6B target requires answer=null."
            )
    else:
        raise ValueError(
            "Stage 6B target status must be 'answerable' or "
            f"'unanswerable', got {status!r}."
        )

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if target != canonical:
        raise ValueError(
            "Stage 6B assistant_target must use the exact compact canonical "
            f"JSON form {canonical!r}, got {target!r}."
        )


def prompt_prefix_length(
    full_token_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
) -> int:
    """Return prompt length after proving it is an exact full-sequence prefix."""
    full = list(full_token_ids)
    prompt = list(prompt_token_ids)

    if not prompt:
        raise ValueError("prompt_token_ids must be non-empty")
    if len(prompt) >= len(full):
        raise ValueError(
            "The full conversation must contain at least one assistant token "
            "after the prompt prefix."
        )
    if full[: len(prompt)] != prompt:
        mismatch = next(
            (
                index
                for index, (full_id, prompt_id) in enumerate(
                    zip(full, prompt, strict=False)
                )
                if full_id != prompt_id
            ),
            None,
        )
        raise ValueError(
            "Prompt token IDs are not an exact prefix of the full "
            f"conversation. First mismatch: {mismatch}."
        )
    return len(prompt)


def masked_label_values(
    full_token_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Mask the system/user/video prefix and supervise only assistant tokens."""
    prompt_length = prompt_prefix_length(full_token_ids, prompt_token_ids)
    return [
        *([ignore_index] * prompt_length),
        *list(full_token_ids[prompt_length:]),
    ]


def build_qwen3vl_sft_messages(
    row: Mapping[str, object],
    *,
    video_root: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Path]:
    """Build prompt-only and full user+assistant multimodal conversations."""
    validate_sft_row(row)
    root = Path(video_root).expanduser().resolve()
    video_path = root / str(row["video_relpath"])
    if not video_path.is_file():
        raise FileNotFoundError(f"SFT video file not found: {video_path}")

    user_message: dict[str, object] = {
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
    assistant_message: dict[str, object] = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": str(row["assistant_target"]),
            }
        ],
    }

    prompt_messages = [user_message]
    full_messages = [user_message, assistant_message]
    return prompt_messages, full_messages, video_path


def _apply_qwen3vl_template(
    processor: Any,
    messages: list[dict[str, object]],
    *,
    num_frames: int,
    add_generation_prompt: bool,
) -> Any:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
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


def build_qwen3vl_sft_batch(
    processor: Any,
    row: Mapping[str, object],
    *,
    video_root: str | Path,
    num_frames: int,
) -> tuple[Any, dict[str, object]]:
    """Build one real video batch with assistant-only labels.

    The same deterministic frame budget is applied to the prompt-only and full
    conversations. The prompt token sequence must be an exact prefix of the
    full sequence before labels are created.
    """
    prompt_messages, full_messages, video_path = build_qwen3vl_sft_messages(
        row,
        video_root=video_root,
    )

    # Qwen3-VL video processors may carry a default FPS sampler. Fixed-frame
    # experiments must disable it so only num_frames controls sampling.
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = None

    prompt_inputs = _apply_qwen3vl_template(
        processor,
        prompt_messages,
        num_frames=num_frames,
        add_generation_prompt=True,
    )
    full_inputs = _apply_qwen3vl_template(
        processor,
        full_messages,
        num_frames=num_frames,
        add_generation_prompt=False,
    )

    full_ids = full_inputs["input_ids"][0]
    prompt_ids = prompt_inputs["input_ids"][0]
    label_values = masked_label_values(
        full_ids.tolist(),
        prompt_ids.tolist(),
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required for SFT batch construction. Install with: '
            'pip install -e ".[vlm]"'
        ) from exc

    labels = torch.tensor(
        [label_values],
        dtype=full_inputs["input_ids"].dtype,
        device=full_inputs["input_ids"].device,
    )

    full_inputs.pop("token_type_ids", None)
    full_inputs["labels"] = labels

    supervised_mask = labels[0] != IGNORE_INDEX
    supervised_ids = labels[0][supervised_mask]
    supervised_with_specials = processor.tokenizer.decode(
        supervised_ids.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    supervised_clean = processor.tokenizer.decode(
        supervised_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    video_keys = [
        key
        for key in full_inputs.keys()
        if "video" in key.lower()
        or key in {"pixel_values_videos", "video_grid_thw", "second_per_grid_ts"}
    ]
    video_tensor_present = any(
        key in full_inputs
        for key in {"pixel_values_videos", "video_grid_thw"}
    )
    if not video_tensor_present:
        raise ValueError(
            "The processor batch does not contain video tensors/grid metadata."
        )

    report = {
        "sample_id": str(row["sample_id"]),
        "video_path": str(video_path),
        "num_frames_requested": num_frames,
        "batch_keys": sorted(full_inputs.keys()),
        "video_keys": sorted(video_keys),
        "full_sequence_length": int(full_ids.shape[-1]),
        "prompt_prefix_length": int(prompt_ids.shape[-1]),
        "supervised_token_count": int(supervised_mask.sum().item()),
        "masked_token_count": int((~supervised_mask).sum().item()),
        "supervision_ratio": float(
            supervised_mask.sum().item() / labels.shape[-1]
        ),
        "assistant_target": str(row["assistant_target"]),
        "supervised_token_ids": supervised_ids.tolist(),
        "supervised_text_with_special_tokens": supervised_with_specials,
        "supervised_text_clean": supervised_clean,
        "prompt_is_exact_prefix": True,
        "all_prompt_labels_masked": bool(
            (labels[0, : prompt_ids.shape[-1]] == IGNORE_INDEX).all().item()
        ),
        "all_assistant_labels_supervised": bool(
            (labels[0, prompt_ids.shape[-1] :] != IGNORE_INDEX).all().item()
        ),
        "tensor_shapes": {
            key: list(value.shape)
            for key, value in full_inputs.items()
            if hasattr(value, "shape")
        },
        "tensor_dtypes": {
            key: str(value.dtype)
            for key, value in full_inputs.items()
            if hasattr(value, "dtype")
        },
    }

    clean_target = str(row["assistant_target"]).strip()
    if clean_target not in supervised_clean:
        raise ValueError(
            "The decoded supervised suffix does not contain the assistant "
            f"target {clean_target!r}: {supervised_clean!r}"
        )

    return full_inputs, report


def write_batch_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
