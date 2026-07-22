from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from longvideoguard.evaluation.frozen_videoqa import (
    FROZEN_CONDITIONS,
    attach_confidence_interval,
    write_json,
    write_jsonl,
)
from longvideoguard.evaluation.stage7_sampling import (
    answer_letter,
    format_prompt,
    load_jsonl,
    method_summary,
    paired_comparison,
    parse_answer,
    question_category,
)


def resolve_dtype(torch: Any, name: str) -> Any:
    return "auto" if name == "auto" else getattr(torch, name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered Stage 8 frozen VideoQA evaluation."
        )
    )
    parser.add_argument("frozen_sampling_manifest", type=Path)
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage8/frozen_videoqa"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The GPU does not support bfloat16.")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is non-empty: {output_dir}"
            )
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.frozen_sampling_manifest)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        condition = str(row.get("frozen_condition", ""))
        if condition not in FROZEN_CONDITIONS:
            raise ValueError(
                f"Unknown frozen condition: {condition!r}"
            )
        grouped[condition].append(row)

    for condition in FROZEN_CONDITIONS:
        grouped[condition].sort(
            key=lambda row: str(row["sample_id"])
        )
        if not grouped[condition]:
            raise ValueError(
                f"No rows for frozen condition {condition}."
            )

    reference_ids = {
        str(row["sample_id"])
        for row in grouped["scene_aware_8"]
    }
    for condition in FROZEN_CONDITIONS:
        if {
            str(row["sample_id"])
            for row in grouped[condition]
        } != reference_ids:
            raise ValueError(
                "Frozen conditions do not share identical sample IDs."
            )

    processor = AutoProcessor.from_pretrained(args.model_name)
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = None

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=resolve_dtype(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    )
    model.to("cuda")
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    project_root = args.project_root.expanduser().resolve()

    predictions_by_condition = {}

    for condition in FROZEN_CONDITIONS:
        predictions = []
        condition_rows = grouped[condition]
        print(
            f"\n=== FROZEN {condition}: "
            f"{len(condition_rows)} samples ==="
        )
        for position, row in enumerate(
            condition_rows,
            start=1,
        ):
            video_path = (
                project_root / str(row["video_relpath"])
            ).resolve()
            if not video_path.is_file():
                absolute = row.get(
                    "frozen_video_absolute_path"
                )
                if absolute:
                    video_path = Path(
                        str(absolute)
                    ).expanduser().resolve()
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Frozen video not found: {video_path}"
                )

            prompt = format_prompt(row)
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
                            "text": prompt,
                        },
                    ],
                }
            ]
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
                        "num_frames": int(
                            row["frozen_num_frames"]
                        ),
                    },
                },
            )
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(device)
            input_length = int(
                inputs["input_ids"].shape[-1]
            )

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            peak_memory = (
                torch.cuda.max_memory_allocated() / (1024**2)
            )

            new_tokens = generated[:, input_length:]
            raw_output = processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            prediction = parse_answer(raw_output)
            gold = answer_letter(row)
            correct = prediction == gold

            predictions.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "video_id": str(row["video_id"]),
                    "condition": condition,
                    "question_category": question_category(row),
                    "gold_answer_letter": gold,
                    "prediction": prediction,
                    "raw_output": raw_output,
                    "is_correct": correct,
                    "latency_seconds": latency,
                    "input_token_count": input_length,
                    "generated_token_count": int(
                        new_tokens.shape[-1]
                    ),
                    "peak_gpu_memory_mb": peak_memory,
                }
            )
            print(
                f"[{position}/{len(condition_rows)}] "
                f"{row['sample_id']} gold={gold} "
                f"pred={prediction} correct={correct}"
            )

        predictions_by_condition[condition] = predictions
        write_jsonl(
            predictions,
            output_dir / f"{condition}_predictions.jsonl",
        )

    summaries = {
        condition: attach_confidence_interval(
            method_summary(predictions)
        )
        for condition, predictions
        in predictions_by_condition.items()
    }
    comparisons = {
        "scene_aware_8_vs_uniform_8": paired_comparison(
            predictions_by_condition["uniform_8"],
            predictions_by_condition["scene_aware_8"],
        ),
        "scene_aware_8_vs_uniform_16": paired_comparison(
            predictions_by_condition["uniform_16"],
            predictions_by_condition["scene_aware_8"],
        ),
        "uniform_8_vs_uniform_16": paired_comparison(
            predictions_by_condition["uniform_16"],
            predictions_by_condition["uniform_8"],
        ),
    }

    summary = {
        "schema_version": "1.0",
        "frozen_evaluation": True,
        "preregistered_selected_policy": "scene_aware_8",
        "no_post_frozen_tuning": True,
        "experiment": {
            "model_name": args.model_name,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "gpu": torch.cuda.get_device_name(0),
        },
        "conditions": summaries,
        "paired_comparisons": comparisons,
        "interpretation_rule": (
            "Scene-aware-8 remains the final selected policy regardless "
            "of which comparator scores highest on this frozen evaluation."
        ),
    }
    summary_path = write_json(
        summary,
        output_dir / "frozen_videoqa_summary.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stage 8 frozen summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
