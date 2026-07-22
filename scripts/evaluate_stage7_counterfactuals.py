from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from longvideoguard.evaluation.counterfactuals import (
    ALL_CONDITIONS,
    diagnostic_summary,
    subset_by_category,
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


def prepare_inputs(
    processor: Any,
    row: Mapping[str, object],
    *,
    condition: str,
    video_path: Path | None,
    num_frames: int,
) -> Any:
    prompt = format_prompt(row)

    if condition == "question_only":
        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]
    else:
        if video_path is None:
            raise ValueError(
                f"{condition} requires a video path."
            )
        content = [
            {
                "type": "video",
                "url": str(video_path),
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]

    messages = [{"role": "user", "content": content}]
    processor_kwargs: dict[str, object] = {
        "text_kwargs": {
            "return_tensors": "pt",
        },
    }
    if condition != "question_only":
        processor_kwargs["videos_kwargs"] = {
            "do_sample_frames": True,
            "num_frames": num_frames,
        }

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs=processor_kwargs,
    )
    inputs.pop("token_type_ids", None)
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate question-only, black-video, temporal-order, and "
            "evidence-removal counterfactuals with Qwen3-VL."
        )
    )
    parser.add_argument("counterfactual_manifest", type=Path)
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/stage7/counterfactual_evaluation"
        ),
    )
    parser.add_argument("--num-frames", type=int, default=8)
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
    parser.add_argument(
        "--max-samples-per-condition",
        type=int,
        help="Optional smoke-test limit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

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

    manifest_rows = load_jsonl(args.counterfactual_manifest)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[str(row["counterfactual_condition"])].append(row)

    for condition in grouped:
        grouped[condition].sort(key=lambda row: str(row["sample_id"]))
        if args.max_samples_per_condition is not None:
            grouped[condition] = grouped[condition][
                : args.max_samples_per_condition
            ]

    if "original" not in grouped:
        raise ValueError("Manifest has no original rows.")

    question_rows = [dict(row) for row in grouped["original"]]
    grouped["question_only"] = question_rows

    reference_ids = {
        str(row["sample_id"]) for row in grouped["original"]
    }
    for condition in ALL_CONDITIONS:
        if condition not in grouped:
            raise ValueError(f"Missing condition: {condition}")
        ids = {str(row["sample_id"]) for row in grouped[condition]}
        if ids != reference_ids:
            raise ValueError(
                f"Condition {condition} has different sample IDs."
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

    predictions_by_condition: dict[
        str,
        list[dict[str, object]],
    ] = {}

    for condition in ALL_CONDITIONS:
        rows = grouped[condition]
        print(f"\n=== {condition}: {len(rows)} samples ===")
        predictions: list[dict[str, object]] = []

        for position, row in enumerate(rows, start=1):
            video_path: Path | None = None
            if condition != "question_only":
                video_path = (
                    project_root / str(row["video_relpath"])
                ).resolve()
                if not video_path.is_file():
                    absolute = row.get(
                        "counterfactual_clip_absolute_path"
                    )
                    if absolute:
                        video_path = Path(
                            str(absolute)
                        ).expanduser().resolve()
                if not video_path.is_file():
                    raise FileNotFoundError(
                        f"Counterfactual clip not found: {video_path}"
                    )

            inputs = prepare_inputs(
                processor,
                row,
                condition=condition,
                video_path=video_path,
                num_frames=args.num_frames,
            )
            inputs = inputs.to(device)
            input_length = int(inputs["input_ids"].shape[-1])

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

            prediction_row = {
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
                "generated_token_count": int(new_tokens.shape[-1]),
                "peak_gpu_memory_mb": peak_memory,
                "masked_indices": row.get(
                    "counterfactual_masked_indices"
                ),
            }
            predictions.append(prediction_row)
            print(
                f"[{position}/{len(rows)}] {row['sample_id']} "
                f"gold={gold} pred={prediction} correct={correct}"
            )

        predictions_by_condition[condition] = predictions
        write_jsonl(
            predictions,
            output_dir / f"{condition}_predictions.jsonl",
        )

    method_summaries = {
        condition: method_summary(rows)
        for condition, rows in predictions_by_condition.items()
    }
    paired = {
        condition: paired_comparison(
            predictions_by_condition["original"],
            predictions_by_condition[condition],
        )
        for condition in ALL_CONDITIONS
        if condition != "original"
    }

    temporal_paired = {}
    for condition in ("reversed", "shuffled"):
        temporal_paired[condition] = paired_comparison(
            subset_by_category(
                predictions_by_condition["original"],
                "temporal",
            ),
            subset_by_category(
                predictions_by_condition[condition],
                "temporal",
            ),
        )

    summary = {
        "schema_version": "1.0",
        "experiment": {
            "model_name": args.model_name,
            "num_frames": args.num_frames,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "gpu": torch.cuda.get_device_name(0),
            "base_sampling_method": "scene_aware",
        },
        "conditions": method_summaries,
        "paired_comparisons_vs_original": paired,
        "temporal_subset_paired_comparisons": temporal_paired,
        "diagnostics": diagnostic_summary(
            predictions_by_condition
        ),
        "interpretation_notes": [
            (
                "A small or negative original-minus-question-only gap "
                "indicates strong text priors."
            ),
            (
                "Temporal accuracy should fall more under reversal/shuffle "
                "than descriptive accuracy when frame order is used."
            ),
            (
                "A larger relevant-mask drop than random-mask drop supports "
                "question-specific evidence sensitivity."
            ),
        ],
    }
    summary_path = write_json(
        summary,
        output_dir / "counterfactual_evaluation_summary.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stage 7C.2 summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
