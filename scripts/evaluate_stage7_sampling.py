from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from longvideoguard.evaluation.stage7_sampling import (
    METHODS,
    answer_letter,
    format_prompt,
    load_jsonl,
    method_summary,
    paired_comparison,
    parse_answer,
    question_category,
    write_json,
    write_jsonl,
)


def resolve_dtype(torch: Any, name: str) -> Any:
    return "auto" if name == "auto" else getattr(torch, name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Uniform, Scene-aware, and Query-aware selected-frame "
            "clips using the same Qwen3-VL model, prompt, and eight-frame "
            "budget."
        )
    )
    parser.add_argument("retrieval_manifest", type=Path)
    parser.add_argument(
        "project_root",
        type=Path,
        help=(
            "Repository root used to resolve video_relpath entries such as "
            "outputs/stage7/frame_retrieval/samples/..."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage7/sampling_evaluation"),
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
        "--max-samples-per-method",
        type=int,
        help="Optional smoke-test limit for each method.",
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
            "Install PyTorch and Transformers before running Stage 7B."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 7B requires a CUDA GPU.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16.")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite."
            )
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.retrieval_manifest)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("stage7_sampling_method", ""))
        if method not in METHODS:
            raise ValueError(
                f"Unknown sampling method {method!r} in "
                f"{row.get('sample_id')!r}."
            )
        grouped[method].append(row)

    for method in METHODS:
        if method not in grouped:
            raise ValueError(f"Manifest has no rows for method {method}.")
        grouped[method].sort(key=lambda row: str(row["sample_id"]))
        if args.max_samples_per_method is not None:
            grouped[method] = grouped[method][
                : args.max_samples_per_method
            ]

    reference_ids = {
        str(row["sample_id"]) for row in grouped["uniform"]
    }
    for method in METHODS[1:]:
        method_ids = {
            str(row["sample_id"]) for row in grouped[method]
        }
        if method_ids != reference_ids:
            raise ValueError(
                "Every method must contain the same sample IDs."
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
    all_predictions: dict[str, list[dict[str, object]]] = {}

    for method in METHODS:
        predictions: list[dict[str, object]] = []
        method_rows = grouped[method]
        print(f"\n=== {method}: {len(method_rows)} samples ===")

        for position, row in enumerate(method_rows, start=1):
            sample_id = str(row["sample_id"])
            video_path = (
                project_root / str(row["video_relpath"])
            ).resolve()
            if not video_path.is_file():
                absolute = row.get("stage7_clip_absolute_path")
                if absolute:
                    video_path = Path(str(absolute)).expanduser().resolve()
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Selected-frame clip not found: {video_path}"
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
                        "num_frames": args.num_frames,
                    },
                },
            )
            inputs.pop("token_type_ids", None)
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

            result = {
                "sample_id": sample_id,
                "video_id": str(row["video_id"]),
                "sampling_method": method,
                "question_category": question_category(row),
                "gold_answer_letter": gold,
                "prediction": prediction,
                "raw_output": raw_output,
                "is_correct": correct,
                "latency_seconds": latency,
                "input_token_count": input_length,
                "generated_token_count": int(new_tokens.shape[-1]),
                "peak_gpu_memory_mb": peak_memory,
                "selected_timestamps_seconds": row.get(
                    "stage7_selected_timestamps_seconds"
                ),
                "mean_query_similarity": row.get(
                    "stage7_mean_query_similarity"
                ),
                "pairwise_redundancy": row.get(
                    "stage7_pairwise_redundancy"
                ),
                "temporal_coverage": row.get(
                    "stage7_temporal_coverage"
                ),
                "selected_clip": str(video_path),
            }
            predictions.append(result)

            print(
                f"[{position}/{len(method_rows)}] {sample_id} "
                f"gold={gold} pred={prediction} correct={correct} "
                f"latency={latency:.3f}s"
            )

        all_predictions[method] = predictions
        write_jsonl(
            predictions,
            output_dir / f"{method}_predictions.jsonl",
        )

    summaries = {
        method: method_summary(all_predictions[method])
        for method in METHODS
    }
    comparisons = {
        method: paired_comparison(
            all_predictions["uniform"],
            all_predictions[method],
        )
        for method in ("scene_aware", "query_aware")
    }

    summary = {
        "schema_version": "1.0",
        "experiment": {
            "model_name": args.model_name,
            "num_frames": args.num_frames,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "gpu": torch.cuda.get_device_name(0),
            "same_sample_ids_across_methods": True,
            "paired_evaluation": True,
        },
        "methods": summaries,
        "paired_comparisons_vs_uniform": comparisons,
        "interpretation_notes": [
            (
                "All methods use the same model, prompt, question set, and "
                "eight-frame budget. Only the selected frame content differs."
            ),
            (
                "McNemar p-values are exact paired tests over the same "
                "questions; small sample size limits statistical power."
            ),
        ],
    }
    summary_path = write_json(
        summary,
        output_dir / "sampling_evaluation_summary.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stage 7B summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
