from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from longvideoguard.nextqa_prompt import parse_option_letter
from longvideoguard.training.adapter_evaluation import (
    answer_letter,
    classification_summary,
    cyclic_swap_video_ids,
    delta_from_baseline,
    format_prompt,
    load_jsonl,
    question_category,
    swap_video_summary,
    video_path_by_id,
    write_json,
    write_jsonl,
)


def resolve_dtype(torch: Any, name: str) -> Any:
    return "auto" if name == "auto" else getattr(torch, name)


def build_inputs(
    processor: Any,
    *,
    video_path: Path,
    prompt: str,
    num_frames: int,
) -> Any:
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
                "num_frames": num_frames,
            },
        },
    )
    inputs.pop("token_type_ids", None)
    return inputs


def evaluate_rows(
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    video_root: str | Path,
    num_frames: int,
    video_override_by_sample: Mapping[str, tuple[str, Path]] | None = None,
    label: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import torch

    model.eval()
    device = next(model.parameters()).device
    default_paths = video_path_by_id(rows, video_root=video_root)
    predictions: list[dict[str, object]] = []

    for position, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        original_video_id = str(row["video_id"])

        if video_override_by_sample is None:
            video_id_used = original_video_id
            video_path = default_paths[original_video_id]
        else:
            video_id_used, video_path = video_override_by_sample[sample_id]

        inputs = build_inputs(
            processor,
            video_path=video_path,
            prompt=format_prompt(row),
            num_frames=num_frames,
        )
        inputs = inputs.to(device)
        input_length = int(inputs["input_ids"].shape[-1])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start

        new_tokens = generated[:, input_length:]
        raw_output = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        predicted = parse_option_letter(raw_output)
        gold = answer_letter(row)

        prediction = {
            "evaluation_label": label,
            "sample_id": sample_id,
            "source_video_id": original_video_id,
            "video_id_used": video_id_used,
            "video_path_used": str(video_path),
            "question_category": question_category(row),
            "gold_answer_letter": gold,
            "raw_output": raw_output,
            "predicted_letter": predicted,
            "is_valid_prediction": predicted in {"A", "B", "C", "D", "E"},
            "is_correct": predicted == gold,
            "generation_latency_seconds": latency,
            "input_token_count": input_length,
            "generated_token_count": int(new_tokens.shape[-1]),
        }
        predictions.append(prediction)
        print(
            f"[{label} {position}/{len(rows)}] {sample_id} "
            f"pred={predicted} gold={gold} "
            f"correct={prediction['is_correct']}"
        )

    return classification_summary(predictions), predictions


def load_optional_baseline(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "overall" in payload:
        overall = payload["overall"]
        return {
            "count": overall["count"],
            "correct": overall["correct"],
            "accuracy": overall["accuracy"],
        }
    if "accuracy" in payload:
        return {
            "count": payload.get("count"),
            "correct": payload.get("correct"),
            "accuracy": payload["accuracy"],
        }
    raise ValueError(
        f"Could not find an accuracy field in baseline metrics: {source}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved Qwen3-VL LoRA adapter on the 48-question "
            "development pilot and run a cyclic swap-video diagnostic on "
            "the 16 overfit training questions."
        )
    )
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("pilot_manifest", type=Path)
    parser.add_argument("pilot_video_root", type=Path)
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("train_video_root", type=Path)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        help="Optional saved processor directory from Stage 5.3.",
    )
    parser.add_argument(
        "--base-pilot-metrics",
        type=Path,
        help="Optional Stage 4 metrics JSON for an automatic accuracy delta.",
    )
    parser.add_argument(
        "--pilot-num-frames",
        type=int,
        default=16,
        help="Frame budget for the development-pilot comparison.",
    )
    parser.add_argument(
        "--swap-num-frames",
        type=int,
        default=8,
        help="Frame budget for the overfit train and swap-video diagnostic.",
    )
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
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluations/stage5_4"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.pilot_num_frames <= 0:
        parser.error("--pilot-num-frames must be positive")
    if args.swap_num_frames <= 0:
        parser.error("--swap-num-frames must be positive")

    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Stage 5.4 dependencies are missing. Install the project and "
            "PEFT before running this script."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5.4 requires a CUDA GPU.")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is non-empty: {output_dir}. "
                "Use --overwrite to replace it."
            )
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = args.adapter_dir.expanduser().resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Adapter configuration not found in {adapter_dir}."
        )

    processor_source = (
        args.processor_dir.expanduser().resolve()
        if args.processor_dir is not None
        else args.model_name
    )
    processor = AutoProcessor.from_pretrained(processor_source)
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = None

    model_kwargs: dict[str, Any] = {
        "dtype": resolve_dtype(torch, args.dtype),
        "attn_implementation": args.attn_implementation,
    }
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    base_model.to("cuda")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
    )
    model.eval()

    pilot_rows = load_jsonl(args.pilot_manifest)
    train_rows = load_jsonl(args.train_manifest)

    print("Evaluating adapter on development pilot...")
    pilot_summary, pilot_predictions = evaluate_rows(
        model,
        processor,
        pilot_rows,
        video_root=args.pilot_video_root,
        num_frames=args.pilot_num_frames,
        label="pilot_correct_video",
    )
    write_jsonl(
        pilot_predictions,
        output_dir / "pilot_adapter_predictions.jsonl",
    )

    print("Evaluating overfit train questions with correct videos...")
    train_correct_summary, train_correct_predictions = evaluate_rows(
        model,
        processor,
        train_rows,
        video_root=args.train_video_root,
        num_frames=args.swap_num_frames,
        label="train_correct_video",
    )
    write_jsonl(
        train_correct_predictions,
        output_dir / "train_correct_video_predictions.jsonl",
    )

    print("Building stable cyclic video swaps...")
    path_mapping = video_path_by_id(
        train_rows,
        video_root=args.train_video_root,
    )
    swap_ids = cyclic_swap_video_ids(train_rows)
    override_by_sample = {
        str(row["sample_id"]): (
            swap_ids[str(row["video_id"])],
            path_mapping[swap_ids[str(row["video_id"])]],
        )
        for row in train_rows
    }

    print("Evaluating the same train questions with mismatched videos...")
    train_swapped_summary, train_swapped_predictions = evaluate_rows(
        model,
        processor,
        train_rows,
        video_root=args.train_video_root,
        num_frames=args.swap_num_frames,
        video_override_by_sample=override_by_sample,
        label="train_swapped_video",
    )
    write_jsonl(
        train_swapped_predictions,
        output_dir / "train_swapped_video_predictions.jsonl",
    )

    swap_summary = swap_video_summary(
        train_correct_predictions,
        train_swapped_predictions,
    )
    write_jsonl(
        swap_summary["comparisons"],
        output_dir / "swap_video_comparisons.jsonl",
    )

    baseline = load_optional_baseline(args.base_pilot_metrics)
    summary = {
        "schema_version": "1.0",
        "experiment": {
            "model_name": args.model_name,
            "adapter_dir": str(adapter_dir),
            "processor_source": str(processor_source),
            "pilot_num_frames": args.pilot_num_frames,
            "swap_num_frames": args.swap_num_frames,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "gpu": torch.cuda.get_device_name(0),
        },
        "pilot_adapter": pilot_summary,
        "pilot_base_reference": baseline,
        "pilot_delta_from_base": delta_from_baseline(
            pilot_summary,
            baseline,
        ),
        "train_correct_video": train_correct_summary,
        "train_swapped_video": train_swapped_summary,
        "swap_video_diagnostic": {
            key: value
            for key, value in swap_summary.items()
            if key != "comparisons"
        },
    }
    summary_path = write_json(summary, output_dir / "summary.json")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stage 5.4 summary: {summary_path}")

    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
