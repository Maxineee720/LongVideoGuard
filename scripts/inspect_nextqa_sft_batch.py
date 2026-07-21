from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from longvideoguard.training.sft_batch import (
    build_qwen3vl_sft_batch,
    load_sft_jsonl_row,
    write_batch_report,
)


def resolve_dtype(torch: Any, name: str) -> Any:
    return "auto" if name == "auto" else getattr(torch, name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one real Qwen3-VL video SFT batch, prove that the "
            "prompt/video prefix is masked, and optionally run a forward loss."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
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
        "--output",
        type=Path,
        default=Path(
            "outputs/training_checks/stage5_2_sft_batch_report.json"
        ),
    )
    parser.add_argument(
        "--forward-pass",
        action="store_true",
        help="Load the base model and verify that the batch produces finite loss.",
    )
    args = parser.parse_args()

    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")

    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            'VLM dependencies are missing. Install with: '
            'pip install -e ".[vlm]"'
        ) from exc

    row = load_sft_jsonl_row(
        args.manifest,
        index=args.sample_index,
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    batch, report = build_qwen3vl_sft_batch(
        processor,
        row,
        video_root=args.video_root,
        num_frames=args.num_frames,
    )

    if args.forward_pass:
        model_kwargs: dict[str, Any] = {
            "dtype": resolve_dtype(torch, args.dtype),
            "device_map": "auto",
            "attn_implementation": args.attn_implementation,
        }
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            **model_kwargs,
        )
        model.eval()
        batch = batch.to(model.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        with torch.inference_mode():
            outputs = model(
                **batch,
                use_cache=False,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        loss = float(outputs.loss.detach().float().cpu().item())
        if not torch.isfinite(outputs.loss):
            raise ValueError(f"Forward loss is not finite: {loss}")

        report["forward_pass"] = {
            "performed": True,
            "loss": loss,
            "finite_loss": True,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else None
            ),
            "model_dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        }
    else:
        report["forward_pass"] = {
            "performed": False,
        }

    destination = write_batch_report(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Batch inspection report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
