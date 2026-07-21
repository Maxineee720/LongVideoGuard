from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any

from longvideoguard.training.lora_overfit import (
    assert_only_lora_trainable,
    count_parameters,
    find_language_lora_targets,
    parameter_delta_summary,
    set_global_seed,
    snapshot_trainable_parameters,
)
from longvideoguard.training.sft_batch import build_qwen3vl_sft_batch
from longvideoguard.training.stage6a_training import (
    append_jsonl,
    cyclic_video_override,
    deterministic_subset,
    evaluate_generation,
    evaluate_teacher_forced_loss,
    finite_mean,
    is_better_checkpoint,
    load_jsonl,
    move_batch_to_device,
    shuffled_epoch_indices,
    swap_summary,
    update_patience,
    write_json,
    write_jsonl,
)


def resolve_dtype(torch: Any, name: str) -> Any:
    return "auto" if name == "auto" else getattr(torch, name)


def load_base_reference(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(
        path.expanduser().resolve().read_text(encoding="utf-8")
    )
    if "overall" in payload:
        return {
            "count": payload["overall"]["count"],
            "correct": payload["overall"]["correct"],
            "accuracy": payload["overall"]["accuracy"],
        }
    if "accuracy" in payload:
        return {
            "count": payload.get("count"),
            "correct": payload.get("correct"),
            "accuracy": payload["accuracy"],
        }
    raise ValueError(f"No accuracy field found in {path}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a larger QA-only Qwen3-VL LoRA with video-disjoint "
            "holdout checkpoint selection."
        )
    )
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("holdout_manifest", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        help="Optional existing 48-question development pilot.",
    )
    parser.add_argument(
        "--development-video-root",
        type=Path,
    )
    parser.add_argument(
        "--base-development-metrics",
        type=Path,
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/adapters/stage6a_qa_lora"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-num-frames", type=int, default=8)
    parser.add_argument("--holdout-num-frames", type=int, default=8)
    parser.add_argument("--development-num-frames", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--swap-samples",
        type=int,
        default=32,
        help="Number of training examples used for the final swap-video diagnostic.",
    )
    parser.add_argument("--seed", type=int, default=42)
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

    for name, value in {
        "--epochs": args.epochs,
        "--train-num-frames": args.train_num_frames,
        "--holdout-num-frames": args.holdout_num_frames,
        "--development-num-frames": args.development_num_frames,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--lora-rank": args.lora_rank,
        "--lora-alpha": args.lora_alpha,
        "--patience": args.patience,
        "--swap-samples": args.swap_samples,
    }.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.lora_dropout < 1:
        parser.error("--lora-dropout must be in [0, 1)")
    if (
        args.development_manifest is None
        != (args.development_video_root is None)
    ):
        parser.error(
            "--development-manifest and --development-video-root "
            "must be provided together"
        )

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the project VLM dependencies and PEFT before training."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 6A.2 requires a CUDA GPU.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16.")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is non-empty: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.jsonl"

    set_global_seed(args.seed)
    train_rows = load_jsonl(args.train_manifest)
    holdout_rows = load_jsonl(args.holdout_manifest)
    development_rows = (
        load_jsonl(args.development_manifest)
        if args.development_manifest is not None
        else None
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_rows)}")
    print(f"Holdout samples: {len(holdout_rows)}")
    print(
        "Development samples: "
        f"{len(development_rows) if development_rows is not None else 0}"
    )

    processor = AutoProcessor.from_pretrained(args.model_name)
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
    base_model.config.use_cache = False

    target_modules = find_language_lora_targets(base_model)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False
    model.train()

    trainable_names = assert_only_lora_trainable(model)
    parameter_counts = count_parameters(model)
    initial_snapshot = snapshot_trainable_parameters(model)

    print(json.dumps(parameter_counts, indent=2))
    print(f"Exact target modules: {len(target_modules)}")
    print(f"Trainable LoRA tensors: {len(trainable_names)}")

    print("Evaluating initial holdout generation...")
    initial_holdout_summary, initial_holdout_predictions = evaluate_generation(
        model,
        processor,
        holdout_rows,
        video_root=args.video_root,
        num_frames=args.holdout_num_frames,
    )
    initial_holdout_loss = evaluate_teacher_forced_loss(
        model,
        processor,
        holdout_rows,
        video_root=args.video_root,
        num_frames=args.holdout_num_frames,
    )
    write_jsonl(
        initial_holdout_predictions,
        output_dir / "holdout_predictions_before.jsonl",
    )
    print(
        f"Initial holdout accuracy={initial_holdout_summary['accuracy']:.4f}, "
        f"loss={initial_holdout_loss:.6f}"
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    schedules = shuffled_epoch_indices(
        num_samples=len(train_rows),
        epochs=args.epochs,
        seed=args.seed,
    )
    best_accuracy: float | None = None
    best_loss: float | None = None
    best_epoch: int | None = None
    bad_epochs = 0
    first_nonzero_gradient_norm: float | None = None
    completed_optimizer_steps = 0
    best_adapter_dir = output_dir / "best_adapter"

    for epoch_index, indices in enumerate(schedules, start=1):
        print(f"\n=== Epoch {epoch_index}/{args.epochs} ===")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        epoch_losses: list[float] = []
        pending_micro_batches = 0

        for position, sample_index in enumerate(indices, start=1):
            row = train_rows[sample_index]
            batch, _ = build_qwen3vl_sft_batch(
                processor,
                row,
                video_root=args.video_root,
                num_frames=args.train_num_frames,
            )
            batch = move_batch_to_device(
                batch,
                next(model.parameters()).device,
            )
            outputs = model(
                **batch,
                use_cache=False,
            )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss in epoch {epoch_index}, "
                    f"sample {row['sample_id']}: {loss}"
                )

            (
                loss / args.gradient_accumulation_steps
            ).backward()
            value = float(loss.detach().float().cpu().item())
            micro_losses.append(value)
            epoch_losses.append(value)
            pending_micro_batches += 1

            should_step = (
                pending_micro_batches == args.gradient_accumulation_steps
                or position == len(indices)
            )
            if not should_step:
                continue

            gradient_norm_squared = 0.0
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    gradient_norm_squared += float(
                        parameter.grad.detach().float().pow(2).sum().item()
                    )
            gradient_norm = math.sqrt(gradient_norm_squared)
            if first_nonzero_gradient_norm is None and gradient_norm > 0:
                first_nonzero_gradient_norm = gradient_norm

            clipped_norm = torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_optimizer_steps += 1

            log_row = {
                "epoch": epoch_index,
                "optimizer_step": completed_optimizer_steps,
                "samples_seen_in_step": pending_micro_batches,
                "mean_micro_loss": finite_mean(micro_losses),
                "gradient_norm_before_clip": gradient_norm,
                "gradient_norm_returned_by_clip": float(clipped_norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            append_jsonl(log_row, history_path)
            print(
                f"epoch={epoch_index} "
                f"step={completed_optimizer_steps} "
                f"loss={log_row['mean_micro_loss']:.6f} "
                f"grad={gradient_norm:.4f}"
            )
            micro_losses.clear()
            pending_micro_batches = 0

        print("Evaluating holdout...")
        holdout_summary, holdout_predictions = evaluate_generation(
            model,
            processor,
            holdout_rows,
            video_root=args.video_root,
            num_frames=args.holdout_num_frames,
        )
        holdout_loss = evaluate_teacher_forced_loss(
            model,
            processor,
            holdout_rows,
            video_root=args.video_root,
            num_frames=args.holdout_num_frames,
        )
        write_jsonl(
            holdout_predictions,
            output_dir
            / f"holdout_predictions_epoch_{epoch_index:02d}.jsonl",
        )

        improved = is_better_checkpoint(
            accuracy=float(holdout_summary["accuracy"]),
            loss=holdout_loss,
            best_accuracy=best_accuracy,
            best_loss=best_loss,
        )
        if improved:
            if best_adapter_dir.exists():
                shutil.rmtree(best_adapter_dir)
            model.save_pretrained(
                best_adapter_dir,
                safe_serialization=True,
            )
            best_accuracy = float(holdout_summary["accuracy"])
            best_loss = holdout_loss
            best_epoch = epoch_index

        bad_epochs = update_patience(
            improved=improved,
            bad_epochs=bad_epochs,
        )
        epoch_summary = {
            "epoch": epoch_index,
            "mean_train_loss": finite_mean(epoch_losses),
            "holdout_generation": holdout_summary,
            "holdout_teacher_forced_loss": holdout_loss,
            "checkpoint_improved": improved,
            "bad_epochs": bad_epochs,
        }
        append_jsonl(
            epoch_summary,
            output_dir / "epoch_history.jsonl",
        )
        print(json.dumps(epoch_summary, indent=2))

        if bad_epochs >= args.patience:
            print(
                f"Early stopping after {bad_epochs} epochs without improvement."
            )
            break

    if first_nonzero_gradient_norm is None:
        raise ValueError("No non-zero LoRA gradient was observed.")
    if best_epoch is None or not best_adapter_dir.is_dir():
        raise ValueError("No best adapter checkpoint was saved.")

    parameter_delta = parameter_delta_summary(model, initial_snapshot)
    if float(parameter_delta["delta_l2"]) <= 0:
        raise ValueError("LoRA parameters did not change.")

    print("Reloading best holdout-selected adapter...")
    del optimizer
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    best_base = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    best_base.to("cuda")
    best_model = PeftModel.from_pretrained(
        best_base,
        best_adapter_dir,
        is_trainable=False,
    )
    best_model.eval()

    final_holdout_summary, final_holdout_predictions = evaluate_generation(
        best_model,
        processor,
        holdout_rows,
        video_root=args.video_root,
        num_frames=args.holdout_num_frames,
    )
    write_jsonl(
        final_holdout_predictions,
        output_dir / "holdout_predictions_best_reloaded.jsonl",
    )

    development_summary = None
    development_predictions = None
    if development_rows is not None:
        print("Evaluating best adapter on the development pilot...")
        development_summary, development_predictions = evaluate_generation(
            best_model,
            processor,
            development_rows,
            video_root=args.development_video_root,
            num_frames=args.development_num_frames,
        )
        write_jsonl(
            development_predictions,
            output_dir / "development_predictions_best.jsonl",
        )

    swap_rows = deterministic_subset(
        train_rows,
        max_samples=min(args.swap_samples, len(train_rows)),
        seed=args.seed + 1000,
    )
    print(
        f"Running swap-video diagnostic on {len(swap_rows)} train examples..."
    )
    correct_swap_summary, correct_swap_predictions = evaluate_generation(
        best_model,
        processor,
        swap_rows,
        video_root=args.video_root,
        num_frames=args.train_num_frames,
    )
    overrides = cyclic_video_override(
        swap_rows,
        video_root=args.video_root,
    )
    swapped_summary, swapped_predictions = evaluate_generation(
        best_model,
        processor,
        swap_rows,
        video_root=args.video_root,
        num_frames=args.train_num_frames,
        video_override_by_sample=overrides,
    )
    diagnostic = swap_summary(
        correct_swap_predictions,
        swapped_predictions,
    )
    write_jsonl(
        correct_swap_predictions,
        output_dir / "swap_correct_video_predictions.jsonl",
    )
    write_jsonl(
        swapped_predictions,
        output_dir / "swap_mismatched_video_predictions.jsonl",
    )

    base_reference = load_base_reference(
        args.base_development_metrics
    )
    development_delta = None
    if development_summary is not None and base_reference is not None:
        development_delta = {
            "adapter_accuracy": development_summary["accuracy"],
            "base_accuracy": base_reference["accuracy"],
            "absolute_accuracy_delta": (
                float(development_summary["accuracy"])
                - float(base_reference["accuracy"])
            ),
            "percentage_point_delta": 100 * (
                float(development_summary["accuracy"])
                - float(base_reference["accuracy"])
            ),
        }

    processor.save_pretrained(output_dir / "processor")
    summary = {
        "schema_version": "1.0",
        "experiment": {
            "model_name": args.model_name,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "optimizer_steps_completed": completed_optimizer_steps,
            "train_num_frames": args.train_num_frames,
            "holdout_num_frames": args.holdout_num_frames,
            "development_num_frames": args.development_num_frames,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "patience": args.patience,
            "seed": args.seed,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "gpu": torch.cuda.get_device_name(0),
        },
        "data": {
            "train_samples": len(train_rows),
            "holdout_samples": len(holdout_rows),
            "development_samples": (
                len(development_rows)
                if development_rows is not None
                else 0
            ),
            "swap_samples": len(swap_rows),
        },
        "lora": {
            "target_module_count": len(target_modules),
            "parameter_counts": parameter_counts,
            "trainable_tensor_count": len(trainable_names),
            "first_nonzero_gradient_norm": first_nonzero_gradient_norm,
            "parameter_delta": parameter_delta,
        },
        "before": {
            "holdout_generation": initial_holdout_summary,
            "holdout_teacher_forced_loss": initial_holdout_loss,
        },
        "best_reloaded": {
            "holdout_generation": final_holdout_summary,
            "development_generation": development_summary,
            "base_development_reference": base_reference,
            "development_delta_from_base": development_delta,
        },
        "swap_video_diagnostic": {
            "correct_video_generation": correct_swap_summary,
            "mismatched_video_generation": swapped_summary,
            **diagnostic,
        },
        "best_adapter_path": str(best_adapter_dir),
    }
    summary_path = write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stage 6A.2 summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
