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
from longvideoguard.training.stage6b_training import (
    append_jsonl,
    build_positive_eval_rows,
    evaluate_generation,
    evaluate_teacher_forced_loss,
    finite_mean,
    is_better_checkpoint,
    load_flexible_qa_jsonl,
    load_training_jsonl,
    move_batch_to_device,
    shuffled_epoch_indices,
    write_json,
    write_jsonl,
)


def resolve_dtype(torch: Any, name: str) -> Any:
    return "auto" if name == "auto" else getattr(torch, name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train Stage 6B structured answerability LoRA with hard negatives "
            "and checkpoint selection that includes the epoch-0 base model."
        )
    )
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("holdout_manifest", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        help=(
            "Optional ordinary QA development manifest. It is converted into "
            "structured answerable-only rows inside this script."
        ),
    )
    parser.add_argument("--development-video-root", type=Path)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/adapters/stage6b_answerability_lora"),
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
        raise RuntimeError("Stage 6B.2 requires a CUDA GPU.")
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

    set_global_seed(args.seed)
    train_rows = load_training_jsonl(args.train_manifest)
    holdout_rows = load_training_jsonl(args.holdout_manifest)

    development_rows = None
    if args.development_manifest is not None:
        ordinary_development_rows = load_flexible_qa_jsonl(
            args.development_manifest
        )
        development_rows = build_positive_eval_rows(
            ordinary_development_rows,
            role="development_positive",
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train examples: {len(train_rows)}")
    print(f"Holdout examples: {len(holdout_rows)}")
    print(
        "Development positives: "
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

    print("\n=== Epoch 0: base model candidate ===")
    base_holdout_metrics, base_holdout_predictions = evaluate_generation(
        model,
        processor,
        holdout_rows,
        video_root=args.video_root,
        num_frames=args.holdout_num_frames,
    )
    base_holdout_loss = evaluate_teacher_forced_loss(
        model,
        processor,
        holdout_rows,
        video_root=args.video_root,
        num_frames=args.holdout_num_frames,
    )
    write_jsonl(
        base_holdout_predictions,
        output_dir / "holdout_predictions_epoch_00_base.jsonl",
    )

    base_development_metrics = None
    base_development_predictions = None
    if development_rows is not None:
        base_development_metrics, base_development_predictions = (
            evaluate_generation(
                model,
                processor,
                development_rows,
                video_root=args.development_video_root,
                num_frames=args.development_num_frames,
            )
        )
        write_jsonl(
            base_development_predictions,
            output_dir / "development_predictions_epoch_00_base.jsonl",
        )

    best_epoch = 0
    best_checkpoint_type = "base"
    best_metrics = base_holdout_metrics
    best_loss = base_holdout_loss
    best_adapter_dir = output_dir / "best_adapter"
    bad_epochs = 0

    append_jsonl(
        {
            "epoch": 0,
            "checkpoint_type": "base",
            "holdout_generation": base_holdout_metrics,
            "holdout_teacher_forced_loss": base_holdout_loss,
            "selected_as_best": True,
            "bad_epochs": 0,
        },
        output_dir / "epoch_history.jsonl",
    )
    print(
        "Base holdout balanced score="
        f"{base_holdout_metrics['balanced_task_score']:.4f}, "
        "answerable accuracy="
        f"{base_holdout_metrics['answerable_exact_accuracy']:.4f}, "
        "unanswerable recall="
        f"{base_holdout_metrics['unanswerable_recall']:.4f}, "
        f"loss={base_holdout_loss:.6f}"
    )

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    schedules = shuffled_epoch_indices(
        num_samples=len(train_rows),
        epochs=args.epochs,
        seed=args.seed,
    )
    first_nonzero_gradient_norm: float | None = None
    completed_optimizer_steps = 0

    for epoch_index, indices in enumerate(schedules, start=1):
        print(f"\n=== Epoch {epoch_index}/{args.epochs} ===")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending_micro_batches = 0
        step_losses: list[float] = []
        epoch_losses: list[float] = []

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
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss in epoch {epoch_index}, "
                    f"sample {row['sample_id']}: {loss}"
                )

            (
                loss / args.gradient_accumulation_steps
            ).backward()
            loss_value = float(loss.detach().float().cpu().item())
            step_losses.append(loss_value)
            epoch_losses.append(loss_value)
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

            history_row = {
                "epoch": epoch_index,
                "optimizer_step": completed_optimizer_steps,
                "samples_seen_in_step": pending_micro_batches,
                "mean_micro_loss": finite_mean(step_losses),
                "gradient_norm_before_clip": gradient_norm,
                "gradient_norm_returned_by_clip": float(clipped_norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            append_jsonl(
                history_row,
                output_dir / "training_history.jsonl",
            )
            print(
                f"epoch={epoch_index} "
                f"step={completed_optimizer_steps} "
                f"loss={history_row['mean_micro_loss']:.6f} "
                f"grad={gradient_norm:.4f}"
            )
            pending_micro_batches = 0
            step_losses.clear()

        print("Evaluating mixed holdout...")
        holdout_metrics, holdout_predictions = evaluate_generation(
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
            holdout_metrics,
            candidate_loss=holdout_loss,
            best_metrics=best_metrics,
            best_loss=best_loss,
        )
        if improved:
            if best_adapter_dir.exists():
                shutil.rmtree(best_adapter_dir)
            model.save_pretrained(
                best_adapter_dir,
                safe_serialization=True,
            )
            best_epoch = epoch_index
            best_checkpoint_type = "adapter"
            best_metrics = holdout_metrics
            best_loss = holdout_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_summary = {
            "epoch": epoch_index,
            "checkpoint_type": "adapter_candidate",
            "mean_train_loss": finite_mean(epoch_losses),
            "holdout_generation": holdout_metrics,
            "holdout_teacher_forced_loss": holdout_loss,
            "selected_as_best": improved,
            "current_best_epoch": best_epoch,
            "current_best_checkpoint_type": best_checkpoint_type,
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

    parameter_delta = parameter_delta_summary(model, initial_snapshot)
    if float(parameter_delta["delta_l2"]) <= 0:
        raise ValueError("LoRA parameters did not change.")

    print(
        f"Selected checkpoint: {best_checkpoint_type}, epoch={best_epoch}"
    )

    del optimizer
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    selected_base = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    selected_base.to("cuda")
    if best_checkpoint_type == "adapter":
        selected_model = PeftModel.from_pretrained(
            selected_base,
            best_adapter_dir,
            is_trainable=False,
        )
    else:
        selected_model = selected_base
    selected_model.eval()

    final_holdout_metrics, final_holdout_predictions = evaluate_generation(
        selected_model,
        processor,
        holdout_rows,
        video_root=args.video_root,
        num_frames=args.holdout_num_frames,
    )
    write_jsonl(
        final_holdout_predictions,
        output_dir / "holdout_predictions_selected_reloaded.jsonl",
    )

    final_development_metrics = None
    final_development_predictions = None
    if development_rows is not None:
        final_development_metrics, final_development_predictions = (
            evaluate_generation(
                selected_model,
                processor,
                development_rows,
                video_root=args.development_video_root,
                num_frames=args.development_num_frames,
            )
        )
        write_jsonl(
            final_development_predictions,
            output_dir / "development_predictions_selected.jsonl",
        )

    development_delta = None
    if (
        base_development_metrics is not None
        and final_development_metrics is not None
    ):
        development_delta = {
            "base_answerable_exact_accuracy": (
                base_development_metrics["answerable_exact_accuracy"]
            ),
            "selected_answerable_exact_accuracy": (
                final_development_metrics["answerable_exact_accuracy"]
            ),
            "absolute_delta": (
                float(
                    final_development_metrics[
                        "answerable_exact_accuracy"
                    ]
                )
                - float(
                    base_development_metrics[
                        "answerable_exact_accuracy"
                    ]
                )
            ),
            "percentage_point_delta": 100
            * (
                float(
                    final_development_metrics[
                        "answerable_exact_accuracy"
                    ]
                )
                - float(
                    base_development_metrics[
                        "answerable_exact_accuracy"
                    ]
                )
            ),
        }

    processor.save_pretrained(output_dir / "processor")
    summary = {
        "schema_version": "1.0",
        "experiment": {
            "model_name": args.model_name,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "best_checkpoint_type": best_checkpoint_type,
            "adapter_selected": best_checkpoint_type == "adapter",
            "optimizer_steps_completed": completed_optimizer_steps,
            "train_num_frames": args.train_num_frames,
            "holdout_num_frames": args.holdout_num_frames,
            "development_num_frames": args.development_num_frames,
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
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
            "train_examples": len(train_rows),
            "holdout_examples": len(holdout_rows),
            "development_positive_examples": (
                len(development_rows)
                if development_rows is not None
                else 0
            ),
        },
        "lora": {
            "target_module_count": len(target_modules),
            "parameter_counts": parameter_counts,
            "trainable_tensor_count": len(trainable_names),
            "first_nonzero_gradient_norm": first_nonzero_gradient_norm,
            "parameter_delta": parameter_delta,
        },
        "epoch_0_base": {
            "holdout_generation": base_holdout_metrics,
            "holdout_teacher_forced_loss": base_holdout_loss,
            "development_generation": base_development_metrics,
        },
        "selected_checkpoint": {
            "type": best_checkpoint_type,
            "epoch": best_epoch,
            "holdout_generation": final_holdout_metrics,
            "development_generation": final_development_metrics,
            "development_delta_from_epoch_0_base": development_delta,
        },
        "best_adapter_path": (
            str(best_adapter_dir)
            if best_checkpoint_type == "adapter"
            else None
        ),
        "selection_policy": {
            "primary": "balanced_task_score",
            "tie_break_1": "answerable_exact_accuracy",
            "tie_break_2": "unanswerable_recall",
            "tie_break_3": "overall_exact_accuracy",
            "tie_break_4": "lower_teacher_forced_loss",
            "epoch_0_base_is_candidate": True,
        },
    }
    summary_path = write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stage 6B.2 summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
