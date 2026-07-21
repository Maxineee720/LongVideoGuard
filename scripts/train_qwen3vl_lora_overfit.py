from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from longvideoguard.training.lora_overfit import (
    assert_only_lora_trainable,
    cleanup_cuda,
    count_parameters,
    evaluate_generation,
    find_language_lora_targets,
    finite_mean,
    load_sft_jsonl,
    mean_teacher_forced_loss,
    move_batch_to_device,
    optimizer_step_sample_indices,
    parameter_delta_summary,
    precompute_generation_batches,
    precompute_teacher_forcing_batches,
    set_global_seed,
    snapshot_trainable_parameters,
    write_jsonl,
)


def resolve_dtype(torch: Any, name: str) -> Any:
    return "auto" if name == "auto" else getattr(torch, name)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inject language-tower LoRA into Qwen3-VL and deliberately "
            "overfit the tiny NExT-QA training split as a sanity check."
        )
    )
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("holdout_manifest", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/adapters/stage5_3_lora_overfit"),
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--min-steps-before-early-stop", type=int, default=40)
    parser.add_argument("--perfect-checks-to-stop", type=int, default=2)
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

    positive_integer_fields = {
        "--num-frames": args.num_frames,
        "--max-steps": args.max_steps,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--lora-rank": args.lora_rank,
        "--lora-alpha": args.lora_alpha,
        "--eval-every": args.eval_every,
        "--perfect-checks-to-stop": args.perfect_checks_to_stop,
    }
    for field, value in positive_integer_fields.items():
        if value <= 0:
            parser.error(f"{field} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.lora_dropout < 1:
        parser.error("--lora-dropout must be in [0, 1)")

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with:\n"
            'pip install -e ".[dev,vlm]"\n'
            "pip install -U peft"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5.3 requires a CUDA GPU.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16.")

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
    history_path = output_dir / "training_history.jsonl"

    set_global_seed(args.seed)
    train_rows = load_sft_jsonl(args.train_manifest)
    holdout_rows = load_sft_jsonl(args.holdout_manifest)

    print(f"Train samples: {len(train_rows)}")
    print(f"Holdout samples: {len(holdout_rows)}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    processor = AutoProcessor.from_pretrained(args.model_name)
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = None

    print("Precomputing training batches...")
    train_teacher_batches, train_batch_reports = (
        precompute_teacher_forcing_batches(
            processor,
            train_rows,
            video_root=args.video_root,
            num_frames=args.num_frames,
        )
    )
    print("Precomputing train generation batches...")
    train_generation_batches = precompute_generation_batches(
        processor,
        train_rows,
        video_root=args.video_root,
        num_frames=args.num_frames,
    )
    print("Precomputing holdout generation batches...")
    holdout_generation_batches = precompute_generation_batches(
        processor,
        holdout_rows,
        video_root=args.video_root,
        num_frames=args.num_frames,
    )

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
    print(f"Exact language LoRA target modules: {len(target_modules)}")
    print("\n".join(target_modules[:8]))
    if len(target_modules) > 8:
        print("...")

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
    print(json.dumps(parameter_counts, indent=2))
    print(f"Trainable LoRA tensors: {len(trainable_names)}")

    initial_snapshot = snapshot_trainable_parameters(model)

    print("Evaluating base/initial adapter...")
    initial_train_loss = mean_teacher_forced_loss(
        model,
        train_teacher_batches,
    )
    initial_train_metrics, initial_train_predictions = evaluate_generation(
        model,
        processor,
        train_rows,
        train_generation_batches,
    )
    initial_holdout_metrics, initial_holdout_predictions = evaluate_generation(
        model,
        processor,
        holdout_rows,
        holdout_generation_batches,
    )
    print(
        f"Initial train loss={initial_train_loss:.6f}, "
        f"train accuracy={initial_train_metrics['accuracy']:.4f}, "
        f"holdout accuracy={initial_holdout_metrics['accuracy']:.4f}"
    )

    write_jsonl(
        initial_train_predictions,
        output_dir / "predictions_train_before.jsonl",
    )
    write_jsonl(
        initial_holdout_predictions,
        output_dir / "predictions_holdout_before.jsonl",
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    schedule = optimizer_step_sample_indices(
        num_samples=len(train_teacher_batches),
        optimizer_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )

    optimizer.zero_grad(set_to_none=True)
    first_nonzero_gradient_norm: float | None = None
    perfect_checks = 0
    completed_steps = 0
    last_train_metrics = initial_train_metrics

    for optimizer_step, sample_indices in enumerate(schedule, start=1):
        model.train()
        micro_losses: list[float] = []

        for sample_index in sample_indices:
            batch = move_batch_to_device(
                train_teacher_batches[sample_index],
                next(model.parameters()).device,
            )
            outputs = model(
                **batch,
                use_cache=False,
            )
            raw_loss = outputs.loss
            if not torch.isfinite(raw_loss):
                raise ValueError(
                    f"Non-finite loss at optimizer step {optimizer_step}: "
                    f"{raw_loss}"
                )

            (raw_loss / args.gradient_accumulation_steps).backward()
            micro_losses.append(
                float(raw_loss.detach().float().cpu().item())
            )

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
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            max_norm=args.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_steps = optimizer_step

        history_row: dict[str, object] = {
            "optimizer_step": optimizer_step,
            "sample_indices": sample_indices,
            "mean_micro_loss": finite_mean(micro_losses),
            "gradient_norm_before_clip": gradient_norm,
            "gradient_norm_returned_by_clip": float(clipped_norm),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        should_evaluate = (
            optimizer_step == 1
            or optimizer_step % args.eval_every == 0
            or optimizer_step == args.max_steps
        )
        if should_evaluate:
            teacher_loss = mean_teacher_forced_loss(
                model,
                train_teacher_batches,
            )
            train_metrics, train_predictions = evaluate_generation(
                model,
                processor,
                train_rows,
                train_generation_batches,
            )
            last_train_metrics = train_metrics
            history_row["evaluation"] = {
                "teacher_forced_loss": teacher_loss,
                "train_generation": train_metrics,
            }
            print(
                f"step={optimizer_step:03d} "
                f"micro_loss={history_row['mean_micro_loss']:.6f} "
                f"teacher_loss={teacher_loss:.6f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"grad_norm={gradient_norm:.6f}"
            )

            if (
                optimizer_step >= args.min_steps_before_early_stop
                and train_metrics["accuracy"] == 1.0
            ):
                perfect_checks += 1
            else:
                perfect_checks = 0

            write_jsonl(
                train_predictions,
                output_dir
                / f"predictions_train_step_{optimizer_step:04d}.jsonl",
            )

        append_jsonl(history_path, history_row)

        if perfect_checks >= args.perfect_checks_to_stop:
            print(
                "Early stop: perfect train generation accuracy was observed "
                f"{perfect_checks} consecutive evaluation times."
            )
            break

    if first_nonzero_gradient_norm is None:
        raise ValueError("No non-zero LoRA gradient was observed.")

    final_delta = parameter_delta_summary(model, initial_snapshot)
    if float(final_delta["delta_l2"]) <= 0:
        raise ValueError("LoRA parameters did not change.")

    print("Final evaluation before saving...")
    final_train_loss = mean_teacher_forced_loss(
        model,
        train_teacher_batches,
    )
    final_train_metrics, final_train_predictions = evaluate_generation(
        model,
        processor,
        train_rows,
        train_generation_batches,
    )
    final_holdout_metrics, final_holdout_predictions = evaluate_generation(
        model,
        processor,
        holdout_rows,
        holdout_generation_batches,
    )

    write_jsonl(
        final_train_predictions,
        output_dir / "predictions_train_after.jsonl",
    )
    write_jsonl(
        final_holdout_predictions,
        output_dir / "predictions_holdout_after.jsonl",
    )

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(output_dir / "processor")

    summary: dict[str, object] = {
        "schema_version": "1.0",
        "experiment": {
            "model_name": args.model_name,
            "num_frames": args.num_frames,
            "max_steps_requested": args.max_steps,
            "optimizer_steps_completed": completed_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "seed": args.seed,
        },
        "data": {
            "train_samples": len(train_rows),
            "holdout_samples": len(holdout_rows),
        },
        "lora": {
            "target_module_count": len(target_modules),
            "target_modules": target_modules,
            "parameter_counts": parameter_counts,
            "trainable_tensor_count": len(trainable_names),
            "first_nonzero_gradient_norm": first_nonzero_gradient_norm,
            "parameter_delta": final_delta,
        },
        "before": {
            "train_teacher_forced_loss": initial_train_loss,
            "train_generation": initial_train_metrics,
            "holdout_generation": initial_holdout_metrics,
        },
        "after_before_reload": {
            "train_teacher_forced_loss": final_train_loss,
            "train_generation": final_train_metrics,
            "holdout_generation": final_holdout_metrics,
        },
        "adapter_path": str(adapter_dir),
        "batch_reports": train_batch_reports,
    }

    (output_dir / "summary_before_reload.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Releasing trained model and reloading Base + Adapter...")
    del optimizer
    del model
    del base_model
    cleanup_cuda()

    reloaded_base = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    reloaded_base.to("cuda")
    reloaded_model = PeftModel.from_pretrained(
        reloaded_base,
        adapter_dir,
        is_trainable=False,
    )
    reloaded_model.eval()

    reload_train_metrics, reload_train_predictions = evaluate_generation(
        reloaded_model,
        processor,
        train_rows,
        train_generation_batches,
    )
    write_jsonl(
        reload_train_predictions,
        output_dir / "predictions_train_reloaded.jsonl",
    )

    before_letters = [
        row["predicted_letter"]
        for row in final_train_predictions
    ]
    reloaded_letters = [
        row["predicted_letter"]
        for row in reload_train_predictions
    ]
    exact_prediction_match = before_letters == reloaded_letters
    match_count = sum(
        before == after
        for before, after in zip(
            before_letters,
            reloaded_letters,
            strict=True,
        )
    )

    summary["reload_check"] = {
        "train_generation": reload_train_metrics,
        "exact_prediction_list_match": exact_prediction_match,
        "matching_predictions": match_count,
        "total_predictions": len(before_letters),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "steps_completed": completed_steps,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "train_before": initial_train_metrics,
            "train_after": final_train_metrics,
            "holdout_before": initial_holdout_metrics,
            "holdout_after": final_holdout_metrics,
            "first_nonzero_gradient_norm": first_nonzero_gradient_norm,
            "parameter_delta": final_delta,
            "reload_check": summary["reload_check"],
            "output_dir": str(output_dir),
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
