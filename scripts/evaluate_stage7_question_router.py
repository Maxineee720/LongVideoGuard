from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from longvideoguard.evaluation.stage7_sampling import (
    load_jsonl,
    method_summary,
    paired_comparison,
)
from longvideoguard.routing.question_router import (
    build_qwen_router_prompt,
    classification_summary,
    gold_category,
    oracle_decisions,
    qwen_decision_from_output,
    route_existing_predictions,
    rule_based_decision,
    write_json,
    write_jsonl,
)


def resolve_dtype(torch: Any, name: str) -> Any:
    return "auto" if name == "auto" else getattr(torch, name)


def build_rule_decisions(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for row in rows:
        decision = rule_based_decision(str(row["question"]))
        decisions.append(
            {
                "sample_id": str(row["sample_id"]),
                "question": str(row["question"]),
                "gold_category": gold_category(row),
                **decision.to_dict(),
            }
        )
    return decisions


def build_qwen_decisions(
    rows: list[dict[str, object]],
    *,
    model_name: str,
    dtype: str,
    attn_implementation: str,
    max_new_tokens: int,
) -> list[dict[str, object]]:
    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required for Qwen routing."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen router evaluation requires a CUDA GPU.")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16.")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=resolve_dtype(torch, dtype),
        attn_implementation=attn_implementation,
    )
    model.to("cuda")
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device

    decisions: list[dict[str, object]] = []
    for position, row in enumerate(rows, start=1):
        question = str(row["question"])
        prompt = build_qwen_router_prompt(question)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
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
            },
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(device)
        input_length = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
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

        decision = qwen_decision_from_output(
            raw_output,
            fallback_question=question,
        )
        decisions.append(
            {
                "sample_id": str(row["sample_id"]),
                "question": question,
                "gold_category": gold_category(row),
                **decision.to_dict(),
            }
        )
        print(
            f"[router {position}/{len(rows)}] {row['sample_id']} "
            f"gold={gold_category(row)} "
            f"pred={decision.predicted_category} "
            f"method={decision.selected_method} "
            f"raw={raw_output!r}"
        )

    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rule-based and optional Qwen question routers by "
            "dynamically selecting existing Uniform, Scene-aware, or "
            "Query-aware Stage 7B predictions."
        )
    )
    parser.add_argument("pilot_manifest", type=Path)
    parser.add_argument("sampling_evaluation_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage7/question_router"),
    )
    parser.add_argument(
        "--run-qwen-router",
        action="store_true",
        help="Also run Qwen3-VL as a text-only zero-shot question router.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
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
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    rows = load_jsonl(args.pilot_manifest)
    rows.sort(key=lambda row: str(row["sample_id"]))

    evaluation_dir = args.sampling_evaluation_dir.expanduser().resolve()
    predictions_by_method = {
        method: load_jsonl(
            evaluation_dir / f"{method}_predictions.jsonl"
        )
        for method in ("uniform", "scene_aware", "query_aware")
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    uniform_predictions = predictions_by_method["uniform"]

    router_payloads: dict[str, dict[str, object]] = {}

    rule_decisions = build_rule_decisions(rows)
    rule_routed = route_existing_predictions(
        rule_decisions,
        predictions_by_method,
        router_name="rule_based",
    )
    write_jsonl(
        rule_decisions,
        output_dir / "rule_router_decisions.jsonl",
    )
    write_jsonl(
        rule_routed,
        output_dir / "rule_router_predictions.jsonl",
    )
    router_payloads["rule_based"] = {
        "classification": classification_summary(rule_decisions),
        "videoqa": method_summary(rule_routed),
        "comparison_vs_uniform": paired_comparison(
            uniform_predictions,
            rule_routed,
        ),
    }

    oracle_router_decisions = oracle_decisions(rows)
    oracle_routed = route_existing_predictions(
        oracle_router_decisions,
        predictions_by_method,
        router_name="oracle_category",
    )
    write_jsonl(
        oracle_routed,
        output_dir / "oracle_router_predictions.jsonl",
    )
    router_payloads["oracle_category"] = {
        "classification": classification_summary(
            oracle_router_decisions
        ),
        "videoqa": method_summary(oracle_routed),
        "comparison_vs_uniform": paired_comparison(
            uniform_predictions,
            oracle_routed,
        ),
    }

    if args.run_qwen_router:
        qwen_decisions = build_qwen_decisions(
            rows,
            model_name=args.model_name,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
        )
        qwen_routed = route_existing_predictions(
            qwen_decisions,
            predictions_by_method,
            router_name="qwen_zero_shot",
        )
        write_jsonl(
            qwen_decisions,
            output_dir / "qwen_router_decisions.jsonl",
        )
        write_jsonl(
            qwen_routed,
            output_dir / "qwen_router_predictions.jsonl",
        )
        router_payloads["qwen_zero_shot"] = {
            "classification": classification_summary(qwen_decisions),
            "videoqa": method_summary(qwen_routed),
            "comparison_vs_uniform": paired_comparison(
                uniform_predictions,
                qwen_routed,
            ),
        }

    summary = {
        "schema_version": "1.0",
        "routing_policy": {
            "causal": "query_aware",
            "temporal": "scene_aware",
            "descriptive": "uniform",
        },
        "fixed_method_baselines": {
            method: method_summary(predictions)
            for method, predictions in predictions_by_method.items()
        },
        "routers": router_payloads,
        "oracle_warning": (
            "The oracle router uses dataset-provided gold categories and is "
            "an upper bound, not a deployable result."
        ),
    }
    summary_path = write_json(
        summary,
        output_dir / "question_router_summary.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stage 7C.1 summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
