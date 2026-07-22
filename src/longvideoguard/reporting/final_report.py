from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


def load_json(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSON file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return payload


def percentage(value: float) -> float:
    return 100.0 * float(value)


def write_csv(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV table.")

    fieldnames = list(rows[0].keys())
    with destination.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return destination


def frozen_accuracy_rows(
    frozen_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    conditions = frozen_summary["conditions"]
    rows = []
    for condition in (
        "uniform_8",
        "scene_aware_8",
        "uniform_16",
    ):
        metrics = conditions[condition]
        rows.append(
            {
                "method": condition,
                "correct": metrics["correct"],
                "count": metrics["count"],
                "accuracy": metrics["accuracy"],
                "accuracy_percent": percentage(metrics["accuracy"]),
                "input_tokens_mean": metrics["input_token_count"]["mean"],
                "latency_seconds_mean": metrics["latency_seconds"]["mean"],
                "peak_gpu_memory_mb_mean": metrics[
                    "peak_gpu_memory_mb"
                ]["mean"],
                "wilson_lower": metrics["wilson_95_interval"]["lower"],
                "wilson_upper": metrics["wilson_95_interval"]["upper"],
            }
        )
    return rows


def frozen_category_rows(
    frozen_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    conditions = frozen_summary["conditions"]
    rows = []
    for method in (
        "uniform_8",
        "scene_aware_8",
        "uniform_16",
    ):
        for category, payload in conditions[method][
            "by_question_category"
        ].items():
            rows.append(
                {
                    "method": method,
                    "category": category,
                    "count": payload["count"],
                    "correct": payload["correct"],
                    "accuracy": payload["accuracy"],
                    "accuracy_percent": percentage(payload["accuracy"]),
                }
            )
    return rows


def development_sampling_rows(
    sampling_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = []
    for method in (
        "uniform",
        "scene_aware",
        "query_aware",
    ):
        metrics = sampling_summary["methods"][method]
        rows.append(
            {
                "method": method,
                "correct": metrics["correct"],
                "count": metrics["count"],
                "accuracy": metrics["accuracy"],
                "accuracy_percent": percentage(metrics["accuracy"]),
                "causal_accuracy": metrics[
                    "by_question_category"
                ]["causal"]["accuracy"],
                "descriptive_accuracy": metrics[
                    "by_question_category"
                ]["descriptive"]["accuracy"],
                "temporal_accuracy": metrics[
                    "by_question_category"
                ]["temporal"]["accuracy"],
            }
        )
    return rows


def counterfactual_rows(
    counterfactual_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    preferred_order = (
        "original",
        "reversed",
        "shuffled",
        "black",
        "relevant_mask",
        "random_mask",
        "question_only",
    )
    conditions = counterfactual_summary["conditions"]
    return [
        {
            "condition": condition,
            "accuracy": conditions[condition]["accuracy"],
            "accuracy_percent": percentage(
                conditions[condition]["accuracy"]
            ),
        }
        for condition in preferred_order
    ]


def error_bucket_rows(
    error_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    return [
        {
            "bucket": name,
            "count": count,
        }
        for name, count in error_summary["bucket_counts"].items()
    ]


def build_final_metrics(
    *,
    sampling_summary: Mapping[str, object],
    router_summary: Mapping[str, object],
    counterfactual_summary: Mapping[str, object],
    frozen_summary: Mapping[str, object],
    error_summary: Mapping[str, object],
    hard_negative_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    frozen = frozen_summary["conditions"]
    uniform8 = frozen["uniform_8"]
    scene8 = frozen["scene_aware_8"]
    uniform16 = frozen["uniform_16"]

    token_reduction = 1.0 - (
        uniform8["input_token_count"]["mean"]
        / uniform16["input_token_count"]["mean"]
    )
    latency_reduction = 1.0 - (
        uniform8["latency_seconds"]["mean"]
        / uniform16["latency_seconds"]["mean"]
    )
    memory_reduction = (
        uniform16["peak_gpu_memory_mb"]["mean"]
        - uniform8["peak_gpu_memory_mb"]["mean"]
    )

    all_wrong = error_summary["bucket_counts"]["all_three_wrong"]
    same_wrong = error_summary["bucket_counts"][
        "all_predictions_same_wrong"
    ]
    disagreement = error_summary["bucket_counts"][
        "prediction_disagreement"
    ]

    qwen_router = router_summary["routers"].get("qwen_zero_shot")
    oracle_router = router_summary["routers"]["oracle_category"]

    payload = {
        "development_sampling": {
            method: sampling_summary["methods"][method]
            for method in ("uniform", "scene_aware", "query_aware")
        },
        "router": {
            "qwen_zero_shot": qwen_router,
            "oracle_category": oracle_router,
        },
        "counterfactual_diagnostics": counterfactual_summary[
            "diagnostics"
        ],
        "frozen": frozen,
        "frozen_paired_comparisons": frozen_summary[
            "paired_comparisons"
        ],
        "error_analysis": error_summary,
        "derived_takeaways": {
            "uniform8_vs_uniform16_accuracy_delta_pp": (
                100.0
                * (
                    uniform8["accuracy"]
                    - uniform16["accuracy"]
                )
            ),
            "uniform8_visual_token_reduction_percent": (
                100.0 * token_reduction
            ),
            "uniform8_latency_reduction_percent": (
                100.0 * latency_reduction
            ),
            "uniform8_peak_memory_reduction_mb": memory_reduction,
            "scene8_vs_uniform8_accuracy_delta_pp": (
                100.0
                * (
                    scene8["accuracy"]
                    - uniform8["accuracy"]
                )
            ),
            "all_three_wrong_rate_percent": (
                100.0 * all_wrong / error_summary["count"]
            ),
            "all_same_wrong_within_all_wrong_percent": (
                100.0 * same_wrong / all_wrong
            ),
            "prediction_disagreement_rate_percent": (
                100.0 * disagreement / error_summary["count"]
            ),
            "any_method_correct_oracle_upper_bound_percent": (
                100.0
                * (
                    error_summary["count"] - all_wrong
                )
                / error_summary["count"]
            ),
        },
    }
    if hard_negative_audit is not None:
        payload["hard_negative_audit"] = hard_negative_audit
    return payload


def plot_frozen_accuracy(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["method"]) for row in rows]
    values = [float(row["accuracy_percent"]) for row in rows]

    plt.figure(figsize=(7.5, 4.8))
    bars = plt.bar(labels, values)
    plt.ylabel("Frozen accuracy (%)")
    plt.xlabel("Sampling policy")
    plt.ylim(0, 100)
    plt.title("Frozen VideoQA accuracy")
    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()
    return destination


def plot_category_accuracy(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    methods = ("uniform_8", "scene_aware_8", "uniform_16")
    categories = ("causal", "descriptive", "temporal")
    lookup = {
        (str(row["method"]), str(row["category"])): float(
            row["accuracy_percent"]
        )
        for row in rows
    }

    x = np.arange(len(categories))
    width = 0.24

    plt.figure(figsize=(8.5, 5.0))
    for offset, method in enumerate(methods):
        values = [
            lookup[(method, category)]
            for category in categories
        ]
        plt.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=method,
        )

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Question category")
    plt.xticks(x, categories)
    plt.ylim(0, 100)
    plt.title("Frozen accuracy by question category")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()
    return destination


def plot_efficiency(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 5.2))
    for row in rows:
        x = float(row["input_tokens_mean"])
        y = float(row["accuracy_percent"])
        plt.scatter(x, y, s=90)
        plt.annotate(
            str(row["method"]),
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
        )

    plt.xlabel("Mean input token count")
    plt.ylabel("Frozen accuracy (%)")
    plt.ylim(60, 75)
    plt.title("Accuracy–token efficiency trade-off")
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()
    return destination


def plot_counterfactuals(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["condition"]) for row in rows]
    values = [float(row["accuracy_percent"]) for row in rows]

    plt.figure(figsize=(10.5, 5.0))
    bars = plt.bar(labels, values)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Counterfactual condition")
    plt.ylim(0, 80)
    plt.title("Visual and temporal counterfactual evaluation")
    plt.xticks(rotation=25, ha="right")
    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()
    return destination


def plot_error_buckets(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    selected = [
        row
        for row in rows
        if row["bucket"]
        in {
            "all_three_correct",
            "all_three_wrong",
            "uniform8_correct_uniform16_wrong",
            "uniform16_correct_uniform8_wrong",
            "uniform8_correct_scene_wrong",
            "scene_correct_uniform8_wrong",
            "prediction_disagreement",
        }
    ]
    labels = [str(row["bucket"]) for row in selected]
    values = [int(row["count"]) for row in selected]

    plt.figure(figsize=(10.5, 5.4))
    bars = plt.barh(labels, values)
    plt.xlabel("Number of frozen questions")
    plt.title("Final frozen error-analysis buckets")
    for bar, value in zip(bars, values, strict=True):
        plt.text(
            value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
        )
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()
    return destination


def build_readme(
    *,
    final_metrics: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    frozen = final_metrics["frozen"]
    derived = final_metrics["derived_takeaways"]
    diagnostics = final_metrics["counterfactual_diagnostics"]
    errors = final_metrics["error_analysis"]
    router = final_metrics["router"]

    hard_negative_text = (
        "A 32-example manual audit found 46.9% potentially noisy "
        "cross-video negatives, so the automatically generated "
        "answerability benchmark was retained as a data-quality case study "
        "rather than used as the final benchmark."
    )
    audit = final_metrics.get("hard_negative_audit")
    if audit is not None:
        risky = 100.0 * float(audit["risky_rate"])
        actual = 100.0 * float(audit["actually_answerable_rate"])
        hard_negative_text = (
            f"A {audit['reviewed_count']}-example manual audit found "
            f"{risky:.1f}% risky negatives and {actual:.1f}% actually "
            "answerable negatives, so the automatic answerability benchmark "
            "was retained as a data-quality case study rather than used as "
            "the final benchmark."
        )

    qwen_router = router.get("qwen_zero_shot")
    qwen_router_accuracy = (
        qwen_router["videoqa"]["accuracy"]
        if qwen_router is not None
        else None
    )
    oracle_accuracy = router["oracle_category"]["videoqa"]["accuracy"]

    lines = [
        "# LongVideoGuard",
        "",
        "**Evidence-aware and efficiency-focused VideoQA with Qwen3-VL.**",
        "",
        "LongVideoGuard is a reproducible VideoQA research project built on "
        "NExT-QA and Qwen3-VL-2B-Instruct. It studies not only accuracy, but "
        "also text shortcuts, frame-budget efficiency, dynamic sampling, "
        "counterfactual visual dependence, noisy negative construction, and "
        "frozen-set generalization.",
        "",
        "## Core contributions",
        "",
        "- Video-level train/holdout/development/frozen splits to prevent "
        "question leakage.",
        "- Verified multimodal SFT batches with assistant-only loss masking.",
        "- LoRA overfit, gradient, parameter-delta, save, and reload checks.",
        "- Swap-video diagnostics exposing text memorization shortcuts.",
        "- Uniform, scene-aware, and CLIP query-aware frame sampling under an "
        "equal eight-frame budget.",
        "- Rule-based and Qwen question routers for dynamic sampling-tool "
        "selection.",
        "- Question-only, black-video, frame-reversal, frame-shuffle, and "
        "evidence-removal counterfactuals.",
        "- A preregistered 128-question frozen evaluation with Wilson "
        "intervals and exact paired McNemar tests.",
        "",
        "## System overview",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[Video + Question] --> B{Sampling policy}",
        "    B --> C[Uniform]",
        "    B --> D[Scene-aware]",
        "    B --> E[CLIP query-aware]",
        "    C --> F[8 or 16 selected frames]",
        "    D --> F",
        "    E --> F",
        "    F --> G[Qwen3-VL-2B]",
        "    G --> H[Multiple-choice answer]",
        "    H --> I[Paired metrics + counterfactual diagnostics]",
        "```",
        "",
        "## Frozen VideoQA results",
        "",
        "| Policy | Correct | Accuracy | Mean input tokens | Mean latency | "
        "Mean peak GPU memory |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method in ("uniform_8", "scene_aware_8", "uniform_16"):
        metrics = frozen[method]
        lines.append(
            f"| {method} | {metrics['correct']}/{metrics['count']} | "
            f"{100 * metrics['accuracy']:.2f}% | "
            f"{metrics['input_token_count']['mean']:.1f} | "
            f"{metrics['latency_seconds']['mean']:.3f}s | "
            f"{metrics['peak_gpu_memory_mb']['mean']:.1f} MB |"
        )

    lines.extend(
        [
            "",
            "The preregistered Scene-aware-8 policy did not outperform "
            "Uniform-8 on the frozen set. The development-set advantage did "
            "not generalize, and none of the paired differences were "
            "statistically significant.",
            "",
            f"Uniform-8 used {derived['uniform8_visual_token_reduction_percent']:.1f}% "
            "fewer input tokens and reduced model inference latency by "
            f"{derived['uniform8_latency_reduction_percent']:.1f}% relative "
            "to Uniform-16, while losing only "
            f"{abs(derived['uniform8_vs_uniform16_accuracy_delta_pp']):.2f} "
            "percentage points of frozen accuracy.",
            "",
            "![Frozen accuracy](figures/frozen_accuracy.png)",
            "",
            "![Accuracy by category](figures/frozen_category_accuracy.png)",
            "",
            "![Accuracy-token trade-off](figures/accuracy_efficiency.png)",
            "",
            "## Counterfactual findings",
            "",
            f"- Original Scene-aware-8 accuracy exceeded question-only by "
            f"{100 * diagnostics['visual_dependence']['original_minus_question_only']:.1f} "
            "percentage points.",
            f"- Original exceeded black-video by "
            f"{100 * diagnostics['visual_dependence']['original_minus_black']:.1f} "
            "percentage points, confirming meaningful visual dependence.",
            f"- Temporal accuracy dropped by "
            f"{100 * diagnostics['temporal_order_sensitivity']['drop_after_shuffle']:.1f} "
            "points after frame shuffling, but did not drop after complete "
            "reversal.",
            f"- Masking top-CLIP frames was less harmful than random masking "
            f"({100 * diagnostics['evidence_removal']['relevant_minus_random_drop']:.1f} "
            "points), showing that CLIP similarity was not a faithful "
            "evidence-attribution score.",
            "",
            "![Counterfactual evaluation](figures/counterfactual_accuracy.png)",
            "",
            "## Dynamic routing",
            "",
        ]
    )

    if qwen_router_accuracy is not None:
        lines.append(
            f"The Qwen text-only router classified question types with "
            f"{100 * qwen_router['classification']['accuracy']:.2f}% "
            f"accuracy and achieved {100 * qwen_router_accuracy:.2f}% "
            "routed VideoQA accuracy."
        )
    lines.append(
        f"The gold-category oracle reached {100 * oracle_accuracy:.2f}%, "
        "showing limited but real complementarity between sampling policies. "
        "The deployable router did not outperform the strongest fixed policy."
    )

    lines.extend(
        [
            "",
            "## Data-quality finding",
            "",
            hard_negative_text,
            "",
            "This negative result is intentional: random cross-video "
            "replacement is not a reliable guarantee that a question becomes "
            "unanswerable.",
            "",
            "## Final error analysis",
            "",
            f"- All three sampling methods answered "
            f"{errors['bucket_counts']['all_three_correct']}/"
            f"{errors['count']} questions correctly.",
            f"- All three failed on "
            f"{errors['bucket_counts']['all_three_wrong']}/"
            f"{errors['count']} questions.",
            f"- {errors['bucket_counts']['all_predictions_same_wrong']} of "
            f"the {errors['bucket_counts']['all_three_wrong']} persistent "
            "errors produced the same wrong answer under all three policies.",
            f"- Uniform-16 rescued "
            f"{errors['bucket_counts']['uniform16_correct_uniform8_wrong']} "
            "questions missed by Uniform-8, while Uniform-8 rescued "
            f"{errors['bucket_counts']['uniform8_correct_uniform16_wrong']} "
            "questions missed by Uniform-16.",
            f"- Post-hoc majority vote reached "
            f"{100 * errors['majority_vote_diagnostic']['accuracy']:.2f}%, "
            "but is reported only as diagnostic analysis.",
            "",
            "![Frozen error buckets](figures/error_buckets.png)",
            "",
            "## Main takeaway",
            "",
            "More frames and more sophisticated retrieval were not "
            "universally better. A simple Uniform-8 policy provided the best "
            "frozen accuracy–efficiency trade-off, while counterfactual tests "
            "showed both real visual dependence and incomplete temporal-order "
            "reasoning. The project therefore emphasizes reliable evaluation "
            "and honest failure analysis rather than post-hoc metric chasing.",
            "",
            "## Reproduce the final stages",
            "",
            "```bash",
            "# Stage 7B: paired sampling evaluation",
            "python scripts/evaluate_stage7_sampling.py ...",
            "",
            "# Stage 7C.2: counterfactual evaluation",
            "python scripts/evaluate_stage7_counterfactuals.py ...",
            "",
            "# Stage 8: frozen evaluation",
            "python scripts/evaluate_stage8_frozen_videoqa.py ...",
            "",
            "# Stage 9A/9B: error analysis and reporting",
            "python scripts/analyze_stage9_final_errors.py ...",
            "python scripts/build_stage9_final_report.py ...",
            "```",
            "",
            "## Repository structure",
            "",
            "```text",
            "src/longvideoguard/       reusable data, training, retrieval, "
            "routing, evaluation, and reporting modules",
            "scripts/                  reproducible CLI entry points",
            "tests/                    unit and integration checks",
            "docs/                     stage-by-stage experiment notes",
            "outputs/                  generated metrics, predictions, and "
            "figures",
            "```",
            "",
        ]
    )

    destination.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return destination
