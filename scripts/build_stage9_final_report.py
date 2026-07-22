from __future__ import annotations

import argparse
import json
from pathlib import Path

from longvideoguard.reporting.final_report import (
    build_final_metrics,
    build_readme,
    counterfactual_rows,
    development_sampling_rows,
    error_bucket_rows,
    frozen_accuracy_rows,
    frozen_category_rows,
    load_json,
    plot_category_accuracy,
    plot_counterfactuals,
    plot_efficiency,
    plot_error_buckets,
    plot_frozen_accuracy,
    write_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build final LongVideoGuard tables, figures, metrics, and a "
            "generated README from completed experiment outputs."
        )
    )
    parser.add_argument(
        "--sampling-summary",
        type=Path,
        default=Path(
            "outputs/stage7/sampling_evaluation/"
            "sampling_evaluation_summary.json"
        ),
    )
    parser.add_argument(
        "--router-summary",
        type=Path,
        default=Path(
            "outputs/stage7/question_router_qwen/"
            "question_router_summary.json"
        ),
    )
    parser.add_argument(
        "--counterfactual-summary",
        type=Path,
        default=Path(
            "outputs/stage7/counterfactual_evaluation/"
            "counterfactual_evaluation_summary.json"
        ),
    )
    parser.add_argument(
        "--frozen-summary",
        type=Path,
        default=Path(
            "outputs/stage8/frozen_videoqa/"
            "frozen_videoqa_summary.json"
        ),
    )
    parser.add_argument(
        "--error-summary",
        type=Path,
        default=Path(
            "outputs/stage9/final_error_analysis/"
            "final_error_summary.json"
        ),
    )
    parser.add_argument(
        "--hard-negative-audit",
        type=Path,
        help="Optional Stage 6B audit_summary.json path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage9/project_report"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sampling = load_json(args.sampling_summary)
    router = load_json(args.router_summary)
    counterfactual = load_json(args.counterfactual_summary)
    frozen = load_json(args.frozen_summary)
    errors = load_json(args.error_summary)
    hard_negative_audit = (
        load_json(args.hard_negative_audit)
        if args.hard_negative_audit is not None
        else None
    )

    frozen_rows = frozen_accuracy_rows(frozen)
    category_rows = frozen_category_rows(frozen)
    development_rows = development_sampling_rows(sampling)
    counterfactual_table = counterfactual_rows(counterfactual)
    error_rows = error_bucket_rows(errors)

    write_csv(
        frozen_rows,
        tables_dir / "frozen_accuracy.csv",
    )
    write_csv(
        category_rows,
        tables_dir / "frozen_category_accuracy.csv",
    )
    write_csv(
        development_rows,
        tables_dir / "development_sampling.csv",
    )
    write_csv(
        counterfactual_table,
        tables_dir / "counterfactual_accuracy.csv",
    )
    write_csv(
        error_rows,
        tables_dir / "error_buckets.csv",
    )

    plot_frozen_accuracy(
        frozen_rows,
        figures_dir / "frozen_accuracy.png",
    )
    plot_category_accuracy(
        category_rows,
        figures_dir / "frozen_category_accuracy.png",
    )
    plot_efficiency(
        frozen_rows,
        figures_dir / "accuracy_efficiency.png",
    )
    plot_counterfactuals(
        counterfactual_table,
        figures_dir / "counterfactual_accuracy.png",
    )
    plot_error_buckets(
        error_rows,
        figures_dir / "error_buckets.png",
    )

    final_metrics = build_final_metrics(
        sampling_summary=sampling,
        router_summary=router,
        counterfactual_summary=counterfactual,
        frozen_summary=frozen,
        error_summary=errors,
        hard_negative_audit=hard_negative_audit,
    )
    (output_dir / "final_metrics.json").write_text(
        json.dumps(
            final_metrics,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    readme_path = build_readme(
        final_metrics=final_metrics,
        output_path=output_dir / "README_GENERATED.md",
    )

    print(
        json.dumps(
            final_metrics["derived_takeaways"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Generated README: {readme_path}")
    print(f"Figures: {figures_dir}")
    print(f"Tables: {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
