from __future__ import annotations

import argparse
import json
from pathlib import Path

from longvideoguard.evaluation.nextqa import (
    evaluate_nextqa_predictions,
    load_prediction_jsonl,
    write_error_cases,
    write_markdown_report,
    write_metrics,
)


def default_output(
    prediction_path: Path,
    *,
    directory: str,
    suffix: str,
) -> Path:
    return Path("outputs") / directory / f"{prediction_path.stem}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LongVideoGuard NExT-QA prediction JSONL files and "
            "write metrics, failure cases, and a Markdown report."
        )
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--errors-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()

    prediction_path = args.predictions.expanduser().resolve()
    metrics_output = args.metrics_output or default_output(
        prediction_path,
        directory="metrics",
        suffix=".metrics.json",
    )
    errors_output = args.errors_output or default_output(
        prediction_path,
        directory="errors",
        suffix=".errors.jsonl",
    )
    report_output = args.report_output or default_output(
        prediction_path,
        directory="reports",
        suffix=".md",
    )

    rows = load_prediction_jsonl(prediction_path)
    metrics = evaluate_nextqa_predictions(rows)

    metrics_path = write_metrics(metrics, metrics_output)
    errors_path = write_error_cases(rows, errors_output)
    report_path = write_markdown_report(
        metrics,
        report_output,
        prediction_path=prediction_path,
    )

    overall = metrics["overall"]
    by_category = metrics["by_question_category"]

    print(json.dumps(
        {
            "samples": overall["count"],
            "correct": overall["correct"],
            "accuracy": overall["accuracy"],
            "valid_rate": overall["valid_rate"],
            "runtime_error_rate": overall["runtime_error_rate"],
            "macro_category_accuracy": metrics["macro_category_accuracy"],
            "by_question_category": by_category,
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"Metrics: {metrics_path}")
    print(f"Failure cases: {errors_path}")
    print(f"Markdown report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
