from __future__ import annotations

import argparse
import json
from pathlib import Path

from longvideoguard.analysis.final_errors import (
    bucket_cases,
    build_case_table,
    build_summary,
    load_jsonl,
    load_prediction_directory,
    majority_vote_summary,
    write_csv,
    write_json,
    write_jsonl,
    write_markdown_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final frozen-set error analysis across Uniform-8, "
            "Scene-aware-8, and Uniform-16."
        )
    )
    parser.add_argument("frozen_manifest", type=Path)
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage9/final_error_analysis"),
    )
    parser.add_argument(
        "--max-cases-per-bucket",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    manifest_rows = load_jsonl(args.frozen_manifest)
    predictions = load_prediction_directory(args.prediction_dir)
    cases = build_case_table(
        manifest_rows,
        predictions,
    )
    buckets = bucket_cases(cases)
    summary = build_summary(cases)
    majority = majority_vote_summary(cases)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        cases,
        output_dir / "all_frozen_cases.jsonl",
    )
    write_csv(
        cases,
        output_dir / "all_frozen_cases.csv",
    )

    for name, rows in buckets.items():
        write_jsonl(
            rows,
            output_dir / f"{name}.jsonl",
        )
        write_csv(
            rows,
            output_dir / f"{name}.csv",
        )

    write_jsonl(
        majority["rows"],
        output_dir / "majority_vote_predictions.jsonl",
    )
    write_json(
        summary,
        output_dir / "final_error_summary.json",
    )
    report_path = write_markdown_report(
        summary,
        buckets,
        output_dir / "FINAL_ERROR_REPORT.md",
        max_cases_per_bucket=args.max_cases_per_bucket,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Final error report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
