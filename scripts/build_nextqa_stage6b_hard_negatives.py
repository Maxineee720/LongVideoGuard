from __future__ import annotations

import argparse
import json
from pathlib import Path

from longvideoguard.training.stage6b_data import (
    assert_cross_split_video_disjoint,
    build_negative_rows,
    build_positive_rows,
    deterministic_audit_sample,
    deterministic_mix,
    load_jsonl,
    split_stats,
    validate_negative_pairs,
    write_jsonl,
    write_qwen_json,
)


def negative_count(
    *,
    positive_count: int,
    fraction: float,
) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("Negative fraction must be in (0, 1].")
    return max(1, round(positive_count * fraction))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build structured answerable/unanswerable Stage 6B data by "
            "pairing selected NExT-QA questions with mismatched videos."
        )
    )
    parser.add_argument("stage6a_train", type=Path)
    parser.add_argument("stage6a_holdout", type=Path)
    parser.add_argument("stage6a_frozen", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/nextqa/stage6b"),
    )
    parser.add_argument(
        "--train-negative-fraction",
        type=float,
        default=0.5,
        help="Negatives per positive. 0.5 gives a 2:1 positive:negative ratio.",
    )
    parser.add_argument(
        "--holdout-negative-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--frozen-negative-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--audit-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.audit_samples <= 0:
        parser.error("--audit-samples must be positive")

    train_source = load_jsonl(args.stage6a_train)
    holdout_source = load_jsonl(args.stage6a_holdout)
    frozen_source = load_jsonl(args.stage6a_frozen)

    source_overlap_checks = assert_cross_split_video_disjoint(
        {
            "train_source": train_source,
            "holdout_source": holdout_source,
            "frozen_source": frozen_source,
        }
    )

    train_positive = build_positive_rows(
        train_source,
        split_name="train",
    )
    holdout_positive = build_positive_rows(
        holdout_source,
        split_name="holdout",
    )
    frozen_positive = build_positive_rows(
        frozen_source,
        split_name="frozen",
    )

    train_negative = build_negative_rows(
        train_source,
        split_name="train",
        negative_count=negative_count(
            positive_count=len(train_source),
            fraction=args.train_negative_fraction,
        ),
        seed=args.seed,
    )
    holdout_negative = build_negative_rows(
        holdout_source,
        split_name="holdout",
        negative_count=negative_count(
            positive_count=len(holdout_source),
            fraction=args.holdout_negative_fraction,
        ),
        seed=args.seed + 1,
    )
    frozen_negative = build_negative_rows(
        frozen_source,
        split_name="frozen",
        negative_count=negative_count(
            positive_count=len(frozen_source),
            fraction=args.frozen_negative_fraction,
        ),
        seed=args.seed + 2,
    )

    negative_validation = {
        "train": validate_negative_pairs(train_negative),
        "holdout": validate_negative_pairs(holdout_negative),
        "frozen": validate_negative_pairs(frozen_negative),
    }

    train_mixed = deterministic_mix(
        train_positive,
        train_negative,
        seed=args.seed + 10,
    )
    holdout_mixed = deterministic_mix(
        holdout_positive,
        holdout_negative,
        seed=args.seed + 11,
    )
    frozen_mixed = deterministic_mix(
        frozen_positive,
        frozen_negative,
        seed=args.seed + 12,
    )

    final_overlap_checks = assert_cross_split_video_disjoint(
        {
            "train_mixed": train_mixed,
            "holdout_mixed": holdout_mixed,
            "frozen_mixed": frozen_mixed,
        }
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train_positive": write_jsonl(
            train_positive,
            output_dir / "train_positive.jsonl",
        ),
        "train_negative": write_jsonl(
            train_negative,
            output_dir / "train_negative.jsonl",
        ),
        "train_mixed": write_jsonl(
            train_mixed,
            output_dir / "train_mixed.jsonl",
        ),
        "holdout_positive": write_jsonl(
            holdout_positive,
            output_dir / "holdout_positive.jsonl",
        ),
        "holdout_negative": write_jsonl(
            holdout_negative,
            output_dir / "holdout_negative.jsonl",
        ),
        "holdout_mixed": write_jsonl(
            holdout_mixed,
            output_dir / "holdout_mixed.jsonl",
        ),
        "frozen_positive": write_jsonl(
            frozen_positive,
            output_dir / "frozen_positive.jsonl",
        ),
        "frozen_negative": write_jsonl(
            frozen_negative,
            output_dir / "frozen_negative.jsonl",
        ),
        "frozen_mixed": write_jsonl(
            frozen_mixed,
            output_dir / "frozen_mixed.jsonl",
        ),
        "train_qwen": write_qwen_json(
            train_mixed,
            output_dir / "train_mixed.qwen.json",
        ),
        "holdout_qwen": write_qwen_json(
            holdout_mixed,
            output_dir / "holdout_mixed.qwen.json",
        ),
    }

    audit_rows = deterministic_audit_sample(
        train_negative,
        count=args.audit_samples,
        seed=args.seed + 100,
    )
    outputs["negative_audit_sample"] = write_jsonl(
        audit_rows,
        output_dir / "negative_audit_sample.jsonl",
    )

    summary = {
        "schema_version": "1.0",
        "seed": args.seed,
        "configuration": {
            "train_negative_fraction": args.train_negative_fraction,
            "holdout_negative_fraction": args.holdout_negative_fraction,
            "frozen_negative_fraction": args.frozen_negative_fraction,
            "audit_samples": len(audit_rows),
            "target_format": {
                "answerable": (
                    '{"status":"answerable","answer":"A"}'
                ),
                "unanswerable": (
                    '{"status":"unanswerable","answer":null}'
                ),
            },
        },
        "train_positive": split_stats(train_positive),
        "train_negative": split_stats(train_negative),
        "train_mixed": split_stats(train_mixed),
        "holdout_positive": split_stats(holdout_positive),
        "holdout_negative": split_stats(holdout_negative),
        "holdout_mixed": split_stats(holdout_mixed),
        "frozen_positive": split_stats(frozen_positive),
        "frozen_negative": split_stats(frozen_negative),
        "frozen_mixed": split_stats(frozen_mixed),
        "negative_validation": negative_validation,
        "source_overlap_checks": source_overlap_checks,
        "final_overlap_checks": final_overlap_checks,
        "outputs": {
            name: str(path)
            for name, path in outputs.items()
        },
        "warnings": [
            (
                "Mismatched-video negatives are candidate hard negatives. "
                "A manual audit is required because a replacement video may "
                "occasionally contain coincidental evidence."
            ),
            (
                "Do not evaluate or tune on frozen_mixed until the Stage 6B "
                "training configuration has been fixed."
            ),
        ],
    }

    summary_path = output_dir / "stage6b_data_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stage 6B data summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
