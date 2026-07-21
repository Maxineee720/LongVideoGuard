from __future__ import annotations

import argparse
import json
from pathlib import Path

from longvideoguard.datasets.nextqa import (
    load_nextqa_csv,
    load_video_id_map,
)
from longvideoguard.training.stage6a_data import (
    assert_disjoint_splits,
    build_video_split,
    load_manifest_rows,
    split_stats,
    split_video_ids,
    write_jsonl,
    write_qwen_json,
    write_video_list,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the larger video-disjoint Stage 6A QA train, holdout, and "
            "frozen-evaluation-candidate splits."
        )
    )
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("val_csv", type=Path)
    parser.add_argument("video_id_map", type=Path)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        required=True,
        help="Existing development-pilot JSONL whose videos stay excluded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/nextqa/stage6a"),
    )
    parser.add_argument("--train-videos", type=int, default=64)
    parser.add_argument("--holdout-videos", type=int, default=16)
    parser.add_argument("--frozen-videos", type=int, default=32)
    parser.add_argument("--max-questions-per-video", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for name, value in {
        "--train-videos": args.train_videos,
        "--holdout-videos": args.holdout_videos,
        "--frozen-videos": args.frozen_videos,
        "--max-questions-per-video": args.max_questions_per_video,
    }.items():
        if value <= 0:
            parser.error(f"{name} must be positive")

    train_records = load_nextqa_csv(args.train_csv)
    val_records = load_nextqa_csv(args.val_csv)
    video_id_map = load_video_id_map(args.video_id_map)
    development_rows = load_manifest_rows(
        [args.development_manifest]
    )
    development_video_ids = {
        str(row["video_id"])
        for row in development_rows
    }

    qa_train = build_video_split(
        train_records,
        role="qa_train",
        source_split="train",
        video_id_map=video_id_map,
        num_videos=args.train_videos,
        max_questions_per_video=args.max_questions_per_video,
        seed=args.seed,
    )
    train_video_ids = split_video_ids(qa_train)

    qa_holdout = build_video_split(
        train_records,
        role="qa_holdout",
        source_split="train",
        video_id_map=video_id_map,
        num_videos=args.holdout_videos,
        max_questions_per_video=args.max_questions_per_video,
        seed=args.seed + 1,
        excluded_video_ids=train_video_ids,
    )
    holdout_video_ids = split_video_ids(qa_holdout)

    frozen_eval = build_video_split(
        val_records,
        role="frozen_eval_candidate",
        source_split="val",
        video_id_map=video_id_map,
        num_videos=args.frozen_videos,
        max_questions_per_video=args.max_questions_per_video,
        seed=args.seed + 2,
        excluded_video_ids=development_video_ids,
    )

    named_rows = {
        "qa_train": qa_train,
        "qa_holdout": qa_holdout,
        "development_pilot": development_rows,
        "frozen_eval_candidate": frozen_eval,
    }

    # Check both logical IDs and mapped physical filenames across all sets.
    overlap_checks = assert_disjoint_splits(named_rows)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "qa_train_jsonl": write_jsonl(
            qa_train,
            output_dir / "qa_train.jsonl",
        ),
        "qa_holdout_jsonl": write_jsonl(
            qa_holdout,
            output_dir / "qa_holdout.jsonl",
        ),
        "frozen_eval_jsonl": write_jsonl(
            frozen_eval,
            output_dir / "frozen_eval_candidate.jsonl",
        ),
        "qa_train_qwen_json": write_qwen_json(
            qa_train,
            output_dir / "qa_train.qwen.json",
        ),
        "qa_holdout_qwen_json": write_qwen_json(
            qa_holdout,
            output_dir / "qa_holdout.qwen.json",
        ),
        "qa_train_video_list": write_video_list(
            qa_train,
            output_dir / "video_lists/qa_train.txt",
        ),
        "qa_holdout_video_list": write_video_list(
            qa_holdout,
            output_dir / "video_lists/qa_holdout.txt",
        ),
        "frozen_eval_video_list": write_video_list(
            frozen_eval,
            output_dir / "video_lists/frozen_eval_candidate.txt",
        ),
    }

    summary = {
        "schema_version": "1.0",
        "seed": args.seed,
        "requested": {
            "train_videos": args.train_videos,
            "holdout_videos": args.holdout_videos,
            "frozen_videos": args.frozen_videos,
            "max_questions_per_video": args.max_questions_per_video,
        },
        "development_pilot": {
            "num_videos": len(development_video_ids),
            "video_ids": sorted(development_video_ids),
        },
        "qa_train": split_stats(qa_train),
        "qa_holdout": split_stats(qa_holdout),
        "frozen_eval_candidate": split_stats(frozen_eval),
        "overlap_checks": overlap_checks,
        "outputs": {
            key: str(path)
            for key, path in paths.items()
        },
        "interpretation": {
            "qa_train": (
                "Used for the first non-overfit QA-only LoRA experiment."
            ),
            "qa_holdout": (
                "Used for checkpoint selection and early stopping; never trained."
            ),
            "development_pilot": (
                "Repeatedly inspected 48-question development set."
            ),
            "frozen_eval_candidate": (
                "New validation-video candidate set. Freeze it before final use "
                "and avoid tuning repeatedly on its results."
            ),
        },
    }

    summary_path = output_dir / "stage6a_split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stage 6A summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
