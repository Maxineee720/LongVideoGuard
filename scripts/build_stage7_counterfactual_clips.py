from __future__ import annotations

import argparse
import json
from pathlib import Path

from longvideoguard.evaluation.counterfactuals import (
    VIDEO_CONDITIONS,
    perturb_frames,
    read_video_frames,
    write_h264_video,
    write_jsonl,
)
from longvideoguard.evaluation.stage7_sampling import load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build temporal and visual counterfactual clips from the "
            "Scene-aware eight-frame Stage 7A clips."
        )
    )
    parser.add_argument("retrieval_manifest", type=Path)
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage7/counterfactuals"),
    )
    parser.add_argument("--base-method", default="scene_aware")
    parser.add_argument("--mask-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    if args.mask_count <= 0:
        parser.error("--mask-count must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    rows = [
        row
        for row in load_jsonl(args.retrieval_manifest)
        if str(row.get("stage7_sampling_method")) == args.base_method
    ]
    rows.sort(key=lambda row: str(row["sample_id"]))
    if not rows:
        raise ValueError(
            f"No retrieval rows found for {args.base_method!r}."
        )

    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    for position, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        source_path = (
            project_root / str(row["video_relpath"])
        ).resolve()
        if not source_path.is_file():
            absolute = row.get("stage7_clip_absolute_path")
            if absolute:
                source_path = Path(str(absolute)).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Base selected-frame clip not found: {source_path}"
            )

        frames = read_video_frames(source_path)
        query_scores = row.get("stage7_selected_query_scores")
        if not isinstance(query_scores, list):
            raise ValueError(
                f"Missing selected query scores for {sample_id!r}."
            )

        sample_directory = source_path.parent.name
        print(
            f"[{position}/{len(rows)}] {sample_id} "
            f"frames={len(frames)}"
        )

        for condition in VIDEO_CONDITIONS:
            if condition == "original":
                output_path = source_path
                masked_indices: list[int] = []
            else:
                perturbed, masked_indices = perturb_frames(
                    frames,
                    condition=condition,
                    sample_id=sample_id,
                    query_scores=query_scores,
                    mask_count=args.mask_count,
                    seed=args.seed,
                )
                output_path = (
                    output_dir
                    / "clips"
                    / sample_directory
                    / f"{condition}.mp4"
                )
                write_h264_video(
                    perturbed,
                    output_path,
                    fps=args.fps,
                )

            relative_path = str(
                output_path.relative_to(project_root)
            )
            manifest_rows.append(
                {
                    **row,
                    "counterfactual_condition": condition,
                    "counterfactual_base_method": args.base_method,
                    "counterfactual_mask_count": args.mask_count,
                    "counterfactual_seed": args.seed,
                    "counterfactual_masked_indices": masked_indices,
                    "video_relpath": relative_path,
                    "counterfactual_clip_absolute_path": str(
                        output_path.resolve()
                    ),
                }
            )

    manifest_path = write_jsonl(
        manifest_rows,
        output_dir / "counterfactual_manifest.jsonl",
    )
    summary = {
        "sample_count": len(rows),
        "video_condition_count": len(VIDEO_CONDITIONS),
        "manifest_row_count": len(manifest_rows),
        "base_method": args.base_method,
        "mask_count": args.mask_count,
        "seed": args.seed,
        "conditions": list(VIDEO_CONDITIONS),
        "manifest": str(manifest_path),
    }
    summary_path = output_dir / "counterfactual_build_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Counterfactual manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
