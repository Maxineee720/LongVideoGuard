from __future__ import annotations

import argparse
import json
from pathlib import Path

from longvideoguard.evaluation.counterfactuals import write_h264_video
from longvideoguard.evaluation.frozen_videoqa import (
    validate_video_disjointness,
    write_json,
    write_jsonl,
)
from longvideoguard.evaluation.stage7_sampling import load_jsonl
from longvideoguard.retrieval.frame_sampling import (
    decode_uniform_candidates,
    evenly_spaced_indices,
    frame_change_scores,
    safe_sample_id,
    scene_aware_indices,
)


def row_video_relpath(row: dict[str, object]) -> str:
    value = row.get("video_relpath") or row.get("video")
    if not value:
        raise ValueError(
            f"Missing video path for {row.get('sample_id')!r}."
        )
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the preregistered Stage 8 frozen VideoQA inputs: "
            "Uniform-8, Scene-aware-8, and original-video Uniform-16."
        )
    )
    parser.add_argument("frozen_manifest", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage8/frozen_sampling"),
    )
    parser.add_argument("--candidate-frames", type=int, default=32)
    parser.add_argument("--selected-frames", type=int, default=8)
    parser.add_argument("--min-gap", type=int, default=2)
    parser.add_argument("--selected-video-fps", type=float, default=1.0)
    parser.add_argument(
        "--reference-manifest",
        action="append",
        default=[],
        help=(
            "Optional NAME=PATH manifest used to verify frozen-video "
            "disjointness. May be repeated."
        ),
    )
    args = parser.parse_args()

    if args.candidate_frames <= 0 or args.selected_frames <= 0:
        parser.error("Frame counts must be positive.")
    if args.selected_frames > args.candidate_frames:
        parser.error("--selected-frames cannot exceed --candidate-frames.")

    project_root = args.project_root.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.frozen_manifest)
    rows.sort(key=lambda row: str(row["sample_id"]))

    reference_sets = {}
    for specification in args.reference_manifest:
        if "=" not in specification:
            parser.error(
                "--reference-manifest must use NAME=PATH format."
            )
        name, raw_path = specification.split("=", 1)
        reference_sets[name] = load_jsonl(Path(raw_path))

    disjointness = validate_video_disjointness(
        rows,
        reference_sets,
    )
    if reference_sets and not disjointness["all_disjoint"]:
        raise ValueError(
            "Frozen video leakage detected: "
            + json.dumps(
                disjointness["overlap_counts"],
                ensure_ascii=False,
            )
        )

    cached_clips: dict[
        str,
        dict[str, Path],
    ] = {}
    manifest_rows: list[dict[str, object]] = []

    unique_video_paths = sorted(
        {row_video_relpath(row) for row in rows}
    )
    for position, relative_video in enumerate(
        unique_video_paths,
        start=1,
    ):
        source_video = (video_root / relative_video).resolve()
        if not source_video.is_file():
            raise FileNotFoundError(
                f"Frozen source video not found: {source_video}"
            )

        print(
            f"[video {position}/{len(unique_video_paths)}] "
            f"{relative_video}"
        )
        candidates = decode_uniform_candidates(
            source_video,
            candidate_count=args.candidate_frames,
        )
        selected_count = min(
            args.selected_frames,
            len(candidates.rgb_frames),
        )
        uniform_indices = evenly_spaced_indices(
            len(candidates.rgb_frames),
            selected_count,
        )
        scene_indices = scene_aware_indices(
            frame_change_scores(candidates.rgb_frames),
            count=selected_count,
            min_gap=args.min_gap,
        )

        video_key = safe_sample_id(Path(relative_video).stem)
        video_directory = (
            output_dir / "clips" / video_key
        )
        uniform_path = write_h264_video(
            [
                candidates.rgb_frames[index]
                for index in uniform_indices
            ],
            video_directory / "uniform_8.mp4",
            fps=args.selected_video_fps,
        )
        scene_path = write_h264_video(
            [
                candidates.rgb_frames[index]
                for index in scene_indices
            ],
            video_directory / "scene_aware_8.mp4",
            fps=args.selected_video_fps,
        )
        cached_clips[relative_video] = {
            "uniform_8": uniform_path,
            "scene_aware_8": scene_path,
        }

    for row in rows:
        relative_video = row_video_relpath(row)
        for condition in (
            "uniform_8",
            "scene_aware_8",
            "uniform_16",
        ):
            if condition == "uniform_16":
                path = (video_root / relative_video).resolve()
                num_frames = 16
            else:
                path = cached_clips[relative_video][condition]
                num_frames = 8

            manifest_rows.append(
                {
                    **row,
                    "frozen_condition": condition,
                    "frozen_num_frames": num_frames,
                    "source_video_relpath": relative_video,
                    "video_relpath": str(
                        path.relative_to(project_root)
                    ),
                    "frozen_video_absolute_path": str(path),
                }
            )

    manifest_path = write_jsonl(
        manifest_rows,
        output_dir / "frozen_sampling_manifest.jsonl",
    )
    protocol = {
        "schema_version": "1.0",
        "status": "preregistered_before_frozen_results",
        "selected_final_policy": "scene_aware_8",
        "comparators": [
            "uniform_8",
            "uniform_16",
        ],
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "candidate_frames": args.candidate_frames,
        "selected_frames": args.selected_frames,
        "scene_min_gap": args.min_gap,
        "no_post_frozen_tuning": True,
        "frozen_question_count": len(rows),
        "frozen_video_count": len(unique_video_paths),
        "disjointness": disjointness,
        "manifest": str(manifest_path),
    }
    protocol_path = write_json(
        protocol,
        output_dir / "frozen_protocol.json",
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))
    print(f"Frozen protocol: {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
