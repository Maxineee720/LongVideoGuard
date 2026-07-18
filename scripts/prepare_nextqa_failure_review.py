from __future__ import annotations

import argparse
import json
from pathlib import Path

from longvideoguard.failure_analysis import (
    build_review_rows,
    load_jsonl,
    make_contact_sheet,
    read_uniform_video_frames,
    write_jsonl,
    write_review_html,
    write_rgb_image,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate uniform-frame contact sheets and an editable review "
            "template for NExT-QA failure cases."
        )
    )
    parser.add_argument("errors", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/nextqa_frames16"),
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    error_rows = load_jsonl(args.errors)
    output_dir = args.output_dir.expanduser().resolve()
    frame_dir = output_dir / "contact_sheets"
    frame_dir.mkdir(parents=True, exist_ok=True)

    unique_video_relpaths: dict[str, str] = {}
    for row in error_rows:
        unique_video_relpaths.setdefault(
            str(row["video_id"]),
            str(row["video_relpath"]),
        )

    contact_sheet_by_video: dict[str, str] = {}
    for position, (video_id, video_relpath) in enumerate(
        sorted(unique_video_relpaths.items()),
        start=1,
    ):
        video_path = args.video_root / video_relpath
        print(
            f"[{position}/{len(unique_video_relpaths)}] "
            f"Sampling {video_id}: {video_path}"
        )
        frames, indices, timestamps = read_uniform_video_frames(
            video_path,
            num_frames=args.num_frames,
        )
        sheet = make_contact_sheet(
            frames,
            frame_indices=indices,
            timestamps=timestamps,
            columns=args.columns,
        )
        image_path = write_rgb_image(
            sheet,
            frame_dir / f"{video_id}.jpg",
        )
        contact_sheet_by_video[video_id] = str(
            image_path.relative_to(output_dir)
        )

    review_rows = build_review_rows(
        error_rows,
        contact_sheet_by_video=contact_sheet_by_video,
    )
    review_path = write_jsonl(
        review_rows,
        output_dir / "failure_review.jsonl",
    )
    html_path = write_review_html(
        review_rows,
        output_dir / "failure_review.html",
    )

    summary = {
        "error_samples": len(error_rows),
        "unique_error_videos": len(unique_video_relpaths),
        "num_frames_per_video": args.num_frames,
        "review_jsonl": str(review_path),
        "review_html": str(html_path),
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
