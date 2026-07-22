from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from longvideoguard.retrieval.frame_sampling import (
    CLIPFrameEncoder,
    build_query_text,
    decode_uniform_candidates,
    evenly_spaced_indices,
    frame_change_scores,
    pairwise_redundancy,
    query_aware_indices,
    safe_sample_id,
    scene_aware_indices,
    temporal_coverage,
    write_jsonl,
    write_selected_video,
)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage7/frame_retrieval"))
    parser.add_argument("--candidate-frames", type=int, default=32)
    parser.add_argument("--selected-frames", type=int, default=8)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--query-mode", choices=("question", "question_options"), default="question")
    parser.add_argument("--relevance-weight", type=float, default=0.8)
    parser.add_argument("--min-gap", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest.resolve())
    if args.max_samples:
        rows = rows[: args.max_samples]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = CLIPFrameEncoder(args.clip_model)
    cache = {}
    output_rows = []
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for position, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        relpath = str(row.get("video_relpath") or row.get("video"))
        print(f"[{position}/{len(rows)}] {sample_id}")
        if relpath not in cache:
            candidates = decode_uniform_candidates(args.video_root / relpath, args.candidate_frames)
            embeddings = encoder.encode_images(candidates.rgb_frames)
            cache[relpath] = (candidates, embeddings, frame_change_scores(candidates.rgb_frames))
        candidates, embeddings, change_scores = cache[relpath]
        count = min(args.selected_frames, len(candidates.rgb_frames))
        query_text = build_query_text(row, args.query_mode)
        text_embedding = encoder.encode_text(query_text)
        query_indices, relevance = query_aware_indices(
            embeddings, text_embedding, count, args.relevance_weight, args.min_gap
        )
        methods = {
            "uniform": evenly_spaced_indices(len(candidates.rgb_frames), count),
            "scene_aware": scene_aware_indices(change_scores, count, args.min_gap),
            "query_aware": query_indices,
        }
        sample_dir = output_dir / "samples" / safe_sample_id(sample_id)
        for method, indices in methods.items():
            frames = [candidates.rgb_frames[i] for i in indices]
            clip_path = write_selected_video(frames, sample_dir / f"{method}.mp4")
            selected_scores = [float(relevance[i]) for i in indices]
            record = {
                **row,
                "stage7_sampling_method": method,
                "stage7_query_text": query_text,
                "source_video_relpath": relpath,
                "stage7_clip_path": str(clip_path),
                "stage7_selected_candidate_positions": indices,
                "stage7_selected_source_frame_indices": [candidates.source_frame_indices[i] for i in indices],
                "stage7_selected_timestamps_seconds": [candidates.timestamps_seconds[i] for i in indices],
                "stage7_selected_query_scores": selected_scores,
                "stage7_mean_query_similarity": mean(selected_scores),
                "stage7_pairwise_redundancy": pairwise_redundancy(embeddings, indices),
                "stage7_temporal_coverage": temporal_coverage(indices, len(candidates.rgb_frames)),
            }
            output_rows.append(record)
            for key in ("stage7_mean_query_similarity", "stage7_pairwise_redundancy", "stage7_temporal_coverage"):
                metrics[method][key].append(float(record[key]))

    manifest_path = write_jsonl(output_rows, output_dir / "retrieval_manifest.jsonl")
    summary = {
        "sample_count": len(rows),
        "retrieval_row_count": len(output_rows),
        "candidate_frames": args.candidate_frames,
        "selected_frames": args.selected_frames,
        "query_mode": args.query_mode,
        "clip_model": args.clip_model,
        "methods": {
            method: {name: mean(values) for name, values in method_metrics.items()}
            for method, method_metrics in sorted(metrics.items())
        },
        "retrieval_manifest": str(manifest_path),
    }
    summary_path = output_dir / "retrieval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
