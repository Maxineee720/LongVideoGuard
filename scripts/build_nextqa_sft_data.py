from __future__ import annotations
import argparse, json
from pathlib import Path
from longvideoguard.datasets.nextqa import load_nextqa_csv, load_video_id_map
from longvideoguard.training.nextqa_sft import build_nextqa_sft_splits, compute_sft_stats, load_manifest_video_ids, write_jsonl, write_qwen_training_json

def main() -> int:
    parser=argparse.ArgumentParser(description="Build video-disjoint NExT-QA overfit and holdout SFT splits")
    parser.add_argument("train_csv", type=Path); parser.add_argument("video_id_map", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/nextqa/sft"))
    parser.add_argument("--train-videos", type=int, default=4); parser.add_argument("--holdout-videos", type=int, default=4)
    parser.add_argument("--max-questions-per-video", type=int, default=4); parser.add_argument("--seed", type=int, default=42)
    args=parser.parse_args()
    records=load_nextqa_csv(args.train_csv); mapping=load_video_id_map(args.video_id_map)
    excluded=load_manifest_video_ids(args.evaluation_manifest)
    train_rows, holdout_rows=build_nextqa_sft_splits(records, video_id_map=mapping, excluded_video_ids=excluded, train_num_videos=args.train_videos, holdout_num_videos=args.holdout_videos, max_questions_per_video=args.max_questions_per_video, seed=args.seed)
    out=args.output_dir.expanduser().resolve()
    paths={
        "train_jsonl": write_jsonl(train_rows, out/"sft_overfit_train.jsonl"),
        "holdout_jsonl": write_jsonl(holdout_rows, out/"sft_tiny_holdout.jsonl"),
        "train_qwen_json": write_qwen_training_json(train_rows, out/"sft_overfit_train.qwen.json"),
        "holdout_qwen_json": write_qwen_training_json(holdout_rows, out/"sft_tiny_holdout.qwen.json"),
    }
    train_stats, holdout_stats=compute_sft_stats(train_rows), compute_sft_stats(holdout_rows)
    tv, hv=set(train_stats["video_ids"]), set(holdout_stats["video_ids"])
    summary={"seed":args.seed,"evaluation_videos_excluded":len(excluded),"train":train_stats,"holdout":holdout_stats,"overlap_checks":{"train_holdout_video_overlap":sorted(tv&hv),"train_evaluation_video_overlap":sorted(tv&excluded),"holdout_evaluation_video_overlap":sorted(hv&excluded)},"outputs":{k:str(v) for k,v in paths.items()}}
    (out/"sft_split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
