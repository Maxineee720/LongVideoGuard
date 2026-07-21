# Stage 5.1 — NExT-QA SFT data construction

The existing 48-question validation pilot remains evaluation-only. This stage
selects small, video-disjoint splits from `train.csv`:

```text
Overfit train: 4 videos × up to 4 questions ≈ 16 examples
Tiny holdout: 4 different videos × up to 4 questions ≈ 16 examples
Frozen development pilot: 12 validation videos, 48 questions
```

## Build the splits

```bash
python scripts/build_nextqa_sft_data.py \
  data/raw/nextqa/annotations/train.csv \
  data/raw/nextqa/annotations/map_vid_vidorID.json \
  --evaluation-manifest data/processed/nextqa/pilot_val.jsonl \
  --output-dir data/processed/nextqa/sft \
  --train-videos 4 \
  --holdout-videos 4 \
  --max-questions-per-video 4 \
  --seed 42
```

Generated files include auditable LongVideoGuard JSONL files and official
Qwen-VL-style JSON files with a `video` path, a `<video>` tag, and
`conversations`.

## Extract the eight required videos

```bash
python scripts/extract_manifest_videos.py \
  data/raw/nextqa/archive/NExTVideo.zip \
  data/processed/nextqa/sft/sft_overfit_train.jsonl \
  data/processed/nextqa/sft/sft_tiny_holdout.jsonl \
  --output-dir data/raw/nextqa/sft_videos
```

Inspect `sft_split_summary.json`; all overlap lists must be empty.

## Build the Colab bundle

```bash
zip -r ~/Desktop/nextqa_sft_overfit_bundle.zip \
  data/processed/nextqa/sft \
  data/raw/nextqa/sft_videos
```

Stage 5.2 will inspect the actual processor batch and verify that only
assistant answer tokens contribute to the loss before LoRA training starts.
