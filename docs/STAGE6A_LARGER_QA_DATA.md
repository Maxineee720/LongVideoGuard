# Stage 6A — Larger video-disjoint QA SFT data

## Objective

Stage 5 deliberately overfit 16 questions and revealed a 75% text-memorisation
warning rate under swapped videos. Stage 6A constructs a larger QA-only
training experiment before adding hard negatives.

Default scale:

```text
QA train: 64 videos × up to 4 questions ≈ 256 questions
QA holdout: 16 different train videos × up to 4 questions ≈ 64 questions
Frozen candidate: 32 unused val videos × up to 4 questions ≈ 128 questions
Development pilot: existing 12 val videos, 48 questions
```

The splits are video-disjoint. Both logical video IDs and mapped physical video
filenames are audited.

## Build the splits locally

```bash
python scripts/build_nextqa_stage6a_data.py \
  data/raw/nextqa/annotations/train.csv \
  data/raw/nextqa/annotations/val.csv \
  data/raw/nextqa/annotations/map_vid_vidorID.json \
  --development-manifest data/processed/nextqa/pilot_val.jsonl \
  --output-dir data/processed/nextqa/stage6a \
  --train-videos 64 \
  --holdout-videos 16 \
  --frozen-videos 32 \
  --max-questions-per-video 4 \
  --seed 42
```

Outputs:

```text
data/processed/nextqa/stage6a/
├── qa_train.jsonl
├── qa_holdout.jsonl
├── frozen_eval_candidate.jsonl
├── qa_train.qwen.json
├── qa_holdout.qwen.json
├── stage6a_split_summary.json
└── video_lists/
    ├── qa_train.txt
    ├── qa_holdout.txt
    └── frozen_eval_candidate.txt
```

## Inspect the summary

```bash
python - <<'PY'
import json
from pathlib import Path

summary = json.loads(
    Path(
        "data/processed/nextqa/stage6a/stage6a_split_summary.json"
    ).read_text(encoding="utf-8")
)

for name in ("qa_train", "qa_holdout", "frozen_eval_candidate"):
    stats = summary[name]
    print(
        name,
        "questions=", stats["num_questions"],
        "videos=", stats["num_videos"],
        "categories=", stats["question_category_counts"],
        "answers=", stats["answer_letter_counts"],
    )

print("Overlap checks:")
for key, value in summary["overlap_checks"].items():
    print(key, value)
PY
```

Every overlap list must be empty.

## Extract only required videos

The existing `extract_manifest_videos.py` accepts multiple manifests:

```bash
python scripts/extract_manifest_videos.py \
  data/raw/nextqa/archive/NExTVideo.zip \
  data/processed/nextqa/stage6a/qa_train.jsonl \
  data/processed/nextqa/stage6a/qa_holdout.jsonl \
  data/processed/nextqa/stage6a/frozen_eval_candidate.jsonl \
  --output-dir data/raw/nextqa/stage6a_videos
```

Expected default maximum: 112 unique videos. Some physical files may already
exist elsewhere, but this directory intentionally forms a self-contained
Stage 6A bundle.

## Validate the extracted videos

```bash
longvideoguard validate-videos \
  data/processed/nextqa/stage6a/qa_train.jsonl \
  data/raw/nextqa/stage6a_videos \
  --output data/processed/nextqa/stage6a/qa_train_video_validation.json

longvideoguard validate-videos \
  data/processed/nextqa/stage6a/qa_holdout.jsonl \
  data/raw/nextqa/stage6a_videos \
  --output data/processed/nextqa/stage6a/qa_holdout_video_validation.json

longvideoguard validate-videos \
  data/processed/nextqa/stage6a/frozen_eval_candidate.jsonl \
  data/raw/nextqa/stage6a_videos \
  --output data/processed/nextqa/stage6a/frozen_video_validation.json
```

All three reports must have `all_videos_ready: true`.

## Freeze policy

The existing 48-question pilot remains a development set.

The new frozen candidate should be inspected only for integrity before the
training configuration is fixed. Once declared frozen, do not repeatedly tune
learning rate, epochs, frame count, or data mixtures against its score.

## Colab bundle

```bash
zip -r ~/Desktop/nextqa_stage6a_bundle.zip \
  data/processed/nextqa/stage6a \
  data/raw/nextqa/stage6a_videos
```

Upload to:

```text
MyDrive/LongVideoGuard/nextqa_stage6a_bundle.zip
```

Stage 6A.2 will train a QA-only LoRA using holdout-based checkpoint selection.
