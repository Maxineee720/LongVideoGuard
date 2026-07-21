# Stage 6B.1 — Hard-negative answerability data

## Motivation

Stage 6A QA-only LoRA restored development-pilot accuracy to the base result
but did not improve unseen-video QA. In the swap-video diagnostic, 68.75% of
the sampled training questions stayed correct with the same answer after the
video was replaced.

Stage 6B introduces explicit answerability supervision.

## Targets

Positive example:

```json
{"status":"answerable","answer":"D"}
```

Mismatched-video example:

```json
{"status":"unanswerable","answer":null}
```

The target is stored as an exact compact JSON string in `assistant_target`.

## Default mixture

```text
Train:
  256 positive QA examples
  128 mismatched-video negatives
  384 total

Holdout:
  64 positive QA examples
  32 mismatched-video negatives
  96 total

Frozen candidate:
  128 positive QA examples
  64 mismatched-video negatives
  192 total
```

No new video extraction is required. Every negative uses a replacement video
from the same Stage 6A split, while train, holdout, and frozen video pools stay
disjoint.

## Build

```bash
python scripts/build_nextqa_stage6b_hard_negatives.py \
  data/processed/nextqa/stage6a/qa_train.jsonl \
  data/processed/nextqa/stage6a/qa_holdout.jsonl \
  data/processed/nextqa/stage6a/frozen_eval_candidate.jsonl \
  --output-dir data/processed/nextqa/stage6b \
  --train-negative-fraction 0.5 \
  --holdout-negative-fraction 0.5 \
  --frozen-negative-fraction 0.5 \
  --audit-samples 32 \
  --seed 42
```

## Outputs

```text
data/processed/nextqa/stage6b/
├── train_positive.jsonl
├── train_negative.jsonl
├── train_mixed.jsonl
├── train_mixed.qwen.json
├── holdout_positive.jsonl
├── holdout_negative.jsonl
├── holdout_mixed.jsonl
├── holdout_mixed.qwen.json
├── frozen_positive.jsonl
├── frozen_negative.jsonl
├── frozen_mixed.jsonl
├── negative_audit_sample.jsonl
└── stage6b_data_summary.json
```

## Pairing policy

The builder:

1. balances negative source questions over question category and answer letter;
2. replaces each selected question's video with a different video;
3. prefers a replacement video represented in the same question category;
4. falls back to any different video only when required;
5. records the original and replacement video IDs;
6. marks every candidate negative as `negative_audit_status: pending`.

## Mandatory manual audit

A mismatched video is not automatically guaranteed to be unanswerable. A
replacement video may coincidentally contain compatible evidence.

Inspect `negative_audit_sample.jsonl` before training. For each sampled row,
review:

```text
source question
source question video ID
replacement video ID
replacement video
options
```

Set aside or regenerate negatives that appear genuinely answerable from the
replacement video.

The first iteration may proceed only after the audit shows the candidate
strategy has an acceptably low accidental-answerability rate.

## Leakage checks

`stage6b_data_summary.json` must show empty lists for all source and final
cross-split video-overlap checks.

All negative-validation fields must report:

```text
all_cross_video: true
all_targets_unanswerable: true
```

## Freeze policy

Do not run the model on `frozen_mixed.jsonl` while choosing learning rate,
epoch count, LoRA rank, negative ratio, or decision threshold.

Stage 6B.2 will train with the mixed train set and select checkpoints using
separate answerable-accuracy and unanswerable-recall metrics on the mixed
holdout.
