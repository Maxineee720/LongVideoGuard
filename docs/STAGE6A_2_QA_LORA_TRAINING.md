# Stage 6A.2 — Larger QA-only LoRA with holdout selection

## Goal

Train Qwen3-VL on the larger video-disjoint QA split created in Stage 6A:

```text
QA train: approximately 256 questions from 64 videos
QA holdout: approximately 64 questions from 16 different videos
Development pilot: existing 48 questions from 12 validation videos
Frozen candidate: not evaluated during tuning
```

Unlike the 16-example sanity experiment, this run does not target 100% train
accuracy. The best adapter is selected using holdout generation accuracy, with
holdout assistant-only loss as the tie-breaker.

## Memory policy

The Stage 5 overfit script precomputed 16 video batches. Stage 6A.2 does not
precompute hundreds of visual tensors. It decodes and processes one sample at
a time, keeping the CPU and GPU memory footprint bounded.

## Environment

```bash
pip uninstall -y torchao
pip install -e ".[dev,vlm]"
pip install -U peft
```

The `torchao` uninstall is only necessary when the Colab image contains an
older incompatible optional installation. Standard BF16 LoRA does not require
TorchAO.

## Run

```bash
python scripts/train_qwen3vl_stage6a_qa_lora.py \
  data/processed/nextqa/stage6a/qa_train.jsonl \
  data/processed/nextqa/stage6a/qa_holdout.jsonl \
  data/raw/nextqa/stage6a_videos \
  --development-manifest data/processed/nextqa/pilot_val.jsonl \
  --development-video-root data/raw/nextqa/videos \
  --base-development-metrics \
    outputs/metrics/nextqa_qwen3vl2b_zero_shot_frames16.metrics.json \
  --epochs 3 \
  --train-num-frames 8 \
  --holdout-num-frames 8 \
  --development-num-frames 16 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --patience 2 \
  --swap-samples 32 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --output-dir outputs/adapters/stage6a_qa_lora \
  --overwrite
```

## Checkpoint policy

At the end of each epoch:

1. generate answers for the complete video-disjoint holdout;
2. calculate holdout assistant-only loss;
3. save the adapter when holdout accuracy improves;
4. when accuracy ties, prefer the lower holdout loss;
5. stop after `patience` consecutive epochs without improvement.

The new frozen-evaluation candidate is not used for checkpoint selection.

## Outputs

```text
outputs/adapters/stage6a_qa_lora/
├── best_adapter/
├── processor/
├── summary.json
├── training_history.jsonl
├── epoch_history.jsonl
├── holdout_predictions_before.jsonl
├── holdout_predictions_epoch_*.jsonl
├── holdout_predictions_best_reloaded.jsonl
├── development_predictions_best.jsonl
├── swap_correct_video_predictions.jsonl
└── swap_mismatched_video_predictions.jsonl
```

## Passing criteria

The experiment is technically valid when:

```text
non-zero LoRA gradient observed
all base parameters remain frozen
LoRA parameter delta > 0
a best adapter is selected from holdout results
the best adapter reloads successfully
all generated outputs are valid A-E letters
```

Research success is judged by comparison:

```text
holdout after vs holdout before
development pilot vs the 70.83% base result
swap-video memorisation warning rate vs the Stage 5 value of 75%
```

A larger QA-only model may improve normal QA while still retaining a
mismatched-video shortcut. Stage 6B will add hard negatives and explicit
unanswerable supervision to address that failure mode.
