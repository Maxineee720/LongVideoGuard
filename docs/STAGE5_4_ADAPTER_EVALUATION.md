# Stage 5.4 — Frozen-pilot and swap-video evaluation

## Purpose

Stage 5.3 proved that the LoRA training pipeline can memorise the 16-example
training split and that the adapter saves and reloads correctly.

Stage 5.4 asks two separate questions:

1. What happens to the existing 48-question development pilot after this
   deliberately overfit adapter is applied?
2. Does the adapter's 100% train accuracy depend on the correct videos, or can
   it reproduce answers after the videos are deliberately replaced?

## Required data

The Colab runtime needs both data bundles:

```text
Pilot:
data/processed/nextqa/pilot_val.jsonl
data/raw/nextqa/videos/*.mp4

SFT:
data/processed/nextqa/sft/sft_overfit_train.jsonl
data/raw/nextqa/sft_videos/*.mp4
```

It also needs the saved Stage 5.3 adapter:

```text
outputs/adapters/stage5_3_lora_overfit/adapter/
```

If the runtime was reset, restore the adapter from Google Drive.

## Run

```bash
python scripts/evaluate_qwen3vl_adapter_stage5_4.py \
  outputs/adapters/stage5_3_lora_overfit/adapter \
  data/processed/nextqa/pilot_val.jsonl \
  data/raw/nextqa/videos \
  data/processed/nextqa/sft/sft_overfit_train.jsonl \
  data/raw/nextqa/sft_videos \
  --processor-dir outputs/adapters/stage5_3_lora_overfit/processor \
  --base-pilot-metrics \
    outputs/metrics/nextqa_qwen3vl2b_zero_shot_frames16.metrics.json \
  --pilot-num-frames 16 \
  --swap-num-frames 8 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --output-dir outputs/evaluations/stage5_4 \
  --overwrite
```

The pilot evaluation uses 16 frames to match the Stage 4 base result. The
swap-video diagnostic uses 8 frames to match the Stage 5.3 overfit training
configuration. Keeping these budgets separate avoids conflating adapter
effects with a frame-count change.

## Outputs

```text
outputs/evaluations/stage5_4/
├── summary.json
├── pilot_adapter_predictions.jsonl
├── train_correct_video_predictions.jsonl
├── train_swapped_video_predictions.jsonl
└── swap_video_comparisons.jsonl
```

## Swap-video metrics

- `prediction_change_rate`: fraction of questions whose answer changes when
  the video is replaced.
- `swapped_video_accuracy`: accuracy despite receiving a mismatched video.
- `accuracy_drop_after_swap`: correct-video accuracy minus swapped-video
  accuracy.
- `text_memorisation_warning_rate`: fraction that remain correct with exactly
  the same prediction after video replacement.

A low change rate or high mismatched-video accuracy is evidence consistent
with text memorisation. It is not a formal proof that the model ignores all
visual information.

## Interpretation constraints

- The 48-question pilot is a repeatedly inspected development set.
- The 16-example adapter was deliberately trained to memorise.
- A swap-video test is a perturbation diagnostic, not a causal attribution
  theorem.
- A controlled base/adapter comparison must use the same model, prompt,
  processor, frame count, decoding, and evaluation rows.
