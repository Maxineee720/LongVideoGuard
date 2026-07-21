# Stage 5.3 — Qwen3-VL LoRA overfit sanity check

## Purpose

This is a deliberate memorisation test on approximately 16 training examples.
It verifies that:

- LoRA is injected only into language-tower attention projections;
- the visual tower and original model weights remain frozen;
- LoRA gradients are non-zero;
- adapter parameters actually change;
- assistant-only loss falls;
- training-set generation accuracy approaches 100%;
- the adapter saves and reloads without changing deterministic predictions.

This experiment does **not** establish generalisation.

## Dependencies

Install the project and PEFT:

```bash
pip install -e ".[dev,vlm]"
pip install -U peft
```

PEFT inserts small trainable low-rank matrices while the base parameters stay
frozen. The script discovers exact language-tower `q_proj`, `k_proj`,
`v_proj`, and `o_proj` modules and excludes names belonging to visual/vision
components.

## Run on an A100

```bash
python scripts/train_qwen3vl_lora_overfit.py \
  data/processed/nextqa/sft/sft_overfit_train.jsonl \
  data/processed/nextqa/sft/sft_tiny_holdout.jsonl \
  data/raw/nextqa/sft_videos \
  --num-frames 8 \
  --max-steps 120 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --eval-every 20 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --output-dir outputs/adapters/stage5_3_lora_overfit \
  --overwrite
```

The 16 teacher-forcing batches and prompt-only generation batches are
precomputed once on CPU. This avoids repeatedly decoding the same videos
during the small sanity experiment.

## Expected outputs

```text
outputs/adapters/stage5_3_lora_overfit/
├── adapter/
├── processor/
├── training_history.jsonl
├── summary_before_reload.json
├── summary.json
├── predictions_train_before.jsonl
├── predictions_train_after.jsonl
├── predictions_train_reloaded.jsonl
├── predictions_holdout_before.jsonl
├── predictions_holdout_after.jsonl
└── predictions_train_step_*.jsonl
```

## Passing criteria

The pipeline passes when:

```text
non-zero LoRA gradient observed
LoRA parameter delta > 0
loss is finite and decreases
train generation accuracy >= 15/16 (ideal: 16/16)
adapter reload predictions match the saved model
```

Tiny-holdout accuracy is diagnostic only. It may stay unchanged or decrease
because the experiment intentionally overfits a very small set.

## If 120 steps do not overfit

Resume by rerunning into a new output directory with one of these controlled
changes:

```text
max steps: 200
learning rate: 3e-4
LoRA rank: 16
```

Change one variable at a time. Do not present train-set memorisation as a
benchmark improvement.
