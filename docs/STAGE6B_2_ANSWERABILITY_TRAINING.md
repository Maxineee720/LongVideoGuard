# Stage 6B.2 — Structured answerability LoRA

## Goal

Train Qwen3-VL to distinguish between:

```json
{"status":"answerable","answer":"D"}
```

and:

```json
{"status":"unanswerable","answer":null}
```

The mixed holdout contains both normal QA examples and mismatched-video hard
negatives.

## Checkpoint selection

Epoch 0, the unmodified base model, is a real checkpoint candidate.

Each candidate is ranked by:

1. balanced task score:
   `(answerable exact accuracy + unanswerable recall) / 2`;
2. answerable exact accuracy;
3. unanswerable recall;
4. overall exact accuracy;
5. lower teacher-forced loss.

This prevents the script from selecting a worse adapter merely because at
least one trained epoch must be chosen. It also prevents an always-answer or
always-refuse model from winning solely through class imbalance.

## Run in Colab

```bash
python scripts/train_qwen3vl_stage6b_answerability_lora.py \
  data/processed/nextqa/stage6b/train_mixed.jsonl \
  data/processed/nextqa/stage6b/holdout_mixed.jsonl \
  data/raw/nextqa/stage6a_videos \
  --development-manifest data/processed/nextqa/pilot_val.jsonl \
  --development-video-root data/raw/nextqa/videos \
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
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --output-dir outputs/adapters/stage6b_answerability_lora \
  --overwrite
```

The development manifest can be the original Stage 4 pilot. The script accepts
the older `video` field and converts every development question into a
structured answerable-only row internally.

## Metrics

### Positive / answerable rows

- `answerable_exact_accuracy`: correct answerable status and correct A-E letter;
- `answerable_status_accuracy`: predicts answerable, regardless of letter;
- `false_refusal_rate`: predicts unanswerable on a normal QA example;
- category-level answerable accuracy.

### Negative / mismatched-video rows

- `unanswerable_recall`: correctly predicts unanswerable;
- `false_answer_rate`: predicts answerable on a mismatched video.

### Format

- `valid_structure_rate`: valid two-key JSON object;
- `exact_canonical_format_rate`: exact compact output with no extra text.

## Outputs

```text
outputs/adapters/stage6b_answerability_lora/
├── best_adapter/                       # only meaningful if an adapter wins
├── processor/
├── summary.json
├── training_history.jsonl
├── epoch_history.jsonl
├── holdout_predictions_epoch_00_base.jsonl
├── holdout_predictions_epoch_*.jsonl
├── holdout_predictions_selected_reloaded.jsonl
├── development_predictions_epoch_00_base.jsonl
└── development_predictions_selected.jsonl
```

When the epoch-0 base remains best:

```text
best_checkpoint_type: base
best_epoch: 0
adapter_selected: false
best_adapter_path: null
```

This is a valid experimental result.

## Passing criteria

Technical validity requires:

```text
valid mixed train and holdout files
non-zero LoRA gradients
only LoRA parameters trainable
adapter parameters change
all checkpoints evaluated on the same holdout
epoch-0 base included in selection
selected model reloads and evaluates successfully
```

Research success requires a selected adapter that improves the balanced task
score while keeping:

```text
answerable accuracy reasonably close to the base
unanswerable recall substantially above the base
false refusal rate controlled
false answer rate reduced
```

Do not evaluate `frozen_mixed.jsonl` until the configuration is fixed.
