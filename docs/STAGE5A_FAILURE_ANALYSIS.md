# Stage 5A — Failure analysis before fine-tuning

## Why this stage exists

Do not fine-tune immediately after observing 14 wrong answers. A wrong answer
can be caused by different mechanisms:

- the 16 uniformly sampled frames missed the relevant event;
- the evidence is visible but the model confused temporal order;
- the model recognised the action but made a causal reasoning error;
- the model confused an object or action;
- the question or annotation is ambiguous;
- language priors overrode visual evidence.

Each cause implies a different intervention. More LoRA training does not solve
a sampling miss.

## Generate review materials

```bash
python scripts/prepare_nextqa_failure_review.py \
  outputs/errors/nextqa_qwen3vl2b_zero_shot_frames16.errors.jsonl \
  data/raw/nextqa/videos \
  --num-frames 16 \
  --output-dir outputs/analysis/nextqa_frames16
```

Outputs:

```text
outputs/analysis/nextqa_frames16/
├── contact_sheets/
│   └── <video_id>.jpg
├── failure_review.jsonl
├── failure_review.html
└── generation_summary.json
```

The HTML report is for visual inspection. The JSONL file is the editable
source of truth.

## Manual fields

For each error, fill:

- `evidence_covered`: `true` or `false`;
- `failure_type`: one of the allowed taxonomy values;
- `attribution_confidence`: `low`, `medium`, or `high`;
- `notes`;
- `proposed_fix`.

Allowed failure types:

```text
sampling_miss
temporal_order_error
causal_reasoning_error
object_confusion
action_confusion
language_prior_guess
ambiguous_question
annotation_issue
other
```

## Validate completed labels

```bash
python scripts/summarize_failure_review.py \
  outputs/analysis/nextqa_frames16/failure_review.jsonl
```

The command exits successfully only when every row has a valid manual label.

## Interpretation

- Many `sampling_miss` errors motivate more frames or key-frame selection.
- Many `temporal_order_error` errors motivate temporal instruction data.
- Many `causal_reasoning_error` errors motivate causal VideoQA SFT.
- Many `language_prior_guess` errors motivate counterfactual and unanswerable
  examples.
- `annotation_issue` and `ambiguous_question` should not be blindly used for
  training.
