# Stage 9A — Final frozen-set error analysis

## Purpose

The model and sampling experiments are complete. Stage 9A organizes the
frozen predictions into interpretable error buckets without changing the
model, prompt, sampling policy, or frozen evaluation.

## Run

```bash
python scripts/analyze_stage9_final_errors.py \
  data/processed/nextqa/stage6a/frozen_eval_candidate.jsonl \
  outputs/stage8/frozen_videoqa \
  --output-dir outputs/stage9/final_error_analysis
```

## Main buckets

```text
all_three_correct
all_three_wrong
uniform8_correct_uniform16_wrong
uniform16_correct_uniform8_wrong
uniform8_correct_scene_wrong
scene_correct_uniform8_wrong
all_predictions_same_wrong
prediction_disagreement
```

## Interpretation

- `all_three_wrong`: persistent model errors not rescued by more frames or
  different sampling;
- `uniform16_correct_uniform8_wrong`: possible missing-context or
  frame-budget errors;
- `uniform8_correct_uniform16_wrong`: extra frames may introduce distraction;
- Uniform/Scene-aware disagreements: frame-selection sensitivity.

The script also reports a post-hoc majority-vote diagnostic. This is analysis
only and must not replace the preregistered Scene-aware-8 result or the later
engineering decision to default to Uniform-8.

## Outputs

```text
outputs/stage9/final_error_analysis/
├── FINAL_ERROR_REPORT.md
├── final_error_summary.json
├── all_frozen_cases.jsonl
├── all_frozen_cases.csv
├── all_three_wrong.jsonl
├── all_three_wrong.csv
├── uniform16_correct_uniform8_wrong.jsonl
├── uniform16_correct_uniform8_wrong.csv
└── ...
```

The Markdown report includes questions, options, gold answers, and all three
predictions for the most important buckets.
