# Stage 7B — Paired sampling-strategy VideoQA evaluation

## Objective

Evaluate three selected-frame strategies under a controlled eight-frame
budget:

- Uniform
- Scene-aware
- CLIP Query-aware

Every strategy uses the same:

```text
Qwen3-VL-2B-Instruct
48 development questions
prompt template
eight-frame budget
generation settings
```

Only frame selection differs.

## Run

```bash
python scripts/evaluate_stage7_sampling.py \
  outputs/stage7/frame_retrieval/retrieval_manifest.jsonl \
  /content/LongVideoGuard \
  --num-frames 8 \
  --max-new-tokens 8 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --output-dir outputs/stage7/sampling_evaluation \
  --overwrite
```

## Outputs

```text
outputs/stage7/sampling_evaluation/
├── uniform_predictions.jsonl
├── scene_aware_predictions.jsonl
├── query_aware_predictions.jsonl
└── sampling_evaluation_summary.json
```

## Metrics

Per method:

- overall and category-level accuracy;
- valid A-E output rate;
- mean, median, and p95 latency;
- input/output token counts;
- peak GPU memory.

Paired comparisons against Uniform report:

- candidate-only and Uniform-only correct counts;
- answer-change rate;
- accuracy delta;
- exact McNemar p-value.

The exact paired test is preferred to an unpaired comparison because all
methods answer the same 48 questions.

## Interpretation

A higher CLIP retrieval score does not guarantee higher VideoQA accuracy.
Query-aware retrieval may improve object/action questions while harming
temporal questions if it concentrates on a short interval. Category-level and
paired case analysis are therefore required.

Do not use the frozen set in Stage 7B. The 48-question pilot is the development
set used to select the final sampling policy.
