# Stage 8 — Preregistered frozen VideoQA evaluation

## Final policy selected before frozen results

```text
Scene-aware-8
```

It was selected on the 48-question development pilot because it tied the best
eight-frame accuracy, preserved the strongest temporal accuracy, required no
CLIP retrieval at runtime, and was simpler than dynamic routing.

## Frozen comparators

1. `uniform_8`
2. `scene_aware_8` — preregistered final policy
3. `uniform_16` — higher-frame reference

The final policy does not change after frozen results are observed.

## Build frozen cache

```bash
python scripts/build_stage8_frozen_sampling_cache.py \
  data/processed/nextqa/stage6a/frozen_eval_candidate.jsonl \
  data/raw/nextqa/stage6a_videos \
  /content/LongVideoGuard \
  --candidate-frames 32 \
  --selected-frames 8 \
  --min-gap 2 \
  --reference-manifest \
    pilot=data/processed/nextqa/pilot_val.jsonl \
  --reference-manifest \
    train=data/processed/nextqa/stage6a/qa_train.jsonl \
  --reference-manifest \
    holdout=data/processed/nextqa/stage6a/qa_holdout.jsonl \
  --output-dir outputs/stage8/frozen_sampling
```

Expected default scale:

```text
128 frozen questions
32 frozen videos
64 new selected-frame clips
384 evaluation-manifest rows
```

## Run once

```bash
python scripts/evaluate_stage8_frozen_videoqa.py \
  outputs/stage8/frozen_sampling/frozen_sampling_manifest.jsonl \
  /content/LongVideoGuard \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --output-dir outputs/stage8/frozen_videoqa \
  --overwrite
```

This performs:

```text
128 questions × 3 conditions = 384 Qwen3-VL evaluations
```

## Reporting

Report:

- total and category-level accuracy;
- Wilson 95% confidence interval;
- exact paired McNemar comparisons;
- latency, token count, and peak memory;
- the development-to-frozen generalization gap.

Do not tune sampling parameters, routing rules, prompts, or model parameters
after examining the frozen result.
