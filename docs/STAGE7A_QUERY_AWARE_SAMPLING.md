# Stage 7A — Query-aware frame sampling

This stage compares three ways to spend the same eight-frame budget:

1. uniform sampling;
2. scene-aware sampling from RGB histogram changes;
3. CLIP query-aware sampling with temporal/visual diversity.

It runs only on the 48-question development pilot. The frozen set remains untouched.

## Smoke test

```bash
python scripts/build_stage7_frame_retrieval_cache.py \
  data/processed/nextqa/pilot_val.jsonl \
  data/raw/nextqa/videos \
  --output-dir outputs/stage7/frame_retrieval_smoke \
  --candidate-frames 32 \
  --selected-frames 8 \
  --max-samples 2
```

## Full cache

```bash
python scripts/build_stage7_frame_retrieval_cache.py \
  data/processed/nextqa/pilot_val.jsonl \
  data/raw/nextqa/videos \
  --output-dir outputs/stage7/frame_retrieval \
  --candidate-frames 32 \
  --selected-frames 8 \
  --query-mode question \
  --relevance-weight 0.8 \
  --min-gap 2
```

Outputs contain three small chronological MP4 clips per question and a retrieval manifest. Stage 7B will reuse the existing Qwen3-VL video runner to compare VideoQA accuracy, latency, token count, memory, and category-level performance across the three strategies.
