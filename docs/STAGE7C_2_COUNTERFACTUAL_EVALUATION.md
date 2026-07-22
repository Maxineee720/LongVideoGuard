# Stage 7C.2 — Temporal and visual counterfactual evaluation

## Goal

Test whether Qwen3-VL uses visual content, temporal order, and
question-relevant evidence rather than relying primarily on text priors.

The experiment uses the fixed **Scene-aware-8** policy because it matched the
best development accuracy without requiring a separate CLIP retrieval pass.

## Conditions

All video conditions preserve an eight-frame input budget:

1. `original` — chronological Scene-aware frames;
2. `reversed` — the same frames in reverse order;
3. `shuffled` — the same frames in a deterministic random order;
4. `black` — all frames replaced by black frames;
5. `relevant_mask` — the two frames with the highest CLIP question similarity
   replaced by black frames;
6. `random_mask` — two deterministic random frames replaced by black frames;
7. `question_only` — no video input.

## Build clips

```bash
python scripts/build_stage7_counterfactual_clips.py \
  outputs/stage7/frame_retrieval/retrieval_manifest_fixed.jsonl \
  /content/LongVideoGuard \
  --base-method scene_aware \
  --mask-count 2 \
  --seed 42 \
  --output-dir outputs/stage7/counterfactuals
```

Expected:

```text
48 samples
6 video conditions
288 manifest rows
240 newly generated MP4 files
```

`original` reuses the existing Scene-aware clips.

## Evaluate

```bash
python scripts/evaluate_stage7_counterfactuals.py \
  outputs/stage7/counterfactuals/counterfactual_manifest.jsonl \
  /content/LongVideoGuard \
  --num-frames 8 \
  --max-new-tokens 8 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --output-dir outputs/stage7/counterfactual_evaluation \
  --overwrite
```

This performs 48 × 7 = 336 evaluations.

## Key diagnostics

### Visual dependence

```text
original accuracy - question-only accuracy
original accuracy - black-video accuracy
```

A small gap indicates strong language priors.

### Temporal order sensitivity

On the temporal subset:

```text
original accuracy - reversed accuracy
original accuracy - shuffled accuracy
```

A positive drop supports genuine order sensitivity.

### Evidence specificity

```text
drop after masking top-CLIP frames
-
drop after masking random frames
```

A positive difference suggests the selected high-relevance frames contain
question-specific evidence.

## Interpretation caution

Counterfactuals can create out-of-distribution inputs. The experiment measures
behavioral sensitivity, not a perfect causal proof of internal reasoning.
Paired prediction changes and category-level effects should be reported
alongside accuracy.
