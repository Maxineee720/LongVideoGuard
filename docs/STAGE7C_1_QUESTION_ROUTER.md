# Stage 7C.1 — Dynamic question router

## Motivation

Stage 7B showed that one fixed sampling policy is not best for every question:

```text
Causal      -> Query-aware performed best
Temporal    -> Scene-aware performed best
Descriptive -> Uniform and Scene-aware tied
```

The oracle category route reaches 36/48 = 75% on the development pilot while
using an eight-frame budget. Stage 7C.1 turns that upper-bound observation into
a deployable routing experiment.

## Routing policy

```text
causal      -> query_aware
temporal    -> scene_aware
descriptive -> uniform
```

Two routers are evaluated:

1. rule-based lexical router;
2. optional Qwen3-VL text-only zero-shot router.

Neither router receives the gold answer or video. They classify only the
question text.

## Efficient evaluation

The three sampling methods have already produced predictions for the same 48
questions. The router therefore selects among these cached predictions rather
than rerunning VideoQA. This isolates routing quality and avoids unnecessary
GPU work.

## Rule-based run

```bash
python scripts/evaluate_stage7_question_router.py \
  data/processed/nextqa/pilot_val.jsonl \
  outputs/stage7/sampling_evaluation \
  --output-dir outputs/stage7/question_router
```

## Rule-based + Qwen zero-shot router

```bash
python scripts/evaluate_stage7_question_router.py \
  data/processed/nextqa/pilot_val.jsonl \
  outputs/stage7/sampling_evaluation \
  --run-qwen-router \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --output-dir outputs/stage7/question_router
```

## Outputs

```text
outputs/stage7/question_router/
├── rule_router_decisions.jsonl
├── rule_router_predictions.jsonl
├── qwen_router_decisions.jsonl
├── qwen_router_predictions.jsonl
├── oracle_router_predictions.jsonl
└── question_router_summary.json
```

## Metrics

Router classification:

- overall accuracy;
- per-category accuracy;
- confusion matrix;
- sampling-method call distribution;
- mean router confidence.

Routed VideoQA:

- overall and category-level accuracy;
- paired comparison against Uniform-8;
- exact McNemar p-value;
- gap to the gold-category oracle.

## Interpretation

The oracle result is not deployable. It is only the maximum result obtainable
from the current three cached method predictions under the chosen mapping.

A practical router is useful only when its end-to-end VideoQA result exceeds
the strongest fixed strategy or offers another benefit such as lower average
retrieval cost.
