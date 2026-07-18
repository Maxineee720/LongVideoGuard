# Stage 4 — Zero-shot evaluation and failure extraction

## Purpose

Stage 3 proved that the video-VLM inference pipeline runs end to end.
Stage 4 answers a different question:

> How well did the model perform, on which question types, at what cost, and
> which examples require manual failure analysis?

## Run in Colab

```bash
python scripts/evaluate_nextqa_predictions.py   outputs/predictions/nextqa_qwen3vl2b_zero_shot_frames16.jsonl
```

The default outputs are:

```text
outputs/
├── metrics/
│   └── nextqa_qwen3vl2b_zero_shot_frames16.metrics.json
├── errors/
│   └── nextqa_qwen3vl2b_zero_shot_frames16.errors.jsonl
└── reports/
    └── nextqa_qwen3vl2b_zero_shot_frames16.md
```

Copy them to persistent storage:

```bash
mkdir -p /content/drive/MyDrive/LongVideoGuard/results/stage4

cp outputs/metrics/*.json   /content/drive/MyDrive/LongVideoGuard/results/stage4/

cp outputs/errors/*.jsonl   /content/drive/MyDrive/LongVideoGuard/results/stage4/

cp outputs/reports/*.md   /content/drive/MyDrive/LongVideoGuard/results/stage4/
```

## Metrics

### Accuracy

Every one of the five options has exactly one gold label. Overall accuracy is:

```text
number of correct predictions / number of evaluated questions
```

Invalid generations and runtime errors count as incorrect in the primary
accuracy denominator.

### Valid output rate

A valid prediction is a successfully parsed label from A to E. This metric
separates task performance from output-format reliability.

### Category and subtype breakdowns

The report separates causal, temporal, and descriptive categories, and also
retains the finer official NExT-QA question subtype.

### Baselines

The evaluator reports:

- random-choice accuracy: 20%;
- majority gold-label baseline for the current subset.

These are sanity baselines, not strong video models.

### Wilson interval

The 48-question pilot is small. The report therefore includes a 95% Wilson
interval for accuracy. This does not fix selection bias, but it makes sampling
uncertainty visible.

### Efficiency

The evaluator aggregates:

- mean, median, and nearest-rank P95 latency;
- input and generated token counts;
- maximum recorded per-sample peak CUDA memory.

## Failure cases

Incorrect, invalid, and runtime-error rows are copied to a separate JSONL file.
The next manual-analysis step should tag failures such as:

- event omission;
- object or attribute confusion;
- wrong temporal order;
- causal reasoning error;
- insufficient sampled evidence;
- language-prior guessing;
- malformed or ambiguous answer.

## Interpretation warning

This is a development pilot, not a full NExT-QA benchmark result. The same 48
questions have already been inspected during development. Do not use this
pilot as the only final test set after repeatedly changing prompts or training
settings.
