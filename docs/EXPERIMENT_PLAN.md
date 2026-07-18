# Experiment Plan

## Stage 0 — Engineering validation

Acceptance criteria:

- project installs in a clean environment;
- unit tests pass;
- local video metadata can be probed;
- uniform frame indices and timestamps are reproducible.

## Stage 1 — Pilot datasets

### NExT-QA pilot

Purpose:

- causal reasoning;
- temporal reasoning;
- descriptive QA.

Deliverables:

- metadata adapter;
- frozen validation subset;
- multiple-choice scoring;
- per-question-type metrics.

### Charades-STA pilot

Purpose:

- temporal grounding from a natural-language query.

Deliverables:

- metadata adapter;
- start/end-time parser;
- temporal IoU;
- Recall@1 at IoU thresholds.

## Stage 2 — Zero-shot baseline

Compare:

- 8, 16, and 32 uniformly sampled frames;
- short and structured prompts;
- free-text versus JSON output.

Record:

- correctness;
- invalid output rate;
- latency;
- peak GPU memory;
- visual-token or input-size proxy.

## Stage 3 — Data construction

Construct:

- answerable QA;
- counterfactual/unanswerable QA;
- temporal interval targets;
- evidence timestamps;
- uncertainty outputs.

Quality controls:

- deterministic schema validation;
- duplicate detection;
- video-level split integrity;
- automatic contradiction checks;
- manual review sample.

## Stage 4 — LoRA/QLoRA SFT

Sanity checks:

1. overfit a tiny sample;
2. verify assistant-only loss masking;
3. inspect generations after a small number of steps;
4. save adapter and exact config;
5. resume from checkpoint.

## Stage 5 — Ablations

Minimum useful ablations:

- 8 vs 16 vs 32 frames;
- uniform vs key-frame sampling;
- caption-only vs mixed instruction data;
- without vs with hard negatives;
- LoRA rank 8 vs 16;
- without vs with explicit evidence supervision.

## Stage 6 — Failure analysis

Tag failures as:

- event omission;
- wrong object;
- wrong attribute;
- wrong relation;
- temporal-boundary error;
- causal reasoning error;
- unsupported answer;
- malformed JSON;
- uncertainty calibration error.
