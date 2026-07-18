# Qwen3-VL zero-shot baseline

## Goal

Run a deterministic multiple-choice video-QA baseline before any prompt tuning
or supervised fine-tuning. Every raw model output is saved to JSONL.

## Why Colab GPU

The official Qwen3-VL Transformers implementation requires
`transformers>=4.57.0`. The 2B model is selected to keep the first complete
pipeline feasible on a single Colab GPU. The smoke test should use one sample
before running all 48 pilot questions.

## Colab setup

```bash
!git clone https://github.com/Maxineee720/LongVideoGuard.git
%cd LongVideoGuard

!pip install -U pip
!pip install -e ".[dev,vlm]"
```

Upload or mount the following local, Git-ignored files:

```text
data/processed/nextqa/pilot_val.jsonl
data/raw/nextqa/videos/*.mp4
```

A practical option is to zip only the 12 pilot videos on the Mac, upload the
small pilot archive to Google Drive, and extract it inside Colab.

## Single-sample smoke test

```bash
!python scripts/run_nextqa_zero_shot.py \
  data/processed/nextqa/pilot_val.jsonl \
  data/raw/nextqa/videos \
  --num-frames 8 \
  --dtype float16 \
  --limit 1 \
  --output outputs/predictions/smoke_qwen3vl2b_frames8.jsonl \
  --overwrite
```

Inspect the output:

```bash
!python scripts/inspect_prediction_file.py \
  outputs/predictions/smoke_qwen3vl2b_frames8.jsonl
```

## Pilot run

After the smoke test succeeds:

```bash
!python scripts/run_nextqa_zero_shot.py \
  data/processed/nextqa/pilot_val.jsonl \
  data/raw/nextqa/videos \
  --num-frames 16 \
  --output outputs/predictions/nextqa_qwen3vl2b_zero_shot_frames16.jsonl \
  --overwrite
```

The script appends one result at a time. Without `--overwrite`, rerunning it
skips sample IDs already present in the output file.

## Output fields

Each JSONL row records:

- model and generation settings;
- sample, question, options, and gold label;
- raw generated text;
- parsed option letter;
- validity and correctness;
- latency;
- input and generated token counts;
- peak CUDA memory when available;
- any runtime error.

Do not report aggregate accuracy until the Stage 4 evaluator has been run.


## Colab GPU dtype note

Use `float16` for the broadest compatibility, especially on a T4. On an
A100 or another GPU with native BF16 support, `bfloat16` is also appropriate.
The default `auto` follows the model/checkpoint and Transformers settings.
