# Stage 5.2 — Inspect the real Qwen3-VL SFT batch

## Goal

Before LoRA training, prove that one real video sample produces a valid
multimodal batch and that loss is applied only to the assistant answer.

The check must establish:

```text
video file → processor → video tensors
user/video prompt → labels = -100
assistant answer and chat terminator → supervised labels
base model forward pass → finite loss
```

## Why prompt-only and full conversations are processed separately

The full training conversation contains:

```text
user(video + question) → assistant(answer)
```

A second prompt-only conversation contains the same user/video input and adds
the assistant-generation prefix. After tokenization, this prompt must be an
exact prefix of the full training sequence. The prefix is masked with `-100`;
only the suffix belonging to the assistant response contributes to
cross-entropy loss.

This is checked instead of assuming a fixed number of text tokens, because
video placeholders expand into a variable number of visual tokens.

## Prepare Colab data

The following files must already exist:

```text
data/processed/nextqa/sft/sft_overfit_train.jsonl
data/raw/nextqa/sft_videos/*.mp4
```

The first Stage 5.2 run uses eight frames and one example.

## Batch-only inspection

```bash
python scripts/inspect_nextqa_sft_batch.py \
  data/processed/nextqa/sft/sft_overfit_train.jsonl \
  data/raw/nextqa/sft_videos \
  --num-frames 8 \
  --sample-index 0
```

This loads only the processor and prints:

- all batch keys;
- text and video tensor shapes;
- full sequence length;
- prompt-prefix length;
- masked and supervised token counts;
- decoded supervised suffix;
- proof that all prompt labels are `-100`;
- proof that all assistant suffix labels are supervised.

## Forward-loss inspection

On an A100:

```bash
python scripts/inspect_nextqa_sft_batch.py \
  data/processed/nextqa/sft/sft_overfit_train.jsonl \
  data/raw/nextqa/sft_videos \
  --num-frames 8 \
  --sample-index 0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --forward-pass
```

Success requires:

```text
pixel_values_videos/video_grid_thw present
prompt_is_exact_prefix = true
all_prompt_labels_masked = true
all_assistant_labels_supervised = true
supervised_token_count > 0
supervised_text_clean contains the gold answer letter
forward loss is finite
```

## Interpretation

The assistant target is usually represented by the answer-letter token plus a
chat-end token. Therefore the number of supervised tokens may be greater than
one even though the decoded answer is a single letter.

This stage proves training-batch correctness. It does not prove that LoRA has
learned or that the model uses video evidence. Stage 5.3 will inject LoRA and
run the small overfit experiment.
