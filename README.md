# LongVideoGuard

**Evidence-grounded post-training and evaluation of lightweight video-language models for temporal reasoning, event localization, and hallucination control.**

> Project status: **Milestone 0 — repository scaffold and video sampling utilities.**  
> No training results are claimed yet.

## Motivation

Video-language models can answer questions about video, but they may miss short events, return inaccurate timestamps, or produce answers unsupported by visual evidence. LongVideoGuard studies whether a lightweight open-source VLM can be improved through:

- evidence-aware video question answering;
- temporal event localization;
- key-frame selection;
- hard negative and unanswerable examples;
- LoRA/QLoRA supervised fine-tuning;
- explicit hallucination and uncertainty evaluation.

## Research questions

1. How do frame count and sampling strategy affect temporal reasoning accuracy and inference cost?
2. Does mixed instruction tuning improve causal/temporal QA over caption-only tuning?
3. Do evidence-frame supervision and hard negatives reduce unsupported answers?
4. How much performance is gained by fine-tuning compared with prompt engineering alone?
5. Which failure modes remain after fine-tuning?

## Planned task suite

| Task | Input | Output | Main metrics |
|---|---|---|---|
| Video QA | video + question | answer | Accuracy / Exact Match / Token F1 |
| Temporal grounding | video + query | start/end time | mIoU / R@1 IoU thresholds |
| Evidence selection | video + question | frame IDs/timestamps | Evidence Recall / frame distance |
| Unanswerable detection | video + unsupported question | uncertain/refusal | Accuracy / false-positive rate |
| Structured generation | video + query | validated JSON | JSON valid rate / field accuracy |

## Initial model and datasets

The first baseline will target a lightweight video-capable VLM such as **Qwen3-VL-2B-Instruct**.

The first two public benchmarks under consideration are:

- **NExT-QA** for causal and temporal video question answering;
- **Charades-STA** for natural-language temporal grounding.

A small pilot subset will be used before any large-scale training.

## Repository layout

```text
LongVideoGuard/
├── configs/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/longvideoguard/
├── tests/
├── AGENTS.md
├── pyproject.toml
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Verify the environment:

```bash
python scripts/verify_environment.py
```

Inspect a local video:

```bash
python -m longvideoguard.cli probe path/to/video.mp4
```

Uniformly sample frame metadata:

```bash
python -m longvideoguard.cli sample-uniform path/to/video.mp4 --num-frames 16
```

Run tests:

```bash
pytest
```

## Reproducibility rules

- Never report a metric without saving the raw prediction file.
- Split data by original video, not by extracted frames.
- Keep test labels frozen before prompt or training experiments.
- Record model name, revision, frame sampling, prompt, seed, precision, hardware, and latency.
- Do not commit datasets, model weights, API keys, or generated checkpoints.
- Clearly label planned, partial, and completed features.

## Roadmap

- [x] GitHub-ready repository scaffold
- [x] Video metadata probing
- [x] Uniform frame-index sampling
- [ ] Dataset adapters for NExT-QA and Charades-STA
- [ ] Qwen3-VL zero-shot baseline
- [ ] Structured prediction schema and parser
- [ ] Frozen pilot test set
- [ ] Baseline evaluation
- [ ] LoRA/QLoRA SFT
- [ ] Hard-negative and uncertainty experiments
- [ ] Temporal-grounding evaluation
- [ ] Error analysis and demo

## License

Code is released under the MIT License. Dataset and model licenses remain governed by their original providers.
