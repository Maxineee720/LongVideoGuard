# Project Proposal

## Title

**LongVideoGuard: Post-training Video-Language Models for Evidence-grounded Temporal Reasoning**

## Problem statement

Lightweight video-language models can generate fluent answers while missing brief events, returning inaccurate time intervals, or asserting events that are not visible. This project builds a reproducible pipeline to measure and reduce these failures.

## Scope

The first version uses text, images/frames, and video only. Audio is excluded to keep the research question and compute budget controlled.

The project covers:

- causal and temporal VideoQA;
- sentence-to-video temporal grounding;
- evidence-frame selection;
- unanswerable/counterfactual questions;
- LoRA or QLoRA SFT;
- accuracy, grounding, hallucination, uncertainty, and efficiency evaluation.

## Initial hypothesis

A mixed instruction dataset containing QA, temporal grounding, evidence supervision, hard negatives, and explicit uncertainty examples will outperform prompt-only and caption-only baselines on supported answers and false-positive rate.

## Baselines

1. Majority/random baseline where applicable.
2. Zero-shot VLM with uniform frame sampling.
3. Prompt-optimized VLM without training.
4. Alternative frame counts.
5. Caption-only LoRA.
6. Mixed-task LoRA.

## Out of scope for version 1

- training a VLM from scratch;
- full-parameter fine-tuning;
- audio understanding;
- multi-agent orchestration;
- real-time surveillance deployment;
- large web application;
- RLHF or large-scale reinforcement learning.
