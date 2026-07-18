# Decision Log

## D001 — Start a clean repository

The earlier YOLO inspection prototype is not reused as the foundation because the new project has a different research question: video-language post-training and evaluation rather than object-detection application integration.

## D002 — Begin with a 2B-class model

The first model target is Qwen3-VL-2B-Instruct to reduce Colab memory risk and shorten iteration cycles. Larger models may be added only after the complete pipeline is reproducible.

## D003 — Use two complementary benchmark families

NExT-QA covers causal and temporal reasoning. Charades-STA covers temporal grounding. Their roles are distinct and should not be merged into one metric.

## D004 — Freeze the pilot test set early

Prompt engineering and training can otherwise leak test information into repeated manual iteration. A small pilot test set will be defined and frozen before model tuning.

## D005 — GitHub is an output, not the research goal

The repository must expose evidence of reproducibility and reasoning. A polished page cannot compensate for missing baselines, raw predictions, and failure analysis.
