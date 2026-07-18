# LongVideoGuard Agent Rules

## Mission

Build a reproducible multimodal research project for evidence-grounded video understanding, temporal localization, and hallucination evaluation.

## Non-negotiable rules

- Never fabricate metrics, plots, logs, or completion claims.
- Do not describe a planned feature as implemented.
- Preserve raw predictions used to calculate every reported metric.
- Split by source video; never allow frames from one video to cross train/validation/test splits.
- Do not overwrite manually reviewed annotations.
- Do not commit datasets, model weights, checkpoints, tokens, or secrets.
- Prefer small, testable changes over broad rewrites.
- Use `pathlib`, type hints, logging, and explicit configuration.
- Add or update tests for behavior changes.
- Run relevant tests before reporting completion.
- Record commands run, files changed, tests passed, and unresolved risks.
- Keep notebooks exploratory; reusable logic belongs in `src/longvideoguard`.
- Keep README claims synchronized with actual implementation.

## Research discipline

Every experiment must record:

- experiment ID;
- dataset split/version;
- model and revision;
- prompt template;
- frame sampling strategy and frame count;
- precision/quantization;
- seed;
- hardware;
- raw predictions;
- metrics;
- failure notes.
