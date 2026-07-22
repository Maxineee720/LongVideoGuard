# Stage 9D — Final Publishing Checklist

## README

- [ ] Replace `<YOUR_REPOSITORY_URL>`.
- [ ] Copy `README_FINAL.md` to repository root as `README.md`.
- [ ] Copy final figures into a version-controlled `assets/` or `docs/assets/` directory.
- [ ] Update image paths in the README.
- [ ] Add one overview screenshot and one live-demo screenshot.
- [ ] Remove all Colab-local `/content/...` paths.
- [ ] Verify all commands from a clean environment.
- [ ] Add license information if appropriate.

## Git

Recommended tracked files:

```text
README.md
streamlit_app.py
src/
scripts/
tests/
docs/
requirements*.txt
```

Do not commit:

```text
data/raw/
outputs/
*.mp4
*.zip
model weights
adapter checkpoints
Google Drive artifacts
```

## Resume

- [ ] Use the three-bullet version for multimodal algorithm roles.
- [ ] Use the engineering-oriented version for ML platform / inference roles.
- [ ] Keep only 2–3 metrics in the final resume.
- [ ] Mention Frozen Evaluation in interviews, not necessarily in every resume bullet.

## Interview

Prepare these four layers:

1. 30-second overview;
2. 3-minute project pitch;
3. one successful experiment;
4. one negative result and what you learned.

Recommended examples:

- Successful: Uniform-8 accuracy–efficiency result.
- Negative: hard-negative audit or Scene-aware development gain not generalizing.
- Debugging: Transformers CLIP output compatibility and selected-frame video encoding.
- Research maturity: Swap-video and Frozen Evaluation.

## Final project positioning

Use this sentence:

> LongVideoGuard is not just a fine-tuning project; it is a reliability-focused VideoQA study covering training validation, shortcut detection, retrieval, routing, counterfactual evaluation, data auditing, and frozen generalization.

## Do not overclaim

Avoid:

- “Scene-aware significantly improved performance.”
- “The model fully understands temporal order.”
- “CLIP selected the true evidence frames.”
- “Hard negatives were clean.”
- “The LoRA adapter improved standard VideoQA generalization.”

Use:

- “Directional improvement on development.”
- “No statistically significant frozen difference.”
- “Evidence of visual dependence.”
- “Incomplete temporal-order sensitivity.”
- “CLIP similarity was useful for retrieval but not faithful attribution.”
