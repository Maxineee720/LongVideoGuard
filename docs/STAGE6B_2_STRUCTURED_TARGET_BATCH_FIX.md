# Stage 6B.2 structured-target batch fix

The original Stage 5 SFT batch validator accepted only single-letter A-E
targets. Stage 6B uses structured JSON assistant targets, so teacher-forced
loss failed before training.

This patch keeps strict A-E validation for ordinary QA rows and adds strict
canonical JSON validation for Stage 6B answerability rows.
