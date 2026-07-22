# Stage 9B — Final tables, figures, and README

This stage converts completed experiment outputs into a compact public-facing
project report.

## Inputs

```text
Stage 7B sampling evaluation summary
Stage 7C.1 Qwen router summary
Stage 7C.2 counterfactual summary
Stage 8 frozen VideoQA summary
Stage 9A final error summary
Optional Stage 6B hard-negative audit summary
```

## Run

```bash
python scripts/build_stage9_final_report.py \
  --hard-negative-audit \
    /content/drive/MyDrive/LongVideoGuard/results/stage6b_3/audit_summary.json \
  --output-dir outputs/stage9/project_report
```

The audit argument is optional. When omitted, the generated README uses the
known 32-example audit finding from the project narrative.

## Outputs

```text
outputs/stage9/project_report/
├── README_GENERATED.md
├── final_metrics.json
├── figures/
│   ├── frozen_accuracy.png
│   ├── frozen_category_accuracy.png
│   ├── accuracy_efficiency.png
│   ├── counterfactual_accuracy.png
│   └── error_buckets.png
└── tables/
    ├── frozen_accuracy.csv
    ├── frozen_category_accuracy.csv
    ├── development_sampling.csv
    ├── counterfactual_accuracy.csv
    └── error_buckets.csv
```

## Publishing

Review `README_GENERATED.md`, then copy it to the repository root as
`README.md` only after checking paths, repository links, installation
commands, and any screenshots you want to include.

Do not change frozen metrics or claim statistical significance that was not
observed.
