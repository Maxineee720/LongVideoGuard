# Data directory

Raw datasets are intentionally excluded from Git.

Expected local layout:

```text
data/
├── raw/
│   ├── nextqa/
│   └── charades_sta/
├── processed/
├── cache/
└── README.md
```

Every processed dataset version should have:

- a manifest;
- a source and license note;
- split statistics;
- a schema version;
- a generation script;
- a validation report.

Never distribute dataset files through this repository unless the original license explicitly permits redistribution.
