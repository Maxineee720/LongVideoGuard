from __future__ import annotations

import argparse
import json
from pathlib import Path

from longvideoguard.failure_analysis import load_jsonl, validate_review_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize manually completed failure labels."
    )
    parser.add_argument("review_jsonl", type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.review_jsonl)
    summary = validate_review_rows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["completion_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
