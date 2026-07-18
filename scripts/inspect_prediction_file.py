from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show a compact summary of a zero-shot prediction JSONL."
    )
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("prediction file is empty")

    valid = sum(bool(row["is_valid_prediction"]) for row in rows)
    correct = sum(bool(row["is_correct"]) for row in rows)
    errors = sum(row.get("error") is not None for row in rows)

    print(f"rows: {len(rows)}")
    print(f"valid predictions: {valid}/{len(rows)}")
    print(f"correct: {correct}/{len(rows)}")
    print(f"errors: {errors}/{len(rows)}")

    for row in rows[:5]:
        print(
            f"{row['sample_id']}: raw={row['raw_output']!r}, "
            f"pred={row['predicted_letter']!r}, "
            f"gold={row['gold_answer_letter']}, "
            f"correct={row['is_correct']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
