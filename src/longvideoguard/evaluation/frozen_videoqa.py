from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FROZEN_CONDITIONS = (
    "uniform_8",
    "scene_aware_8",
    "uniform_16",
)


def video_id(row: Mapping[str, object]) -> str:
    value = row.get("video_id") or row.get("source_video_id")
    if value is None:
        raise ValueError(
            f"Missing video_id for sample {row.get('sample_id')!r}."
        )
    return str(value)


def validate_video_disjointness(
    frozen_rows: Sequence[Mapping[str, object]],
    reference_sets: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
) -> dict[str, object]:
    frozen_ids = {video_id(row) for row in frozen_rows}
    overlaps: dict[str, list[str]] = {}

    for name, rows in reference_sets.items():
        reference_ids = {video_id(row) for row in rows}
        common = sorted(frozen_ids & reference_ids)
        overlaps[name] = common

    return {
        "frozen_video_count": len(frozen_ids),
        "reference_video_counts": {
            name: len({video_id(row) for row in rows})
            for name, rows in reference_sets.items()
        },
        "overlap_counts": {
            name: len(values) for name, values in overlaps.items()
        },
        "overlaps": overlaps,
        "all_disjoint": all(not values for values in overlaps.values()),
    }


def wilson_interval(
    correct: int,
    count: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if count <= 0:
        raise ValueError("count must be positive")
    if correct < 0 or correct > count:
        raise ValueError("correct must be in [0, count]")

    proportion = correct / count
    denominator = 1.0 + (z * z) / count
    center = (
        proportion + (z * z) / (2.0 * count)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + (z * z) / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def attach_confidence_interval(
    summary: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(summary)
    correct = int(payload["correct"])
    count = int(payload["count"])
    lower, upper = wilson_interval(correct, count)
    payload["wilson_95_interval"] = {
        "lower": lower,
        "upper": upper,
    }
    return payload


def write_jsonl(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return destination


def write_json(
    payload: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
