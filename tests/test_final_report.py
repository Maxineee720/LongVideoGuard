from __future__ import annotations

import pytest

from longvideoguard.reporting.final_report import (
    frozen_accuracy_rows,
    percentage,
)


def test_percentage() -> None:
    assert percentage(0.6875) == pytest.approx(68.75)


def test_frozen_accuracy_rows() -> None:
    summary = {
        "conditions": {
            method: {
                "correct": 10,
                "count": 20,
                "accuracy": 0.5,
                "input_token_count": {"mean": 100.0},
                "latency_seconds": {"mean": 0.2},
                "peak_gpu_memory_mb": {"mean": 1000.0},
                "wilson_95_interval": {
                    "lower": 0.3,
                    "upper": 0.7,
                },
            }
            for method in (
                "uniform_8",
                "scene_aware_8",
                "uniform_16",
            )
        }
    }
    rows = frozen_accuracy_rows(summary)
    assert len(rows) == 3
    assert rows[0]["accuracy_percent"] == pytest.approx(50.0)
