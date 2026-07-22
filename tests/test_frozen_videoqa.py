from __future__ import annotations

import pytest

from longvideoguard.evaluation.frozen_videoqa import (
    attach_confidence_interval,
    validate_video_disjointness,
    wilson_interval,
)


def test_wilson_interval_contains_proportion() -> None:
    lower, upper = wilson_interval(75, 100)
    assert lower < 0.75 < upper


def test_disjointness() -> None:
    frozen = [
        {"video_id": "f1"},
        {"video_id": "f2"},
    ]
    references = {
        "pilot": [{"video_id": "p1"}],
        "train": [{"video_id": "t1"}],
    }
    result = validate_video_disjointness(
        frozen,
        references,
    )
    assert result["all_disjoint"]
    assert result["frozen_video_count"] == 2


def test_overlap_is_reported() -> None:
    result = validate_video_disjointness(
        [{"video_id": "v1"}],
        {"pilot": [{"video_id": "v1"}]},
    )
    assert not result["all_disjoint"]
    assert result["overlap_counts"]["pilot"] == 1


def test_attach_confidence_interval() -> None:
    payload = attach_confidence_interval(
        {
            "count": 48,
            "correct": 33,
            "accuracy": 33 / 48,
        }
    )
    assert "wilson_95_interval" in payload
    assert payload["wilson_95_interval"]["lower"] < 33 / 48
