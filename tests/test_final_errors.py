from __future__ import annotations

import pytest

from longvideoguard.analysis.final_errors import (
    bucket_cases,
    build_case_table,
    build_summary,
    majority_vote_prediction,
)


def sample_manifest() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "a",
            "video_id": "v1",
            "question": "Question a?",
            "question_category": "causal",
            "options": ["a", "b", "c", "d", "e"],
            "gold_answer_letter": "A",
        },
        {
            "sample_id": "b",
            "video_id": "v2",
            "question": "Question b?",
            "question_category": "temporal",
            "options": ["a", "b", "c", "d", "e"],
            "gold_answer_letter": "B",
        },
    ]


def prediction(
    sample_id: str,
    value: str,
    correct: bool,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "prediction": value,
        "is_correct": correct,
    }


def test_build_and_bucket_cases() -> None:
    predictions = {
        "uniform_8": [
            prediction("a", "A", True),
            prediction("b", "C", False),
        ],
        "scene_aware_8": [
            prediction("a", "A", True),
            prediction("b", "B", True),
        ],
        "uniform_16": [
            prediction("a", "D", False),
            prediction("b", "D", False),
        ],
    }
    cases = build_case_table(
        sample_manifest(),
        predictions,
    )
    buckets = bucket_cases(cases)
    assert len(buckets["uniform8_correct_uniform16_wrong"]) == 1
    assert len(buckets["scene_correct_uniform8_wrong"]) == 1


def test_majority_vote() -> None:
    case = {
        "uniform_8_prediction": "A",
        "scene_aware_8_prediction": "A",
        "uniform_16_prediction": "B",
    }
    assert majority_vote_prediction(case) == "A"


def test_summary_accuracy() -> None:
    predictions = {
        "uniform_8": [
            prediction("a", "A", True),
            prediction("b", "C", False),
        ],
        "scene_aware_8": [
            prediction("a", "A", True),
            prediction("b", "B", True),
        ],
        "uniform_16": [
            prediction("a", "D", False),
            prediction("b", "D", False),
        ],
    }
    cases = build_case_table(
        sample_manifest(),
        predictions,
    )
    summary = build_summary(cases)
    assert summary["method_accuracy"]["uniform_8"] == pytest.approx(0.5)
    assert summary["method_accuracy"]["scene_aware_8"] == pytest.approx(1.0)


def test_prediction_id_mismatch_is_rejected() -> None:
    predictions = {
        "uniform_8": [prediction("a", "A", True)],
        "scene_aware_8": [prediction("a", "A", True)],
        "uniform_16": [prediction("a", "A", True)],
    }
    with pytest.raises(ValueError, match="do not match"):
        build_case_table(sample_manifest(), predictions)
