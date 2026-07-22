from __future__ import annotations

import numpy as np
import pytest

from longvideoguard.evaluation.counterfactuals import (
    diagnostic_summary,
    perturb_frames,
    random_mask_indices,
    top_relevant_indices,
)


def sample_frames() -> list[np.ndarray]:
    return [
        np.full((4, 4, 3), index, dtype=np.uint8)
        for index in range(8)
    ]


def test_reversal_and_shuffle_are_deterministic() -> None:
    frames = sample_frames()
    reversed_frames, _ = perturb_frames(
        frames,
        condition="reversed",
        sample_id="x",
    )
    assert int(reversed_frames[0][0, 0, 0]) == 7

    shuffled_a, indices_a = perturb_frames(
        frames,
        condition="shuffled",
        sample_id="x",
        seed=42,
    )
    shuffled_b, indices_b = perturb_frames(
        frames,
        condition="shuffled",
        sample_id="x",
        seed=42,
    )
    assert indices_a == indices_b
    assert [
        int(frame[0, 0, 0]) for frame in shuffled_a
    ] == [
        int(frame[0, 0, 0]) for frame in shuffled_b
    ]


def test_relevant_and_random_mask() -> None:
    frames = sample_frames()
    scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.5, 0.6]
    relevant, masked = perturb_frames(
        frames,
        condition="relevant_mask",
        sample_id="x",
        query_scores=scores,
        mask_count=2,
    )
    assert masked == [1, 3]
    assert relevant[1].sum() == 0
    assert relevant[3].sum() == 0

    first = random_mask_indices(
        8,
        count=2,
        sample_id="x",
        seed=42,
    )
    second = random_mask_indices(
        8,
        count=2,
        sample_id="x",
        seed=42,
    )
    assert first == second


def test_top_relevant_indices() -> None:
    assert top_relevant_indices(
        [0.1, 0.9, 0.5],
        count=2,
    ) == [1, 2]


def test_black_frames() -> None:
    black, masked = perturb_frames(
        sample_frames(),
        condition="black",
        sample_id="x",
    )
    assert masked == list(range(8))
    assert all(frame.sum() == 0 for frame in black)


def make_prediction(
    sample_id: str,
    *,
    category: str,
    correct: bool,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "question_category": category,
        "is_correct": correct,
    }


def test_diagnostic_summary() -> None:
    original = [
        make_prediction("a", category="temporal", correct=True),
        make_prediction("b", category="causal", correct=True),
    ]
    conditions = {
        "original": original,
        "reversed": [
            make_prediction("a", category="temporal", correct=False),
            make_prediction("b", category="causal", correct=True),
        ],
        "shuffled": [
            make_prediction("a", category="temporal", correct=False),
            make_prediction("b", category="causal", correct=True),
        ],
        "black": [
            make_prediction("a", category="temporal", correct=False),
            make_prediction("b", category="causal", correct=False),
        ],
        "relevant_mask": [
            make_prediction("a", category="temporal", correct=False),
            make_prediction("b", category="causal", correct=True),
        ],
        "random_mask": [
            make_prediction("a", category="temporal", correct=True),
            make_prediction("b", category="causal", correct=True),
        ],
        "question_only": [
            make_prediction("a", category="temporal", correct=False),
            make_prediction("b", category="causal", correct=True),
        ],
    }
    summary = diagnostic_summary(conditions)
    assert summary["visual_dependence"][
        "original_minus_black"
    ] == pytest.approx(1.0)
    assert summary["temporal_order_sensitivity"][
        "drop_after_reversal"
    ] == pytest.approx(1.0)
    assert summary["evidence_removal"][
        "relevant_minus_random_drop"
    ] == pytest.approx(0.5)
