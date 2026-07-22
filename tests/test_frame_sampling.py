from __future__ import annotations

import numpy as np
import pytest

from longvideoguard.retrieval.frame_sampling import (
    build_query_text,
    cosine_scores,
    evenly_spaced_indices,
    frame_change_scores,
    query_aware_indices,
    scene_aware_indices,
    temporal_coverage,
)


def test_evenly_spaced_indices() -> None:
    result = evenly_spaced_indices(32, 8)
    assert len(result) == 8
    assert result == sorted(set(result))
    assert result[0] == 0 and result[-1] == 31


def test_scene_change() -> None:
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    scores = frame_change_scores([black, black.copy(), white])
    assert scores[2] > 0.5
    assert 2 in scene_aware_indices(scores, 2)


def test_query_aware_selection() -> None:
    images = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    selected, relevance = query_aware_indices(images, np.array([1.0, 0.0]), 2, 0.9, 1)
    assert 0 in selected
    assert relevance[0] > relevance[2]


def test_cosine_scores() -> None:
    scores = cosine_scores(np.array([[2.0, 0.0], [0.0, 3.0]]), np.array([10.0, 0.0]))
    assert scores.tolist() == pytest.approx([1.0, 0.0])


def test_query_text() -> None:
    row = {"question": "What happens?", "options": ["a", "b", "c", "d", "e"]}
    assert build_query_text(row) == "What happens?"
    assert "A: a" in build_query_text(row, "question_options")


def test_temporal_coverage() -> None:
    assert temporal_coverage([0, 7], 8) == pytest.approx(1.0)
