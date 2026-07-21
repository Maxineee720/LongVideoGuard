from __future__ import annotations

import pytest

from longvideoguard.training.stage6a_training import (
    deterministic_subset,
    is_better_checkpoint,
    shuffled_epoch_indices,
    swap_summary,
    update_patience,
)


def test_shuffled_epoch_indices_are_deterministic() -> None:
    first = shuffled_epoch_indices(
        num_samples=8,
        epochs=3,
        seed=42,
    )
    second = shuffled_epoch_indices(
        num_samples=8,
        epochs=3,
        seed=42,
    )
    assert first == second
    assert len(first) == 3
    assert all(sorted(epoch) == list(range(8)) for epoch in first)
    assert first[0] != first[1]


def test_checkpoint_selection_prioritises_accuracy_then_loss() -> None:
    assert is_better_checkpoint(
        accuracy=0.7,
        loss=1.0,
        best_accuracy=None,
        best_loss=None,
    )
    assert is_better_checkpoint(
        accuracy=0.8,
        loss=2.0,
        best_accuracy=0.7,
        best_loss=1.0,
    )
    assert is_better_checkpoint(
        accuracy=0.8,
        loss=0.9,
        best_accuracy=0.8,
        best_loss=1.0,
    )
    assert not is_better_checkpoint(
        accuracy=0.7,
        loss=0.1,
        best_accuracy=0.8,
        best_loss=1.0,
    )


def test_deterministic_subset_preserves_source_order() -> None:
    rows = [{"sample_id": str(index)} for index in range(10)]
    first = deterministic_subset(rows, max_samples=4, seed=7)
    second = deterministic_subset(rows, max_samples=4, seed=7)
    assert first == second
    selected = [int(row["sample_id"]) for row in first]
    assert selected == sorted(selected)


def test_patience_and_swap_summary() -> None:
    assert update_patience(improved=True, bad_epochs=2) == 0
    assert update_patience(improved=False, bad_epochs=2) == 3

    correct = [
        {
            "sample_id": "a",
            "predicted_letter": "A",
            "is_correct": True,
        },
        {
            "sample_id": "b",
            "predicted_letter": "B",
            "is_correct": True,
        },
    ]
    swapped = [
        {
            "sample_id": "a",
            "predicted_letter": "A",
            "is_correct": True,
        },
        {
            "sample_id": "b",
            "predicted_letter": "C",
            "is_correct": False,
        },
    ]
    summary = swap_summary(correct, swapped)
    assert summary["prediction_change_rate"] == pytest.approx(0.5)
    assert summary["swapped_video_accuracy"] == pytest.approx(0.5)
    assert summary["text_memorisation_warning_rate"] == pytest.approx(0.5)
