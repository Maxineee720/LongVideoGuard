from __future__ import annotations

import pytest

from longvideoguard.training.stage6b_data import (
    ANSWERABLE,
    UNANSWERABLE,
    assert_cross_split_video_disjoint,
    build_negative_rows,
    build_positive_rows,
    deterministic_mix,
    structured_target,
    validate_negative_pairs,
)


def make_rows(prefix: str, count: int = 12) -> list[dict[str, object]]:
    categories = ("causal", "temporal", "descriptive")
    rows: list[dict[str, object]] = []
    for index in range(count):
        video_id = f"{prefix}_video_{index // 2}"
        rows.append(
            {
                "sample_id": f"{prefix}:{index}",
                "video_id": video_id,
                "video_relpath": f"{video_id}.mp4",
                "question": f"Question {index}?",
                "options": ["one", "two", "three", "four", "five"],
                "answer_index": index % 5,
                "answer_letter": ("A", "B", "C", "D", "E")[index % 5],
                "question_category": categories[index % 3],
            }
        )
    return rows


def test_structured_targets_are_exact() -> None:
    assert structured_target(
        status=ANSWERABLE,
        answer="D",
    ) == '{"status":"answerable","answer":"D"}'
    assert structured_target(
        status=UNANSWERABLE,
        answer=None,
    ) == '{"status":"unanswerable","answer":null}'

    with pytest.raises(ValueError):
        structured_target(status=ANSWERABLE, answer=None)


def test_positive_and_negative_builders() -> None:
    rows = make_rows("train")
    positives = build_positive_rows(rows, split_name="train")
    negatives = build_negative_rows(
        rows,
        split_name="train",
        negative_count=6,
        seed=42,
    )

    assert len(positives) == 12
    assert len(negatives) == 6
    assert all(row["is_answerable"] for row in positives)
    assert all(not row["is_answerable"] for row in negatives)
    assert all(
        row["video_id"] != row["source_question_video_id"]
        for row in negatives
    )
    assert validate_negative_pairs(negatives)["all_cross_video"]


def test_negative_builder_is_deterministic() -> None:
    rows = make_rows("train")
    first = build_negative_rows(
        rows,
        split_name="train",
        negative_count=6,
        seed=7,
    )
    second = build_negative_rows(
        rows,
        split_name="train",
        negative_count=6,
        seed=7,
    )
    assert first == second


def test_mixed_shuffle_is_deterministic() -> None:
    rows = make_rows("train")
    positives = build_positive_rows(rows, split_name="train")
    negatives = build_negative_rows(
        rows,
        split_name="train",
        negative_count=6,
        seed=5,
    )
    first = deterministic_mix(positives, negatives, seed=99)
    second = deterministic_mix(positives, negatives, seed=99)
    assert first == second
    assert len(first) == 18


def test_cross_split_video_leakage_is_rejected() -> None:
    train = make_rows("train", count=4)
    holdout = make_rows("holdout", count=4)
    assert_cross_split_video_disjoint(
        {
            "train": train,
            "holdout": holdout,
        }
    )

    holdout[0]["video_id"] = train[0]["video_id"]
    with pytest.raises(ValueError, match="Video leakage"):
        assert_cross_split_video_disjoint(
            {
                "train": train,
                "holdout": holdout,
            }
        )
