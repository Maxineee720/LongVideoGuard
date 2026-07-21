from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from longvideoguard.training.stage6a_data import (
    assert_disjoint_splits,
    build_video_split,
    load_manifest_rows,
    normalize_video_filename,
    split_stats,
    write_qwen_json,
)


@dataclass(frozen=True)
class FakeRecord:
    video_id: str
    qid: str
    question: str
    options: tuple[str, str, str, str, str]
    answer_index: int
    question_type: str
    category: str

    @property
    def sample_id(self) -> str:
        return f"{self.video_id}:{self.qid}"


def make_records(prefix: str, num_videos: int) -> list[FakeRecord]:
    categories = ("causal", "temporal", "descriptive")
    question_types = ("CW", "TN", "DL")
    rows: list[FakeRecord] = []

    for video_index in range(num_videos):
        for question_index in range(5):
            category_index = (video_index + question_index) % 3
            rows.append(
                FakeRecord(
                    video_id=f"{prefix}{video_index}",
                    qid=str(question_index),
                    question=f"Question {video_index}-{question_index}?",
                    options=("a", "b", "c", "d", "e"),
                    answer_index=question_index % 5,
                    question_type=question_types[category_index],
                    category=categories[category_index],
                )
            )
    return rows


def test_normalize_video_filename() -> None:
    assert normalize_video_filename("nested/123") == "123.mp4"
    assert normalize_video_filename(r"nested\123.avi") == "123.avi"


def test_build_video_split_is_deterministic() -> None:
    records = make_records("train_", 20)
    first = build_video_split(
        records,
        role="qa_train",
        source_split="train",
        video_id_map={},
        num_videos=8,
        max_questions_per_video=4,
        seed=42,
    )
    second = build_video_split(
        records,
        role="qa_train",
        source_split="train",
        video_id_map={},
        num_videos=8,
        max_questions_per_video=4,
        seed=42,
    )

    assert first == second
    assert len(first) == 32
    assert split_stats(first)["num_videos"] == 8


def test_disjoint_split_check_rejects_physical_overlap() -> None:
    with pytest.raises(ValueError, match="Data leakage"):
        assert_disjoint_splits(
            {
                "left": [
                    {
                        "video_id": "logical_a",
                        "video_relpath": "same.mp4",
                    }
                ],
                "right": [
                    {
                        "video_id": "logical_b",
                        "video_relpath": "same.mp4",
                    }
                ],
            }
        )


def test_qwen_export(tmp_path: Path) -> None:
    rows = build_video_split(
        make_records("train_", 6),
        role="qa_train",
        source_split="train",
        video_id_map={},
        num_videos=2,
        max_questions_per_video=3,
        seed=7,
    )
    path = write_qwen_json(rows, tmp_path / "qa_train.qwen.json")
    text = path.read_text(encoding="utf-8")
    assert '"video"' in text
    assert '"conversations"' in text
    assert "<video>" in text


def test_load_manifest_rows_preserves_physical_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "pilot.jsonl"
    manifest.write_text(
        '{"video_id":"v1","video_relpath":"nested/one.mp4"}\n'
        '{"video_id":"v1","video_relpath":"nested/one.mp4"}\n',
        encoding="utf-8",
    )
    rows = load_manifest_rows([manifest])
    assert rows[0]["video_id"] == "v1"
    assert rows[0]["video_relpath"] == "nested/one.mp4"
