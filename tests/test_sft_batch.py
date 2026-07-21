from __future__ import annotations

from pathlib import Path

import pytest

from longvideoguard.training.sft_batch import (
    load_sft_jsonl_row,
    masked_label_values,
    prompt_prefix_length,
    validate_sft_row,
)


def test_prompt_prefix_and_assistant_only_mask() -> None:
    prompt_ids = [10, 11, 12, 13]
    full_ids = [10, 11, 12, 13, 20, 21]

    assert prompt_prefix_length(full_ids, prompt_ids) == 4
    assert masked_label_values(full_ids, prompt_ids) == [
        -100,
        -100,
        -100,
        -100,
        20,
        21,
    ]


def test_prefix_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="not an exact prefix"):
        prompt_prefix_length(
            [10, 11, 99, 20],
            [10, 11, 12],
        )


def test_validate_sft_row_rejects_bad_target() -> None:
    with pytest.raises(ValueError, match="assistant_target"):
        validate_sft_row(
            {
                "sample_id": "v1:1",
                "video_id": "v1",
                "video_relpath": "v1.mp4",
                "prompt": "Question",
                "assistant_target": "F",
            }
        )


def test_load_sft_jsonl_row(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text(
        '{"sample_id":"v1:1","video_id":"v1",'
        '"video_relpath":"v1.mp4","prompt":"Question",'
        '"assistant_target":"B"}\n'
        '{"sample_id":"v2:1","video_id":"v2",'
        '"video_relpath":"v2.mp4","prompt":"Question 2",'
        '"assistant_target":"C"}\n',
        encoding="utf-8",
    )

    row = load_sft_jsonl_row(path, index=1)
    assert row["sample_id"] == "v2:1"
    assert row["assistant_target"] == "C"
