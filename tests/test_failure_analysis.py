from pathlib import Path

import numpy as np
import pytest

from longvideoguard.failure_analysis import (
    build_review_rows,
    make_contact_sheet,
    validate_review_rows,
    write_review_html,
)


def test_make_contact_sheet_shape() -> None:
    frames = [
        np.full((100, 200, 3), fill_value=index * 30, dtype=np.uint8)
        for index in range(5)
    ]
    sheet = make_contact_sheet(
        frames,
        frame_indices=[0, 10, 20, 30, 40],
        timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
        columns=4,
        tile_width=160,
        tile_height=100,
    )
    assert sheet.shape == (2 * 134, 4 * 160, 3)


def test_build_review_rows_preserves_ground_truth() -> None:
    errors = [
        {
            "sample_id": "v1:1",
            "video_id": "v1",
            "question_category": "temporal",
            "question_type": "TN",
            "question": "What happened next?",
            "options": ["a", "b", "c", "d", "e"],
            "gold_answer_letter": "B",
            "predicted_letter": "C",
            "raw_output": "C",
        }
    ]
    rows = build_review_rows(
        errors,
        contact_sheet_by_video={"v1": "contact_sheets/v1.jpg"},
    )
    assert rows[0]["gold_answer_letter"] == "B"
    assert rows[0]["predicted_letter"] == "C"
    assert rows[0]["failure_type"] is None
    assert rows[0]["evidence_covered"] is None


def test_validate_review_rows() -> None:
    rows = [
        {
            "sample_id": "v1:1",
            "failure_type": "sampling_miss",
            "evidence_covered": False,
            "attribution_confidence": "high",
        },
        {
            "sample_id": "v2:1",
            "failure_type": None,
            "evidence_covered": None,
            "attribution_confidence": None,
        },
    ]
    summary = validate_review_rows(rows)
    assert summary["labelled"] == 1
    assert summary["completion_rate"] == pytest.approx(0.5)
    assert summary["failure_type_counts"] == {"sampling_miss": 1}
    assert summary["incomplete_sample_ids"] == ["v2:1"]


def test_write_review_html(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "v1:1",
            "video_id": "v1",
            "question_category": "causal",
            "question": "Why?",
            "options": ["a", "b", "c", "d", "e"],
            "gold_answer_letter": "A",
            "predicted_letter": "B",
            "contact_sheet": "contact_sheets/v1.jpg",
        }
    ]
    output = write_review_html(rows, tmp_path / "review.html")
    text = output.read_text(encoding="utf-8")
    assert "v1:1" in text
    assert "contact_sheets/v1.jpg" in text
    assert "sampling_miss" in text
