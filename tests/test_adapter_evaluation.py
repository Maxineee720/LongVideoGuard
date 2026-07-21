from __future__ import annotations

import pytest

from longvideoguard.training.adapter_evaluation import (
    answer_letter,
    classification_summary,
    cyclic_swap_video_ids,
    delta_from_baseline,
    format_prompt,
    swap_video_summary,
)


def test_answer_letter_supports_multiple_schemas() -> None:
    assert answer_letter({"gold_answer_letter": "B"}) == "B"
    assert answer_letter({"assistant_target": "C"}) == "C"
    assert answer_letter({"answer_index": 3}) == "D"


def test_format_prompt_builds_five_option_prompt() -> None:
    prompt = format_prompt(
        {
            "sample_id": "v1:1",
            "question": "What happened?",
            "options": ["one", "two", "three", "four", "five"],
        }
    )
    assert "Question: What happened?" in prompt
    assert "A. one" in prompt
    assert "E. five" in prompt
    assert prompt.endswith("Answer:")


def test_cyclic_swap_never_maps_video_to_itself() -> None:
    rows = [
        {"video_id": "v1"},
        {"video_id": "v1"},
        {"video_id": "v2"},
        {"video_id": "v3"},
    ]
    mapping = cyclic_swap_video_ids(rows)
    assert set(mapping) == {"v1", "v2", "v3"}
    assert all(source != target for source, target in mapping.items())


def test_classification_and_baseline_delta() -> None:
    predictions = [
        {
            "question_category": "causal",
            "predicted_letter": "A",
            "is_correct": True,
        },
        {
            "question_category": "temporal",
            "predicted_letter": "B",
            "is_correct": False,
        },
    ]
    summary = classification_summary(predictions)
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["valid_rate"] == pytest.approx(1.0)

    delta = delta_from_baseline(
        summary,
        {"accuracy": 0.25},
    )
    assert delta is not None
    assert delta["percentage_point_delta"] == pytest.approx(25.0)


def test_swap_video_summary_flags_unchanged_correct_answers() -> None:
    correct = [
        {
            "sample_id": "v1:1",
            "video_id_used": "v1",
            "gold_answer_letter": "A",
            "predicted_letter": "A",
            "is_correct": True,
        },
        {
            "sample_id": "v2:1",
            "video_id_used": "v2",
            "gold_answer_letter": "B",
            "predicted_letter": "B",
            "is_correct": True,
        },
    ]
    swapped = [
        {
            "sample_id": "v1:1",
            "video_id_used": "v2",
            "gold_answer_letter": "A",
            "predicted_letter": "A",
            "is_correct": True,
        },
        {
            "sample_id": "v2:1",
            "video_id_used": "v1",
            "gold_answer_letter": "B",
            "predicted_letter": "C",
            "is_correct": False,
        },
    ]
    summary = swap_video_summary(correct, swapped)
    assert summary["prediction_change_rate"] == pytest.approx(0.5)
    assert summary["swapped_video_accuracy"] == pytest.approx(0.5)
    assert summary["text_memorisation_warning_count"] == 1
