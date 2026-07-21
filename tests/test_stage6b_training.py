from __future__ import annotations

import pytest

from longvideoguard.training.stage6b_training import (
    build_positive_eval_rows,
    checkpoint_key,
    evaluation_summary,
    is_better_checkpoint,
    parse_structured_output,
    prediction_is_correct,
)


def test_structured_output_parser() -> None:
    answerable = parse_structured_output(
        '{"status":"answerable","answer":"D"}'
    )
    assert answerable["valid_structure"]
    assert answerable["exact_canonical_format"]
    assert answerable["predicted_status"] == "answerable"
    assert answerable["predicted_answer_letter"] == "D"

    wrapped = parse_structured_output(
        'Result: {"status":"unanswerable","answer":null}'
    )
    assert wrapped["valid_structure"]
    assert not wrapped["exact_canonical_format"]
    assert wrapped["predicted_status"] == "unanswerable"

    invalid = parse_structured_output(
        '{"status":"answerable","answer":null}'
    )
    assert not invalid["valid_structure"]


def test_prediction_correctness() -> None:
    parsed = parse_structured_output(
        '{"status":"answerable","answer":"B"}'
    )
    assert prediction_is_correct(
        gold_status="answerable",
        gold_answer_letter="B",
        parsed=parsed,
    )
    assert not prediction_is_correct(
        gold_status="answerable",
        gold_answer_letter="A",
        parsed=parsed,
    )


def test_mixed_evaluation_metrics() -> None:
    predictions = [
        {
            "gold_status": "answerable",
            "question_category": "causal",
            "predicted_status": "answerable",
            "predicted_answer_letter": "A",
            "valid_structure": True,
            "exact_canonical_format": True,
            "is_correct": True,
        },
        {
            "gold_status": "answerable",
            "question_category": "temporal",
            "predicted_status": "unanswerable",
            "predicted_answer_letter": None,
            "valid_structure": True,
            "exact_canonical_format": True,
            "is_correct": False,
        },
        {
            "gold_status": "unanswerable",
            "question_category": "causal",
            "predicted_status": "unanswerable",
            "predicted_answer_letter": None,
            "valid_structure": True,
            "exact_canonical_format": True,
            "is_correct": True,
        },
        {
            "gold_status": "unanswerable",
            "question_category": "temporal",
            "predicted_status": "answerable",
            "predicted_answer_letter": "C",
            "valid_structure": True,
            "exact_canonical_format": True,
            "is_correct": False,
        },
    ]
    summary = evaluation_summary(predictions)
    assert summary["answerable_exact_accuracy"] == pytest.approx(0.5)
    assert summary["unanswerable_recall"] == pytest.approx(0.5)
    assert summary["balanced_task_score"] == pytest.approx(0.5)
    assert summary["false_refusal_rate"] == pytest.approx(0.5)
    assert summary["false_answer_rate"] == pytest.approx(0.5)


def test_checkpoint_selection_includes_balanced_score() -> None:
    base = {
        "balanced_task_score": 0.60,
        "answerable_exact_accuracy": 0.80,
        "unanswerable_recall": 0.40,
        "overall_exact_accuracy": 0.66,
    }
    candidate = {
        "balanced_task_score": 0.70,
        "answerable_exact_accuracy": 0.70,
        "unanswerable_recall": 0.70,
        "overall_exact_accuracy": 0.70,
    }
    assert is_better_checkpoint(
        candidate,
        candidate_loss=1.2,
        best_metrics=base,
        best_loss=0.8,
    )
    assert checkpoint_key(
        candidate,
        teacher_forced_loss=1.2,
    )[0] == pytest.approx(0.70)


def test_positive_eval_conversion() -> None:
    rows = [
        {
            "sample_id": "v1:0",
            "video_id": "v1",
            "video_relpath": "v1.mp4",
            "question": "What happened?",
            "options": ["a", "b", "c", "d", "e"],
            "answer_index": 3,
            "question_category": "causal",
        }
    ]
    converted = build_positive_eval_rows(
        rows,
        role="development_positive",
    )
    assert converted[0]["gold_status"] == "answerable"
    assert converted[0]["gold_answer_letter"] == "D"
    assert (
        converted[0]["assistant_target"]
        == '{"status":"answerable","answer":"D"}'
    )


def test_positive_only_development_summary_is_supported() -> None:
    predictions = [
        {
            "gold_status": "answerable",
            "question_category": "causal",
            "predicted_status": "answerable",
            "predicted_answer_letter": "A",
            "valid_structure": True,
            "exact_canonical_format": True,
            "is_correct": True,
        }
    ]
    summary = evaluation_summary(predictions)
    assert summary["answerable_exact_accuracy"] == pytest.approx(1.0)
    assert summary["unanswerable_recall"] is None
    assert summary["balanced_task_score"] is None
    assert summary["false_answer_rate"] is None
