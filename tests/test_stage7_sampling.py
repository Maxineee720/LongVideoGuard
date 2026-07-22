from __future__ import annotations

import pytest

from longvideoguard.evaluation.stage7_sampling import (
    exact_mcnemar_p_value,
    format_prompt,
    method_summary,
    paired_comparison,
    parse_answer,
)


def test_parse_answer() -> None:
    assert parse_answer("D") == "D"
    assert parse_answer("Answer: B") == "B"
    assert parse_answer("invalid") is None


def test_prompt_contains_question_and_options() -> None:
    prompt = format_prompt(
        {
            "sample_id": "x",
            "question": "What happened?",
            "options": ["a", "b", "c", "d", "e"],
        }
    )
    assert "Question: What happened?" in prompt
    assert "A. a" in prompt
    assert "E. e" in prompt


def test_method_summary() -> None:
    rows = [
        {
            "is_correct": True,
            "prediction": "A",
            "question_category": "causal",
            "latency_seconds": 0.2,
            "input_token_count": 100,
            "generated_token_count": 1,
            "peak_gpu_memory_mb": 1000,
        },
        {
            "is_correct": False,
            "prediction": None,
            "question_category": "temporal",
            "latency_seconds": 0.4,
            "input_token_count": 110,
            "generated_token_count": 2,
            "peak_gpu_memory_mb": 1100,
        },
    ]
    summary = method_summary(rows)
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["valid_rate"] == pytest.approx(0.5)
    assert summary["latency_seconds"]["mean"] == pytest.approx(0.3)


def test_exact_mcnemar() -> None:
    assert exact_mcnemar_p_value(0, 0) == pytest.approx(1.0)
    assert exact_mcnemar_p_value(0, 5) == pytest.approx(0.0625)


def test_paired_comparison() -> None:
    uniform = [
        {"sample_id": "a", "is_correct": True, "prediction": "A"},
        {"sample_id": "b", "is_correct": False, "prediction": "B"},
        {"sample_id": "c", "is_correct": False, "prediction": "C"},
    ]
    query = [
        {"sample_id": "a", "is_correct": True, "prediction": "A"},
        {"sample_id": "b", "is_correct": True, "prediction": "D"},
        {"sample_id": "c", "is_correct": False, "prediction": "C"},
    ]
    result = paired_comparison(uniform, query)
    assert result["candidate_only_correct"] == 1
    assert result["baseline_only_correct"] == 0
    assert result["percentage_point_delta"] == pytest.approx(100 / 3)
