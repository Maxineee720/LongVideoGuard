from __future__ import annotations

import pytest

from longvideoguard.routing.question_router import (
    classification_summary,
    parse_qwen_router_output,
    qwen_decision_from_output,
    route_existing_predictions,
    rule_based_decision,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Why did the man leave the room?", "causal"),
        ("What happened after the woman sat down?", "temporal"),
        ("What color is the car?", "descriptive"),
        ("Who is holding the cup?", "descriptive"),
        ("What happens next?", "temporal"),
    ],
)
def test_rule_based_router(question: str, expected: str) -> None:
    decision = rule_based_decision(question)
    assert decision.predicted_category == expected


def test_parse_qwen_output() -> None:
    assert parse_qwen_router_output("causal") == "causal"
    assert parse_qwen_router_output("Category: temporal") == "temporal"
    assert parse_qwen_router_output("causal or temporal") is None


def test_qwen_parse_failure_falls_back() -> None:
    decision = qwen_decision_from_output(
        "I am unsure.",
        fallback_question="What happens after the door opens?",
    )
    assert decision.predicted_category == "temporal"
    assert "qwen_parse_failed" in decision.matched_rules


def test_classification_summary() -> None:
    rows = [
        {
            "gold_category": "causal",
            "predicted_category": "causal",
            "selected_method": "query_aware",
            "confidence": 0.9,
        },
        {
            "gold_category": "temporal",
            "predicted_category": "descriptive",
            "selected_method": "uniform",
            "confidence": 0.5,
        },
    ]
    summary = classification_summary(rows)
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["confusion_matrix"]["temporal"]["descriptive"] == 1


def test_route_existing_predictions() -> None:
    decisions = [
        {
            "sample_id": "s1",
            "gold_category": "causal",
            "predicted_category": "causal",
            "selected_method": "query_aware",
            "confidence": 0.9,
            "matched_rules": ["test"],
            "raw_output": None,
        }
    ]
    predictions = {
        "uniform": [
            {"sample_id": "s1", "prediction": "A", "is_correct": False}
        ],
        "scene_aware": [
            {"sample_id": "s1", "prediction": "B", "is_correct": False}
        ],
        "query_aware": [
            {"sample_id": "s1", "prediction": "C", "is_correct": True}
        ],
    }
    routed = route_existing_predictions(
        decisions,
        predictions,
        router_name="test_router",
    )
    assert routed[0]["prediction"] == "C"
    assert routed[0]["router_selected_method"] == "query_aware"
