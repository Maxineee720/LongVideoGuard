from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROUTER_CATEGORIES = ("causal", "descriptive", "temporal")
CATEGORY_TO_METHOD = {
    "causal": "query_aware",
    "descriptive": "uniform",
    "temporal": "scene_aware",
}


@dataclass(frozen=True)
class RouterDecision:
    predicted_category: str
    selected_method: str
    confidence: float
    matched_rules: tuple[str, ...]
    raw_output: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_category(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "cause": "causal",
        "causality": "causal",
        "reasoning": "causal",
        "description": "descriptive",
        "describe": "descriptive",
        "spatial": "descriptive",
        "time": "temporal",
        "sequence": "temporal",
        "ordering": "temporal",
    }
    text = aliases.get(text, text)
    if text not in ROUTER_CATEGORIES:
        raise ValueError(f"Unsupported question category: {value!r}")
    return text


def gold_category(row: Mapping[str, object]) -> str:
    value = (
        row.get("question_category")
        or row.get("category")
        or row.get("qtype")
    )
    if value is None:
        raise ValueError(
            f"Missing gold question category for {row.get('sample_id')!r}."
        )
    return normalize_category(value)


def _score_patterns(
    text: str,
    patterns: Sequence[tuple[str, float, str]],
) -> tuple[float, list[str]]:
    score = 0.0
    matched: list[str] = []
    for pattern, weight, label in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            score += weight
            matched.append(label)
    return score, matched


_CAUSAL_PATTERNS = (
    (r"^\s*why\b", 4.0, "starts_with_why"),
    (r"\bwhat\s+(?:is|was)\s+the\s+reason\b", 4.0, "asks_reason"),
    (r"\b(?:reason|cause|caused|causing)\b", 2.5, "causal_keyword"),
    (r"\b(?:because|therefore|so that|in order to)\b", 2.5, "purpose_or_reason"),
    (r"\bwhat\s+(?:made|caused)\b", 3.0, "asks_cause"),
    (r"\b(?:purpose|motivation|motivated)\b", 2.5, "purpose_keyword"),
)

_TEMPORAL_PATTERNS = (
    (r"\b(?:before|after)\b", 3.0, "before_after"),
    (r"\bwhat\s+happens?\s+next\b", 4.0, "asks_next"),
    (r"\bwhat\s+(?:did|does|happened)\b.*\bnext\b", 3.5, "next_action"),
    (r"\b(?:first|last|finally|initially|subsequently)\b", 2.5, "order_keyword"),
    (r"\b(?:then|prior to|following|earlier|later)\b", 2.0, "relative_time"),
    (r"\b(?:sequence|order)\b", 3.0, "sequence_keyword"),
    (r"\bwhat\s+happened\s+when\b", 1.5, "event_time_relation"),
)

_DESCRIPTIVE_PATTERNS = (
    (r"^\s*(?:who|where|which)\b", 2.5, "entity_or_location"),
    (r"^\s*how\s+many\b", 3.0, "counting"),
    (r"^\s*what\s+(?:is|are|was|were)\b", 1.5, "state_or_identity"),
    (r"\b(?:color|colour|wearing|holding|object|location|place)\b", 1.5, "visual_attribute"),
    (r"\b(?:look like|appearance|visible)\b", 1.5, "appearance"),
)


def rule_based_decision(question: str) -> RouterDecision:
    text = str(question).strip()
    if not text:
        raise ValueError("Question must be non-empty.")

    causal_score, causal_rules = _score_patterns(text, _CAUSAL_PATTERNS)
    temporal_score, temporal_rules = _score_patterns(text, _TEMPORAL_PATTERNS)
    descriptive_score, descriptive_rules = _score_patterns(
        text,
        _DESCRIPTIVE_PATTERNS,
    )

    scores = {
        "causal": causal_score,
        "temporal": temporal_score,
        "descriptive": descriptive_score,
    }
    matched_by_category = {
        "causal": causal_rules,
        "temporal": temporal_rules,
        "descriptive": descriptive_rules,
    }

    # Prefer explicit causal/temporal cues over the descriptive fallback.
    best_score = max(scores.values())
    if best_score <= 0:
        predicted = "descriptive"
        confidence = 0.45
        matched = ("default_descriptive",)
    else:
        priority = ("causal", "temporal", "descriptive")
        predicted = max(
            priority,
            key=lambda category: (
                scores[category],
                -priority.index(category),
            ),
        )
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = best_score - sorted_scores[1]
        confidence = min(
            0.99,
            0.55 + 0.08 * best_score + 0.05 * max(margin, 0.0),
        )
        matched = tuple(matched_by_category[predicted])

    return RouterDecision(
        predicted_category=predicted,
        selected_method=CATEGORY_TO_METHOD[predicted],
        confidence=confidence,
        matched_rules=matched,
    )


def build_qwen_router_prompt(question: str) -> str:
    text = str(question).strip()
    if not text:
        raise ValueError("Question must be non-empty.")

    return (
        "Classify the VideoQA question into exactly one category.\n"
        "causal: asks why something happened, its reason, cause, or purpose.\n"
        "temporal: asks about before/after, next/previous events, or event order.\n"
        "descriptive: asks about people, objects, attributes, counts, places, "
        "or actions without requiring causal or temporal ordering.\n\n"
        f"Question: {text}\n\n"
        "Return exactly one lowercase word: causal, temporal, or descriptive."
    )


def parse_qwen_router_output(raw_output: str) -> str | None:
    text = str(raw_output).strip().lower()
    if text in ROUTER_CATEGORIES:
        return text

    matches = re.findall(
        r"(?<![a-z])(causal|temporal|descriptive)(?![a-z])",
        text,
    )
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def qwen_decision_from_output(
    raw_output: str,
    *,
    fallback_question: str,
) -> RouterDecision:
    category = parse_qwen_router_output(raw_output)
    if category is not None:
        return RouterDecision(
            predicted_category=category,
            selected_method=CATEGORY_TO_METHOD[category],
            confidence=0.9,
            matched_rules=("qwen_zero_shot",),
            raw_output=str(raw_output),
        )

    fallback = rule_based_decision(fallback_question)
    return RouterDecision(
        predicted_category=fallback.predicted_category,
        selected_method=fallback.selected_method,
        confidence=min(fallback.confidence, 0.5),
        matched_rules=("qwen_parse_failed",) + fallback.matched_rules,
        raw_output=str(raw_output),
    )


def confusion_matrix(
    gold_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Gold and predicted labels must have equal length.")

    matrix = {
        gold: {predicted: 0 for predicted in ROUTER_CATEGORIES}
        for gold in ROUTER_CATEGORIES
    }
    for gold, predicted in zip(
        gold_labels,
        predicted_labels,
        strict=True,
    ):
        matrix[normalize_category(gold)][normalize_category(predicted)] += 1
    return matrix


def classification_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("Router rows must be non-empty.")

    gold_labels = [normalize_category(row["gold_category"]) for row in rows]
    predicted_labels = [
        normalize_category(row["predicted_category"]) for row in rows
    ]
    correct = sum(
        gold == predicted
        for gold, predicted in zip(
            gold_labels,
            predicted_labels,
            strict=True,
        )
    )

    by_category: dict[str, dict[str, object]] = {}
    for category in ROUTER_CATEGORIES:
        indices = [
            index
            for index, gold in enumerate(gold_labels)
            if gold == category
        ]
        category_correct = sum(
            predicted_labels[index] == category
            for index in indices
        )
        by_category[category] = {
            "count": len(indices),
            "correct": category_correct,
            "accuracy": (
                category_correct / len(indices)
                if indices
                else None
            ),
        }

    route_counts = Counter(
        str(row["selected_method"]) for row in rows
    )

    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "by_gold_category": by_category,
        "confusion_matrix": confusion_matrix(
            gold_labels,
            predicted_labels,
        ),
        "selected_method_distribution": dict(sorted(route_counts.items())),
        "mean_confidence": sum(
            float(row["confidence"]) for row in rows
        ) / len(rows),
    }


def route_existing_predictions(
    decisions: Sequence[Mapping[str, object]],
    predictions_by_method: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
    *,
    router_name: str,
) -> list[dict[str, object]]:
    prediction_indices = {
        method: {
            str(row["sample_id"]): row for row in predictions
        }
        for method, predictions in predictions_by_method.items()
    }

    routed: list[dict[str, object]] = []
    for decision in decisions:
        sample_id = str(decision["sample_id"])
        method = str(decision["selected_method"])
        if method not in prediction_indices:
            raise ValueError(f"Unknown selected method: {method!r}")
        if sample_id not in prediction_indices[method]:
            raise ValueError(
                f"No {method} prediction for sample {sample_id!r}."
            )

        selected = dict(prediction_indices[method][sample_id])
        selected.update(
            {
                "router_name": router_name,
                "router_predicted_category": decision[
                    "predicted_category"
                ],
                "router_gold_category": decision["gold_category"],
                "router_selected_method": method,
                "router_confidence": decision["confidence"],
                "router_matched_rules": decision["matched_rules"],
                "router_raw_output": decision.get("raw_output"),
            }
        )
        routed.append(selected)
    return routed


def oracle_decisions(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for row in rows:
        category = gold_category(row)
        decisions.append(
            {
                "sample_id": str(row["sample_id"]),
                "gold_category": category,
                "predicted_category": category,
                "selected_method": CATEGORY_TO_METHOD[category],
                "confidence": 1.0,
                "matched_rules": ["gold_category_oracle"],
                "raw_output": None,
            }
        )
    return decisions


def write_jsonl(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return destination


def write_json(
    payload: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
