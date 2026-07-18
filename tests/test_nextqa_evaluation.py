import json
from pathlib import Path

import pytest

from longvideoguard.evaluation.nextqa import (
    error_cases,
    evaluate_nextqa_predictions,
    load_prediction_jsonl,
    wilson_interval,
    write_error_cases,
    write_markdown_report,
    write_metrics,
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "nextqa_predictions_sample.jsonl"
)


def test_load_and_evaluate_predictions() -> None:
    rows = load_prediction_jsonl(FIXTURE)
    metrics = evaluate_nextqa_predictions(rows)

    assert metrics["overall"]["count"] == 5
    assert metrics["overall"]["correct"] == 2
    assert metrics["overall"]["accuracy"] == pytest.approx(0.4)
    assert metrics["overall"]["valid_rate"] == pytest.approx(0.6)
    assert metrics["overall"]["runtime_error_rate"] == pytest.approx(0.2)
    assert metrics["by_question_category"]["causal"]["accuracy"] == pytest.approx(0.5)
    assert metrics["by_question_category"]["temporal"]["accuracy"] == pytest.approx(1.0)
    assert metrics["by_question_category"]["descriptive"]["accuracy"] == pytest.approx(0.0)
    assert metrics["predicted_label_distribution"]["INVALID"] == 2
    assert metrics["confusion_matrix"]["D"]["INVALID"] == 1
    assert metrics["efficiency"]["latency_seconds"]["median"] == pytest.approx(2.5)
    assert metrics["efficiency"]["latency_seconds"]["p95_nearest_rank"] == pytest.approx(4.0)


def test_error_cases_have_failure_tags() -> None:
    rows = load_prediction_jsonl(FIXTURE)
    cases = error_cases(rows)
    assert len(cases) == 3
    tags = [case["failure_tags"] for case in cases]
    assert ["wrong_answer"] in tags
    assert ["invalid_prediction"] in tags
    assert ["runtime_error", "invalid_prediction"] in tags


def test_wilson_interval_is_bounded() -> None:
    interval = wilson_interval(24, 48)
    assert interval is not None
    assert 0 <= interval["lower"] < 0.5
    assert 0.5 < interval["upper"] <= 1


def test_writers_create_reproducible_outputs(tmp_path: Path) -> None:
    rows = load_prediction_jsonl(FIXTURE)
    metrics = evaluate_nextqa_predictions(rows)

    metrics_path = write_metrics(metrics, tmp_path / "metrics.json")
    errors_path = write_error_cases(rows, tmp_path / "errors.jsonl")
    report_path = write_markdown_report(
        metrics,
        tmp_path / "report.md",
        prediction_path=FIXTURE,
    )

    assert json.loads(metrics_path.read_text(encoding="utf-8"))["overall"]["count"] == 5
    assert len(errors_path.read_text(encoding="utf-8").splitlines()) == 3
    assert "Preliminary pilot result" in report_path.read_text(encoding="utf-8")


def test_duplicate_sample_id_is_rejected(tmp_path: Path) -> None:
    first_line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_prediction_jsonl(duplicate)
