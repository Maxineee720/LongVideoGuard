from __future__ import annotations

from dataclasses import dataclass

import pytest

from longvideoguard.training.lora_overfit import (
    count_parameters,
    finite_mean,
    optimizer_step_sample_indices,
    parameter_delta_summary,
)


def test_optimizer_schedule_is_deterministic_and_bounded() -> None:
    first = optimizer_step_sample_indices(
        num_samples=5,
        optimizer_steps=4,
        gradient_accumulation_steps=3,
        seed=42,
    )
    second = optimizer_step_sample_indices(
        num_samples=5,
        optimizer_steps=4,
        gradient_accumulation_steps=3,
        seed=42,
    )
    assert first == second
    assert len(first) == 4
    assert all(len(step) == 3 for step in first)
    assert all(0 <= index < 5 for step in first for index in step)


def test_finite_mean_rejects_non_finite_values() -> None:
    assert finite_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="Non-finite"):
        finite_mean([1.0, float("nan")])


def test_count_parameters_and_delta() -> None:
    torch = pytest.importorskip("torch")

    model = torch.nn.Linear(3, 2, bias=False)
    for parameter in model.parameters():
        parameter.requires_grad = True

    counts = count_parameters(model)
    assert counts["total_parameters"] == 6
    assert counts["trainable_parameters"] == 6
    assert counts["trainable_percentage"] == pytest.approx(100.0)

    before = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        model.weight.add_(1.0)

    delta = parameter_delta_summary(model, before)
    assert delta["compared_tensors"] == 1
    assert delta["changed_tensors"] == 1
    assert delta["delta_l2"] > 0


def test_parameter_delta_rejects_name_changes() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, bias=False)
    before = {"wrong_name": model.weight.detach().clone()}

    with pytest.raises(ValueError, match="names changed"):
        parameter_delta_summary(model, before)
