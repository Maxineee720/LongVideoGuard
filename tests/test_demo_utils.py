from __future__ import annotations

import pytest

from longvideoguard.demo.app_utils import (
    build_multiple_choice_prompt,
    normalize_options,
    parse_answer,
)


def test_normalize_options() -> None:
    assert normalize_options(
        ["a", "b", "c", "d", "e"]
    ) == ["a", "b", "c", "d", "e"]

    with pytest.raises(ValueError):
        normalize_options(["a", "b"])


def test_prompt() -> None:
    prompt = build_multiple_choice_prompt(
        "What happens next?",
        ["a", "b", "c", "d", "e"],
    )
    assert "Question: What happens next?" in prompt
    assert "A. a" in prompt
    assert "E. e" in prompt


def test_parse_answer() -> None:
    assert parse_answer("D") == "D"
    assert parse_answer("Answer: B") == "B"
    assert parse_answer("unknown") is None
