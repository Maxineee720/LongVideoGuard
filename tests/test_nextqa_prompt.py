import pytest

from longvideoguard.nextqa_prompt import (
    answer_index_to_letter,
    format_nextqa_prompt,
    parse_option_letter,
    validate_manifest_row,
)


OPTIONS = ["one", "two", "three", "four", "five"]


def test_prompt_has_all_options_and_strict_instruction() -> None:
    prompt = format_nextqa_prompt("What happened?", OPTIONS)
    assert "A. one" in prompt
    assert "E. five" in prompt
    assert "Return exactly one uppercase letter" in prompt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C", "C"),
        ("Answer: B", "B"),
        ("final answer - E", "E"),
        ("(A)", "A"),
        ("D. ", "D"),
    ],
)
def test_parse_option_letter(raw: str, expected: str) -> None:
    assert parse_option_letter(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "I do not know", "A or B", "Answer: A, maybe B"],
)
def test_parse_option_letter_rejects_invalid_or_ambiguous(raw: str) -> None:
    assert parse_option_letter(raw) is None


def test_answer_index_to_letter() -> None:
    assert answer_index_to_letter(0) == "A"
    assert answer_index_to_letter(4) == "E"
    with pytest.raises(ValueError):
        answer_index_to_letter(5)


def test_validate_manifest_row() -> None:
    validate_manifest_row(
        {
            "sample_id": "v1:1",
            "video_id": "v1",
            "video_relpath": "v1.mp4",
            "question": "What happened?",
            "options": OPTIONS,
            "answer_index": 2,
            "question_category": "temporal",
        }
    )
