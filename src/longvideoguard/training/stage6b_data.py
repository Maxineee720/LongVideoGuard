from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

OPTION_LABELS = ("A", "B", "C", "D", "E")
ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load a non-empty JSONL manifest with unique sample IDs."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSONL file not found: {source}")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} of {source} must be a JSON object."
                )

            required = {
                "sample_id",
                "video_id",
                "video_relpath",
                "question",
                "options",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {missing}"
                )

            sample_id = str(row["sample_id"]).strip()
            if not sample_id:
                raise ValueError(f"Line {line_number}: empty sample_id.")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample_id!r}")
            seen.add(sample_id)

            options = row["options"]
            if not isinstance(options, list) or len(options) != 5:
                raise ValueError(
                    f"Line {line_number}: expected exactly five options."
                )

            _ = answer_letter(row)
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {source}.")
    return rows


def answer_letter(row: Mapping[str, object]) -> str:
    """Resolve a gold A-E answer from the Stage 6A schema."""
    for field in (
        "answer_letter",
        "gold_answer_letter",
        "assistant_target",
    ):
        value = row.get(field)
        if value is None:
            continue
        letter = str(value).strip().upper()
        if letter in OPTION_LABELS:
            return letter

    value = row.get("answer_index")
    if value is not None:
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid answer_index for {row.get('sample_id')}: {value!r}"
            ) from exc
        if index in range(5):
            return OPTION_LABELS[index]

    raise ValueError(
        f"Could not resolve answer letter for {row.get('sample_id')!r}."
    )


def question_category(row: Mapping[str, object]) -> str:
    return str(
        row.get("question_category")
        or row.get("category")
        or "unknown"
    )


def structured_target(
    *,
    status: str,
    answer: str | None,
) -> str:
    """Return the exact compact JSON target used for Stage 6B supervision."""
    if status == ANSWERABLE:
        if answer not in OPTION_LABELS:
            raise ValueError("Answerable targets require an A-E answer.")
    elif status == UNANSWERABLE:
        if answer is not None:
            raise ValueError("Unanswerable targets must use answer=None.")
    else:
        raise ValueError(f"Unsupported status: {status!r}")

    return json.dumps(
        {
            "status": status,
            "answer": answer,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def answerability_prompt(row: Mapping[str, object]) -> str:
    """Build the shared structured-output prompt for positive and negative rows."""
    question = str(row["question"]).strip()
    options = [str(value).strip() for value in row["options"]]
    if not question:
        raise ValueError(f"Empty question in {row.get('sample_id')!r}.")
    if len(options) != 5:
        raise ValueError("Exactly five options are required.")

    option_block = "\n".join(
        f"{label}. {option}"
        for label, option in zip(OPTION_LABELS, options, strict=True)
    )

    return (
        "Watch the current video carefully and inspect the question and options.\n"
        "First decide whether this video contains sufficient visual evidence "
        "to answer the question.\n"
        "Return exactly one compact JSON object and nothing else.\n"
        "When the video is sufficient, use "
        '{"status":"answerable","answer":"A"} '
        "with the correct letter A, B, C, D, or E.\n"
        "When the video is unrelated or lacks the required evidence, use "
        '{"status":"unanswerable","answer":null}.\n\n'
        f"Question: {question}\n\n"
        f"Options:\n{option_block}\n\n"
        "Output:"
    )


def _base_payload(
    row: Mapping[str, object],
    *,
    split_role: str,
    sample_id: str,
    video_id: str,
    video_relpath: str,
    status: str,
    answer: str | None,
) -> dict[str, object]:
    target = structured_target(status=status, answer=answer)
    prompt = answerability_prompt(row)

    payload = dict(row)
    payload.update(
        {
            "schema_version": "2.0",
            "task": "videoqa_answerability",
            "stage6b_role": split_role,
            "sample_id": sample_id,
            "video_id": video_id,
            "video_relpath": video_relpath,
            "video": video_relpath,
            "is_answerable": status == ANSWERABLE,
            "gold_status": status,
            "gold_answer_letter": answer,
            "prompt": prompt,
            "assistant_target": target,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<video>\n{prompt}",
                },
                {
                    "from": "gpt",
                    "value": target,
                },
            ],
        }
    )
    return payload


def build_positive_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    split_name: str,
) -> list[dict[str, object]]:
    """Convert normal QA rows into structured answerable examples."""
    positives: list[dict[str, object]] = []

    for row in rows:
        original_sample_id = str(row["sample_id"])
        letter = answer_letter(row)
        payload = _base_payload(
            row,
            split_role=f"{split_name}_positive",
            sample_id=f"pos::{original_sample_id}",
            video_id=str(row["video_id"]),
            video_relpath=str(row["video_relpath"]),
            status=ANSWERABLE,
            answer=letter,
        )
        payload.update(
            {
                "source_question_sample_id": original_sample_id,
                "source_question_video_id": str(row["video_id"]),
                "original_answer_letter": letter,
                "negative_pairing_strategy": None,
            }
        )
        positives.append(payload)

    return positives


def _stratified_source_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    count: int,
    seed: int,
) -> list[Mapping[str, object]]:
    """Select negative-source questions balanced over category and answer letter."""
    if count <= 0:
        raise ValueError("Negative count must be positive.")
    if count > len(rows):
        raise ValueError(
            f"Requested {count} negatives from only {len(rows)} questions."
        )

    buckets: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(question_category(row), answer_letter(row))].append(row)

    for key, bucket in buckets.items():
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(bucket)

    keys = sorted(buckets)
    selected: list[Mapping[str, object]] = []
    cursor = 0

    while len(selected) < count:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop())
        cursor += 1

        if cursor % len(keys) == 0 and not any(buckets.values()):
            break

    if len(selected) != count:
        raise RuntimeError(
            f"Could select only {len(selected)} of {count} negative sources."
        )

    return sorted(selected, key=lambda row: str(row["sample_id"]))


def _video_path_map(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        video_id = str(row["video_id"])
        relpath = str(row["video_relpath"])
        previous = mapping.get(video_id)
        if previous is not None and previous != relpath:
            raise ValueError(
                f"Video {video_id!r} maps to both {previous!r} and {relpath!r}."
            )
        mapping[video_id] = relpath
    return mapping


def _replacement_video_id(
    row: Mapping[str, object],
    *,
    all_rows: Sequence[Mapping[str, object]],
    seed: int,
) -> tuple[str, str]:
    """
    Prefer a different video associated with the same question category.

    The same-category preference makes the mismatch less trivial while the
    strict cross-video requirement prevents direct positive leakage.
    """
    source_video_id = str(row["video_id"])
    category = question_category(row)

    same_category = sorted(
        {
            str(candidate["video_id"])
            for candidate in all_rows
            if question_category(candidate) == category
            and str(candidate["video_id"]) != source_video_id
        }
    )
    candidates = same_category

    strategy = "same_category_different_video"
    if not candidates:
        candidates = sorted(
            {
                str(candidate["video_id"])
                for candidate in all_rows
                if str(candidate["video_id"]) != source_video_id
            }
        )
        strategy = "fallback_any_different_video"

    if not candidates:
        raise ValueError("At least two distinct videos are required.")

    rng = random.Random(f"{seed}:{row['sample_id']}")
    return rng.choice(candidates), strategy


def build_negative_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    split_name: str,
    negative_count: int,
    seed: int,
) -> list[dict[str, object]]:
    """Create paired mismatched-video examples with unanswerable targets."""
    source_rows = _stratified_source_rows(
        rows,
        count=negative_count,
        seed=seed,
    )
    path_by_video = _video_path_map(rows)
    negatives: list[dict[str, object]] = []

    for row in source_rows:
        source_sample_id = str(row["sample_id"])
        source_video_id = str(row["video_id"])
        replacement_video_id, strategy = _replacement_video_id(
            row,
            all_rows=rows,
            seed=seed,
        )
        if replacement_video_id == source_video_id:
            raise AssertionError("A negative pair reused the source video.")

        replacement_relpath = path_by_video[replacement_video_id]
        sample_id = (
            f"neg::{source_sample_id}::video::{replacement_video_id}"
        )
        payload = _base_payload(
            row,
            split_role=f"{split_name}_negative",
            sample_id=sample_id,
            video_id=replacement_video_id,
            video_relpath=replacement_relpath,
            status=UNANSWERABLE,
            answer=None,
        )
        payload.update(
            {
                "source_question_sample_id": source_sample_id,
                "source_question_video_id": source_video_id,
                "negative_video_id": replacement_video_id,
                "paired_positive_sample_id": f"pos::{source_sample_id}",
                "original_answer_letter": answer_letter(row),
                "negative_pairing_strategy": strategy,
                "negative_audit_status": "pending",
            }
        )
        negatives.append(payload)

    return negatives


def deterministic_mix(
    positives: Sequence[Mapping[str, object]],
    negatives: Sequence[Mapping[str, object]],
    *,
    seed: int,
) -> list[dict[str, object]]:
    """Shuffle positive and negative rows reproducibly."""
    rows = [dict(row) for row in (*positives, *negatives)]
    random.Random(seed).shuffle(rows)
    return rows


def deterministic_audit_sample(
    negatives: Sequence[Mapping[str, object]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    """Select a balanced manual-audit subset of candidate hard negatives."""
    if count <= 0:
        raise ValueError("Audit count must be positive.")
    count = min(count, len(negatives))
    selected = _stratified_source_rows(
        negatives,
        count=count,
        seed=seed,
    )
    return [dict(row) for row in selected]


def split_video_ids(rows: Sequence[Mapping[str, object]]) -> set[str]:
    """Return all actual videos consumed by a Stage 6B split."""
    return {str(row["video_id"]) for row in rows}


def assert_cross_split_video_disjoint(
    named_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, list[str]]:
    """Reject actual-video overlap across train, holdout, and frozen splits."""
    names = sorted(named_rows)
    checks: dict[str, list[str]] = {}

    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = sorted(
                split_video_ids(named_rows[left_name])
                & split_video_ids(named_rows[right_name])
            )
            checks[f"{left_name}__{right_name}__video_ids"] = overlap
            if overlap:
                raise ValueError(
                    f"Video leakage between {left_name} and {right_name}: "
                    f"{overlap[:10]}"
                )
    return checks


def validate_negative_pairs(
    negatives: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    invalid_same_video: list[str] = []
    invalid_target: list[str] = []

    expected_target = structured_target(
        status=UNANSWERABLE,
        answer=None,
    )
    for row in negatives:
        if str(row["video_id"]) == str(row["source_question_video_id"]):
            invalid_same_video.append(str(row["sample_id"]))
        if row.get("assistant_target") != expected_target:
            invalid_target.append(str(row["sample_id"]))

    if invalid_same_video or invalid_target:
        raise ValueError(
            "Invalid negatives: "
            f"same_video={invalid_same_video[:5]}, "
            f"wrong_target={invalid_target[:5]}"
        )

    return {
        "count": len(negatives),
        "all_cross_video": True,
        "all_targets_unanswerable": True,
    }


def split_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    status_counts = Counter(str(row["gold_status"]) for row in rows)
    category_counts = Counter(question_category(row) for row in rows)
    answer_counts = Counter(
        str(row["gold_answer_letter"])
        for row in rows
        if row.get("gold_answer_letter") in OPTION_LABELS
    )
    strategy_counts = Counter(
        str(row["negative_pairing_strategy"])
        for row in rows
        if row.get("negative_pairing_strategy")
    )

    return {
        "num_examples": len(rows),
        "num_actual_videos": len(split_video_ids(rows)),
        "status_counts": dict(sorted(status_counts.items())),
        "question_category_counts": dict(sorted(category_counts.items())),
        "answer_letter_counts_for_positives": {
            label: answer_counts.get(label, 0)
            for label in OPTION_LABELS
        },
        "negative_pairing_strategy_counts": dict(
            sorted(strategy_counts.items())
        ),
    }


def write_jsonl(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("Refusing to write an empty JSONL file.")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination


def write_qwen_json(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("Refusing to write an empty Qwen JSON file.")

    payload = [
        {
            "video": row["video_relpath"],
            "conversations": row["conversations"],
        }
        for row in materialized
    ]
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
