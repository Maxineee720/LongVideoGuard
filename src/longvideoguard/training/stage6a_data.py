from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

OPTION_LABELS = ("A", "B", "C", "D", "E")


def normalize_video_filename(
    mapped_video_id: object,
    *,
    extension: str = ".mp4",
) -> str:
    """Return a portable filename for a mapped NExT-QA video identifier."""
    if not extension.startswith("."):
        raise ValueError("extension must start with '.'")

    raw = str(mapped_video_id).strip()
    if not raw:
        raise ValueError("mapped_video_id must be non-empty")

    filename = PurePosixPath(raw.replace("\\", "/")).name
    if not filename:
        raise ValueError(f"Could not derive a filename from {raw!r}")
    if PurePosixPath(filename).suffix:
        return filename
    return f"{filename}{extension}"


def load_manifest_rows(
    paths: Sequence[str | Path],
) -> list[dict[str, object]]:
    """Load manifest rows needed for logical and physical leakage checks."""
    rows: list[dict[str, object]] = []

    for path in paths:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Manifest not found: {source}")

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

                video_id = str(row.get("video_id", "")).strip()
                if not video_id:
                    raise ValueError(
                        f"Line {line_number} of {source} has no video_id."
                    )

                video_relpath = (
                    row.get("video_relpath")
                    or row.get("video")
                )
                if video_relpath is None or not str(video_relpath).strip():
                    raise ValueError(
                        f"Line {line_number} of {source} has no "
                        "video_relpath/video field."
                    )

                rows.append(
                    {
                        "video_id": video_id,
                        "video_relpath": str(video_relpath).strip(),
                    }
                )

    if not rows:
        raise ValueError("No manifest rows were loaded.")
    return rows


def load_manifest_video_ids(paths: Sequence[str | Path]) -> set[str]:
    """Load all logical video IDs from one or more JSONL manifests."""
    return {
        str(row["video_id"])
        for row in load_manifest_rows(paths)
    }


def _sample_id(record: object) -> str:
    value = getattr(record, "sample_id", None)
    if value is not None and str(value).strip():
        return str(value)
    return f"{getattr(record, 'video_id')}:{getattr(record, 'qid')}"


def _category(record: object) -> str:
    value = getattr(record, "category", None)
    if value is None:
        value = getattr(record, "question_category", None)
    return str(value or "unknown")


def _question_type(record: object) -> str:
    return str(getattr(record, "question_type", "unknown"))


def _group_by_video(
    records: Iterable[object],
    *,
    excluded_video_ids: set[str],
) -> dict[str, list[object]]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for record in records:
        video_id = str(getattr(record, "video_id"))
        if video_id in excluded_video_ids:
            continue
        grouped[video_id].append(record)
    return dict(grouped)


def _select_video_ids(
    grouped: Mapping[str, Sequence[object]],
    *,
    num_videos: int,
    seed: int,
) -> list[str]:
    """Greedily favour videos covering underrepresented question categories."""
    if num_videos <= 0:
        raise ValueError("num_videos must be positive")
    if num_videos > len(grouped):
        raise ValueError(
            f"Requested {num_videos} videos, but only {len(grouped)} are available."
        )

    rng = random.Random(seed)
    tie_break = {video_id: rng.random() for video_id in grouped}
    remaining = set(grouped)
    selected: list[str] = []
    selected_categories: Counter[str] = Counter()

    while len(selected) < num_videos:
        scored: list[tuple[float, float, str]] = []

        for video_id in remaining:
            category_counts = Counter(
                _category(record) for record in grouped[video_id]
            )
            balance_score = sum(
                count / (1 + selected_categories[category])
                for category, count in category_counts.items()
            )
            diversity_bonus = 0.05 * len(category_counts)
            scored.append(
                (
                    balance_score + diversity_bonus,
                    tie_break[video_id],
                    video_id,
                )
            )

        _, _, chosen = max(scored)
        selected.append(chosen)
        remaining.remove(chosen)
        selected_categories.update(
            _category(record) for record in grouped[chosen]
        )

    return selected


def _sample_questions(
    records: Sequence[object],
    *,
    max_questions: int,
    seed: int,
    video_id: str,
) -> list[object]:
    """Select questions while favouring category diversity inside each video."""
    if max_questions <= 0:
        raise ValueError("max_questions must be positive")

    ordered = sorted(records, key=_sample_id)
    if len(ordered) <= max_questions:
        return ordered

    rng = random.Random(f"{seed}:{video_id}")
    pool = ordered[:]
    rng.shuffle(pool)

    selected: list[object] = []
    category_counts: Counter[str] = Counter()

    while pool and len(selected) < max_questions:
        index = max(
            range(len(pool)),
            key=lambda item: (
                1 / (1 + category_counts[_category(pool[item])]),
                -item,
            ),
        )
        record = pool.pop(index)
        selected.append(record)
        category_counts[_category(record)] += 1

    return sorted(selected, key=_sample_id)


def _record_payload(
    record: object,
    *,
    role: str,
    source_split: str,
    video_id_map: Mapping[str, str],
    video_extension: str,
) -> dict[str, object]:
    video_id = str(getattr(record, "video_id"))
    options = [str(item) for item in getattr(record, "options")]
    answer_index = int(getattr(record, "answer_index"))

    if len(options) != 5:
        raise ValueError(
            f"{_sample_id(record)} must contain five options, got {len(options)}."
        )
    if answer_index not in range(5):
        raise ValueError(
            f"{_sample_id(record)} has invalid answer index {answer_index}."
        )

    answer_letter = OPTION_LABELS[answer_index]
    mapped_video_id = video_id_map.get(video_id, video_id)
    video_relpath = normalize_video_filename(
        mapped_video_id,
        extension=video_extension,
    )
    question = str(getattr(record, "question")).strip()

    option_block = "\n".join(
        f"{label}. {option}"
        for label, option in zip(OPTION_LABELS, options, strict=True)
    )
    prompt = (
        "Watch the video carefully and answer the multiple-choice question.\n"
        "Choose the single best option using visual evidence from the video.\n"
        "Return exactly one uppercase letter: A, B, C, D, or E. "
        "Do not include an explanation.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_block}\n\n"
        "Answer:"
    )

    return {
        "schema_version": "1.0",
        "dataset": "nextqa",
        "source_split": source_split,
        "stage6a_role": role,
        "sample_id": _sample_id(record),
        "video_id": video_id,
        "video_relpath": video_relpath,
        "question": question,
        "options": options,
        "answer_index": answer_index,
        "answer_letter": answer_letter,
        "gold_answer_letter": answer_letter,
        "answer_text": options[answer_index],
        "question_type": _question_type(record),
        "question_category": _category(record),
        "prompt": prompt,
        "assistant_target": answer_letter,
        "video": video_relpath,
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n{prompt}",
            },
            {
                "from": "gpt",
                "value": answer_letter,
            },
        ],
    }


def build_video_split(
    records: Sequence[object],
    *,
    role: str,
    source_split: str,
    video_id_map: Mapping[str, str],
    num_videos: int,
    max_questions_per_video: int,
    seed: int,
    excluded_video_ids: set[str] | None = None,
    video_extension: str = ".mp4",
) -> list[dict[str, object]]:
    """Build one deterministic split from complete videos."""
    grouped = _group_by_video(
        records,
        excluded_video_ids=set(excluded_video_ids or set()),
    )
    selected_video_ids = _select_video_ids(
        grouped,
        num_videos=num_videos,
        seed=seed,
    )

    rows: list[dict[str, object]] = []
    for video_id in selected_video_ids:
        sampled_records = _sample_questions(
            grouped[video_id],
            max_questions=max_questions_per_video,
            seed=seed,
            video_id=video_id,
        )
        rows.extend(
            _record_payload(
                record,
                role=role,
                source_split=source_split,
                video_id_map=video_id_map,
                video_extension=video_extension,
            )
            for record in sampled_records
        )

    return sorted(rows, key=lambda row: str(row["sample_id"]))


def split_video_ids(rows: Sequence[Mapping[str, object]]) -> set[str]:
    return {str(row["video_id"]) for row in rows}


def split_video_filenames(rows: Sequence[Mapping[str, object]]) -> set[str]:
    return {
        PurePosixPath(str(row["video_relpath"]).replace("\\", "/")).name
        for row in rows
    }


def assert_disjoint_splits(
    named_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, list[str]]:
    """Reject logical or physical video overlap across every pair of splits."""
    names = sorted(named_rows)
    overlaps: dict[str, list[str]] = {}

    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            logical_overlap = sorted(
                split_video_ids(named_rows[left_name])
                & split_video_ids(named_rows[right_name])
            )
            physical_overlap = sorted(
                split_video_filenames(named_rows[left_name])
                & split_video_filenames(named_rows[right_name])
            )

            overlaps[f"{left_name}__{right_name}__logical_video_ids"] = (
                logical_overlap
            )
            overlaps[f"{left_name}__{right_name}__physical_video_files"] = (
                physical_overlap
            )

            if logical_overlap or physical_overlap:
                raise ValueError(
                    f"Data leakage between {left_name!r} and {right_name!r}: "
                    f"logical={logical_overlap[:5]}, physical={physical_overlap[:5]}"
                )

    return overlaps


def split_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    videos = split_video_ids(rows)
    category_counts = Counter(
        str(row["question_category"]) for row in rows
    )
    question_type_counts = Counter(
        str(row["question_type"]) for row in rows
    )
    answer_counts = Counter(str(row["answer_letter"]) for row in rows)

    per_video = Counter(str(row["video_id"]) for row in rows)
    return {
        "num_questions": len(rows),
        "num_videos": len(videos),
        "questions_per_video": {
            "min": min(per_video.values()) if per_video else None,
            "max": max(per_video.values()) if per_video else None,
            "mean": (
                sum(per_video.values()) / len(per_video)
                if per_video
                else None
            ),
        },
        "question_category_counts": dict(sorted(category_counts.items())),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "answer_letter_counts": {
            label: answer_counts.get(label, 0)
            for label in OPTION_LABELS
        },
        "video_ids": sorted(videos),
        "video_files": sorted(split_video_filenames(rows)),
    }


def write_jsonl(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    if not rows:
        raise ValueError("Refusing to write an empty split.")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return destination


def write_qwen_json(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    if not rows:
        raise ValueError("Refusing to write an empty Qwen training split.")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "video": row["video"],
            "conversations": row["conversations"],
        }
        for row in rows
    ]
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def write_video_list(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(sorted(split_video_filenames(rows))) + "\n",
        encoding="utf-8",
    )
    return destination
