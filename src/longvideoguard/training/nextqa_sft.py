from __future__ import annotations
import json, random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

OPTION_LABELS = ("A", "B", "C", "D", "E")

def normalize_video_filename(mapped_video_id: object, *, extension: str = ".mp4") -> str:
    if not extension.startswith("."):
        raise ValueError("extension must start with '.'")
    raw = str(mapped_video_id).strip()
    if not raw:
        raise ValueError("mapped_video_id must be non-empty")
    name = PurePosixPath(raw.replace("\\", "/")).name
    if not name:
        raise ValueError(f"Could not derive a filename from {raw!r}")
    return name if PurePosixPath(name).suffix else f"{name}{extension}"

def load_manifest_video_ids(path: str | Path) -> set[str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Manifest not found: {source}")
    video_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            video_id = str(row.get("video_id", "")).strip()
            if not video_id:
                raise ValueError(f"Line {line_number} has no video_id")
            video_ids.add(video_id)
    return video_ids

def _category(record: object) -> str:
    return str(getattr(record, "category", getattr(record, "question_category", "unknown")))

def _sample_id(record: object) -> str:
    value = getattr(record, "sample_id", None)
    return str(value) if value else f"{getattr(record, 'video_id')}:{getattr(record, 'qid')}"

def _group_by_video(records: Iterable[object], excluded: set[str]) -> dict[str, list[object]]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for record in records:
        video_id = str(getattr(record, "video_id"))
        if video_id not in excluded:
            grouped[video_id].append(record)
    return dict(grouped)

def _select_video_ids(grouped: Mapping[str, Sequence[object]], num_videos: int, seed: int) -> list[str]:
    if num_videos <= 0 or num_videos > len(grouped):
        raise ValueError(f"Invalid num_videos={num_videos}; available={len(grouped)}")
    rng = random.Random(seed)
    tie_break = {video_id: rng.random() for video_id in grouped}
    remaining, selected = set(grouped), []
    selected_categories: Counter[str] = Counter()
    while len(selected) < num_videos:
        def score(video_id: str):
            counts = Counter(_category(r) for r in grouped[video_id])
            coverage = sum(count / (1 + selected_categories[c]) for c, count in counts.items())
            return coverage + 0.05 * len(counts), tie_break[video_id], video_id
        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
        selected_categories.update(_category(r) for r in grouped[chosen])
    return selected

def _sample_questions(records: Sequence[object], max_questions: int, seed: int, video_id: str) -> list[object]:
    if max_questions <= 0:
        raise ValueError("max_questions must be positive")
    ordered = sorted(records, key=_sample_id)
    if len(ordered) <= max_questions:
        return ordered
    rng = random.Random(f"{seed}:{video_id}")
    rng.shuffle(ordered)
    selected, category_counts = [], Counter()
    while ordered and len(selected) < max_questions:
        index = max(range(len(ordered)), key=lambda i: (1/(1+category_counts[_category(ordered[i])]), -i))
        item = ordered.pop(index)
        selected.append(item)
        category_counts[_category(item)] += 1
    return sorted(selected, key=_sample_id)

def _record_to_row(record: object, role: str, mapping: Mapping[str, str], extension: str) -> dict[str, object]:
    video_id = str(getattr(record, "video_id"))
    options = [str(x) for x in getattr(record, "options")]
    answer_index = int(getattr(record, "answer_index"))
    if len(options) != 5 or answer_index not in range(5):
        raise ValueError(f"Invalid NExT-QA record {_sample_id(record)}")
    letter = OPTION_LABELS[answer_index]
    video_name = normalize_video_filename(mapping.get(video_id, video_id), extension=extension)
    question = str(getattr(record, "question")).strip()
    option_block = "\n".join(f"{label}. {text}" for label, text in zip(OPTION_LABELS, options, strict=True))
    prompt = (
        "Watch the video carefully and answer the multiple-choice question.\n"
        "Choose the single best option using visual evidence from the video.\n"
        "Return exactly one uppercase letter: A, B, C, D, or E. Do not include an explanation.\n\n"
        f"Question: {question}\n\nOptions:\n{option_block}\n\nAnswer:"
    )
    return {
        "schema_version": "1.0", "dataset": "nextqa", "source_split": "train",
        "sft_role": role, "sample_id": _sample_id(record), "video_id": video_id,
        "video_relpath": video_name, "question": question, "options": options,
        "answer_index": answer_index, "answer_letter": letter,
        "answer_text": options[answer_index],
        "question_type": str(getattr(record, "question_type", "unknown")),
        "question_category": _category(record), "prompt": prompt,
        "assistant_target": letter, "video": video_name,
        "conversations": [
            {"from": "human", "value": f"<video>\n{prompt}"},
            {"from": "gpt", "value": letter},
        ],
    }

def build_nextqa_sft_splits(records: Sequence[object], *, video_id_map: Mapping[str, str], excluded_video_ids: set[str] | None = None, train_num_videos: int = 4, holdout_num_videos: int = 4, max_questions_per_video: int = 4, seed: int = 42, video_extension: str = ".mp4") -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    excluded = set(excluded_video_ids or set())
    grouped = _group_by_video(records, excluded)
    if len(grouped) < train_num_videos + holdout_num_videos:
        raise ValueError("Not enough eligible videos")
    train_ids = _select_video_ids(grouped, train_num_videos, seed)
    remaining = {k:v for k,v in grouped.items() if k not in set(train_ids)}
    holdout_ids = _select_video_ids(remaining, holdout_num_videos, seed+1)
    def build(video_ids: Sequence[str], role: str):
        rows=[]
        for video_id in video_ids:
            for record in _sample_questions(grouped[video_id], max_questions_per_video, seed, video_id):
                rows.append(_record_to_row(record, role, video_id_map, video_extension))
        return sorted(rows, key=lambda x: str(x["sample_id"]))
    train_rows, holdout_rows = build(train_ids, "overfit_train"), build(holdout_ids, "tiny_holdout")
    train_videos={str(r["video_id"]) for r in train_rows}; holdout_videos={str(r["video_id"]) for r in holdout_rows}
    if train_videos & holdout_videos or (train_videos|holdout_videos) & excluded:
        raise AssertionError("Video leakage detected")
    return train_rows, holdout_rows

def compute_sft_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    videos={str(r["video_id"]) for r in rows}
    return {
        "num_questions": len(rows), "num_videos": len(videos),
        "question_category_counts": dict(sorted(Counter(str(r["question_category"]) for r in rows).items())),
        "question_type_counts": dict(sorted(Counter(str(r["question_type"]) for r in rows).items())),
        "answer_letter_counts": {label: sum(r["answer_letter"]==label for r in rows) for label in OPTION_LABELS},
        "video_ids": sorted(videos),
    }

def write_jsonl(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> Path:
    if not rows: raise ValueError("Refusing to write an empty split")
    destination=Path(output_path).expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(dict(r), ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    return destination

def write_qwen_training_json(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> Path:
    if not rows: raise ValueError("Refusing to write an empty split")
    destination=Path(output_path).expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    payload=[{"video": r["video"], "conversations": r["conversations"]} for r in rows]
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return destination
