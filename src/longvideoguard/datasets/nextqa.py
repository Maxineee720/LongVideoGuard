from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


REQUIRED_COLUMNS = {
    "video",
    "frame_count",
    "width",
    "height",
    "question",
    "answer",
    "qid",
    "type",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
}

CATEGORY_BY_PREFIX = {
    "C": "causal",
    "T": "temporal",
    "D": "descriptive",
}


@dataclass(frozen=True)
class NextQARecord:
    """One multiple-choice NExT-QA annotation."""

    video_id: str
    frame_count: int
    width: int
    height: int
    question: str
    answer_index: int
    qid: str
    question_type: str
    options: tuple[str, str, str, str, str]

    # Most official rows keep the original readable ``video_id:qid`` ID.
    # A suffix is assigned only when the same qid is reused for a different
    # question in the same video. It is loader-generated rather than part of
    # the annotation itself, so it is excluded from dataclass equality.
    sample_id_suffix: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def base_sample_id(self) -> str:
        """Return the human-readable ID used before collision resolution."""
        return f"{self.video_id}:{self.qid}"

    @property
    def content_fingerprint(self) -> str:
        """Return a stable fingerprint for disambiguating reused qids.

        The fingerprint excludes the answer and media metadata on purpose.
        Therefore, two otherwise identical annotations with conflicting
        answers or metadata resolve to the same candidate ID and are rejected
        by ``load_nextqa_csv`` rather than silently treated as separate tasks.
        """
        payload = "\x1f".join(
            [
                self.video_id,
                self.qid,
                self.question_type,
                self.question,
                *self.options,
            ]
        )
        return hashlib.sha1(
            payload.encode("utf-8")
        ).hexdigest()[:12]

    @property
    def sample_id(self) -> str:
        """Return a stable sample ID, adding a suffix only for qid collisions."""
        if self.sample_id_suffix is None:
            return self.base_sample_id
        return f"{self.base_sample_id}:{self.sample_id_suffix}"

    @property
    def answer_text(self) -> str:
        """Return the correct multiple-choice answer text."""
        return self.options[self.answer_index]

    @property
    def category(self) -> str:
        """Map the official subtype prefix to a broad reasoning category."""
        try:
            return CATEGORY_BY_PREFIX[self.question_type[0].upper()]
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"Unsupported NExT-QA question type: "
                f"{self.question_type!r}"
            ) from exc

    def to_manifest_row(
        self,
        *,
        split: str,
        video_relpath: str | None = None,
    ) -> dict[str, object]:
        """Convert this record into LongVideoGuard's JSONL manifest schema."""
        return {
            "dataset": "nextqa",
            "split": split,
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "video_relpath": video_relpath,
            "question": self.question,
            "options": list(self.options),
            "answer_index": self.answer_index,
            "answer_text": self.answer_text,
            "question_type": self.question_type,
            "question_category": self.category,
            "source_metadata": {
                "frame_count": self.frame_count,
                "width": self.width,
                "height": self.height,
                "qid": self.qid,
            },
        }


def _parse_positive_int(
    raw: str,
    *,
    field: str,
    line_number: int,
) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Line {line_number}: field {field!r} must be an integer, "
            f"got {raw!r}."
        ) from exc

    if value <= 0:
        raise ValueError(
            f"Line {line_number}: field {field!r} must be positive, "
            f"got {value}."
        )
    return value


def _conflicting_record_message(
    *,
    line_number: int,
    sample_id: str,
    previous: NextQARecord,
    current: NextQARecord,
) -> str:
    """Build an actionable error for conflicting annotations."""
    previous_payload = asdict(previous)
    current_payload = asdict(current)
    previous_payload.pop("sample_id_suffix", None)
    current_payload.pop("sample_id_suffix", None)

    return (
        f"Line {line_number}: conflicting duplicate annotation "
        f"{sample_id!r}. The question identity is the same, but one or more "
        "answer/media fields differ. "
        f"Existing={previous_payload!r}; new={current_payload!r}."
    )


def load_nextqa_csv(path: str | Path) -> list[NextQARecord]:
    """Load and validate an official NExT-QA multiple-choice CSV split.

    Behaviour for repeated IDs:

    - unique ``video_id:qid`` pairs keep the readable original sample ID;
    - exact duplicate annotation rows are deduplicated;
    - different questions reusing one ``video_id:qid`` receive stable
      content-fingerprint suffixes;
    - the same question/options with conflicting answers or media metadata
      raise an error instead of being silently accepted.
    """
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"NExT-QA annotation file not found: {csv_path}"
        )

    records: list[NextQARecord] = []

    # Each base ID points to entries in ``records`` so the first occurrence can
    # be replaced with a suffixed version if a later distinct question reuses
    # the same qid.
    seen_by_base_id: dict[
        str,
        list[tuple[int, NextQARecord]],
    ] = defaultdict(list)

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(
                f"Missing required NExT-QA columns in {csv_path}: "
                f"{missing_columns}"
            )

        for line_number, row in enumerate(reader, start=2):
            video_id = (row["video"] or "").strip()
            question = (row["question"] or "").strip()
            qid = (row["qid"] or "").strip()
            question_type = (row["type"] or "").strip().upper()
            options = tuple(
                (row[f"a{index}"] or "").strip()
                for index in range(5)
            )

            if not video_id:
                raise ValueError(
                    f"Line {line_number}: video ID is empty."
                )
            if not question:
                raise ValueError(
                    f"Line {line_number}: question is empty."
                )
            if not qid:
                raise ValueError(
                    f"Line {line_number}: qid is empty."
                )
            if not all(options):
                raise ValueError(
                    f"Line {line_number}: all five answer options "
                    "are required."
                )

            try:
                answer_index = int(row["answer"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Line {line_number}: answer must be an integer "
                    "from 0 to 4."
                ) from exc

            if answer_index not in range(5):
                raise ValueError(
                    f"Line {line_number}: answer index must be from 0 to 4, "
                    f"got {answer_index}."
                )

            record = NextQARecord(
                video_id=video_id,
                frame_count=_parse_positive_int(
                    row["frame_count"],
                    field="frame_count",
                    line_number=line_number,
                ),
                width=_parse_positive_int(
                    row["width"],
                    field="width",
                    line_number=line_number,
                ),
                height=_parse_positive_int(
                    row["height"],
                    field="height",
                    line_number=line_number,
                ),
                question=question,
                answer_index=answer_index,
                qid=qid,
                question_type=question_type,
                options=options,  # type: ignore[arg-type]
            )

            # Trigger category validation here rather than much later.
            _ = record.category

            base_id = record.base_sample_id
            existing_entries = seen_by_base_id[base_id]

            # ``sample_id_suffix`` is excluded from equality, so this also
            # recognises an exact duplicate appearing after collision
            # disambiguation has already taken place.
            if any(
                previous == record
                for _, previous in existing_entries
            ):
                continue

            if not existing_entries:
                record_index = len(records)
                records.append(record)
                seen_by_base_id[base_id].append(
                    (record_index, record)
                )
                continue

            # A distinct question reused this video's qid. Add a stable
            # fingerprint to every record sharing that base ID, including the
            # first row that was previously stored without a suffix.
            disambiguated_entries: list[
                tuple[int, NextQARecord]
            ] = []

            for record_index, previous in existing_entries:
                disambiguated = previous
                if previous.sample_id_suffix is None:
                    disambiguated = replace(
                        previous,
                        sample_id_suffix=(
                            previous.content_fingerprint
                        ),
                    )
                    records[record_index] = disambiguated

                disambiguated_entries.append(
                    (record_index, disambiguated)
                )

            current = replace(
                record,
                sample_id_suffix=record.content_fingerprint,
            )

            # Identical question identity but a different answer or media
            # field is a real annotation conflict, not a valid qid reuse.
            conflicting = next(
                (
                    previous
                    for _, previous in disambiguated_entries
                    if previous.sample_id == current.sample_id
                ),
                None,
            )
            if conflicting is not None:
                raise ValueError(
                    _conflicting_record_message(
                        line_number=line_number,
                        sample_id=current.sample_id,
                        previous=conflicting,
                        current=current,
                    )
                )

            # A truncated SHA-1 collision is extremely unlikely, but refusing
            # it is safer than silently creating duplicate IDs.
            existing_sample_ids = {
                previous.sample_id
                for _, previous in disambiguated_entries
            }
            if current.sample_id in existing_sample_ids:
                raise ValueError(
                    f"Line {line_number}: sample fingerprint collision "
                    f"for {current.sample_id!r}."
                )

            record_index = len(records)
            records.append(current)
            disambiguated_entries.append(
                (record_index, current)
            )
            seen_by_base_id[base_id] = disambiguated_entries

    if not records:
        raise ValueError(
            f"No NExT-QA records found in {csv_path}."
        )
    return records


def compute_nextqa_stats(
    records: Sequence[NextQARecord],
) -> dict[str, object]:
    """Compute deterministic summary statistics for a split or pilot."""
    if not records:
        raise ValueError(
            "Cannot compute statistics for an empty record sequence."
        )

    category_counts = Counter(
        record.category for record in records
    )
    subtype_counts = Counter(
        record.question_type for record in records
    )
    answer_index_counts = Counter(
        str(record.answer_index) for record in records
    )
    per_video_counts = Counter(
        record.video_id for record in records
    )

    return {
        "num_questions": len(records),
        "num_videos": len(per_video_counts),
        "category_counts": dict(
            sorted(category_counts.items())
        ),
        "subtype_counts": dict(
            sorted(subtype_counts.items())
        ),
        "answer_index_counts": dict(
            sorted(answer_index_counts.items())
        ),
        "questions_per_video": {
            "min": min(per_video_counts.values()),
            "max": max(per_video_counts.values()),
            "mean": (
                sum(per_video_counts.values())
                / len(per_video_counts)
            ),
        },
    }


def build_video_level_pilot(
    records: Sequence[NextQARecord],
    *,
    num_videos: int,
    seed: int,
    max_questions_per_video: int | None = None,
) -> list[NextQARecord]:
    """Select a reproducible pilot while keeping selection video-level.

    A greedy score favours videos covering broad question categories that are
    currently under-represented. When ``max_questions_per_video`` is supplied,
    questions are deterministically sampled within each selected video.
    """
    if not records:
        raise ValueError(
            "Cannot build a pilot from an empty record sequence."
        )
    if num_videos <= 0:
        raise ValueError("num_videos must be positive.")
    if (
        max_questions_per_video is not None
        and max_questions_per_video <= 0
    ):
        raise ValueError(
            "max_questions_per_video must be positive when provided."
        )

    by_video: dict[str, list[NextQARecord]] = defaultdict(list)
    for record in records:
        by_video[record.video_id].append(record)

    if num_videos > len(by_video):
        raise ValueError(
            f"Requested {num_videos} videos, but only "
            f"{len(by_video)} are available."
        )

    rng = random.Random(seed)
    candidate_ids = list(by_video)
    rng.shuffle(candidate_ids)

    selected_ids: list[str] = []
    selected_category_counts: Counter[str] = Counter()

    while len(selected_ids) < num_videos:
        best_video_id: str | None = None
        best_score = float("-inf")

        for video_id in candidate_ids:
            if video_id in selected_ids:
                continue

            video_categories = Counter(
                record.category
                for record in by_video[video_id]
            )
            coverage_score = sum(
                count / (
                    1
                    + selected_category_counts[category]
                )
                for category, count
                in video_categories.items()
            )

            # Tiny deterministic tie-breaker retains seed-controlled order.
            tie_breaker = (
                -candidate_ids.index(video_id) * 1e-9
            )
            score = coverage_score + tie_breaker

            if score > best_score:
                best_score = score
                best_video_id = video_id

        assert best_video_id is not None
        selected_ids.append(best_video_id)
        selected_category_counts.update(
            record.category
            for record in by_video[best_video_id]
        )

    pilot: list[NextQARecord] = []

    for video_id in selected_ids:
        video_records = sorted(
            by_video[video_id],
            key=lambda record: (
                record.question_type,
                record.qid,
                record.sample_id,
            ),
        )

        if (
            max_questions_per_video is not None
            and len(video_records)
            > max_questions_per_video
        ):
            local_rng = random.Random(
                f"{seed}:{video_id}"
            )
            chosen_indices = sorted(
                local_rng.sample(
                    range(len(video_records)),
                    max_questions_per_video,
                )
            )
            video_records = [
                video_records[index]
                for index in chosen_indices
            ]

        pilot.extend(video_records)

    return pilot


def load_video_id_map(
    path: str | Path | None,
) -> dict[str, str]:
    """Load the official NExT-QA-to-VidOR ID mapping when provided."""
    if path is None:
        return {}

    map_path = Path(path).expanduser().resolve()
    if not map_path.is_file():
        raise FileNotFoundError(
            f"Video ID map not found: {map_path}"
        )

    with map_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError(
            "The video ID map must be a JSON object."
        )

    return {
        str(key): str(value)
        for key, value in payload.items()
    }


def write_nextqa_manifest(
    records: Iterable[NextQARecord],
    output_path: str | Path,
    *,
    split: str,
    video_id_map: Mapping[str, str] | None = None,
    video_extension: str = ".mp4",
) -> Path:
    """Write a JSONL manifest consumable by later VLM inference code."""
    if not split.strip():
        raise ValueError("split must be non-empty.")

    mapping = video_id_map or {}
    extension = (
        video_extension
        if video_extension.startswith(".")
        else f".{video_extension}"
    )

    destination = Path(
        output_path
    ).expanduser().resolve()
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with destination.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            mapped_id = mapping.get(
                record.video_id,
                record.video_id,
            )
            mapped_path = PurePosixPath(
                str(mapped_id).replace("\\", "/")
            )
            mapped_name = mapped_path.name

            if not mapped_name:
                raise ValueError(
                    "Could not derive a video filename "
                    f"from mapping value {mapped_id!r}."
                )

            if mapped_path.suffix:
                video_filename = mapped_name
            else:
                video_filename = (
                    f"{mapped_name}{extension}"
                )

            row = record.to_manifest_row(
                split=split,
                video_relpath=video_filename,
            )
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    if count == 0:
        destination.unlink(missing_ok=True)
        raise ValueError(
            "Refusing to write an empty NExT-QA manifest."
        )

    return destination


def write_stats(
    stats: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    """Write summary statistics as formatted JSON."""
    destination = Path(
        output_path
    ).expanduser().resolve()
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_text(
        json.dumps(
            dict(stats),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
