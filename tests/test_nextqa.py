import json
from pathlib import Path

import pytest

from longvideoguard.datasets.nextqa import (
    build_video_level_pilot,
    compute_nextqa_stats,
    load_nextqa_csv,
    write_nextqa_manifest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "nextqa_sample.csv"


def test_load_nextqa_csv_parses_correct_answer() -> None:
    records = load_nextqa_csv(FIXTURE)
    assert len(records) == 5
    assert records[0].sample_id == "video_001:1"
    assert records[0].answer_text == "to rest"
    assert records[0].category == "causal"


def test_compute_nextqa_stats() -> None:
    stats = compute_nextqa_stats(load_nextqa_csv(FIXTURE))
    assert stats["num_questions"] == 5
    assert stats["num_videos"] == 3
    assert stats["category_counts"] == {
        "causal": 2,
        "descriptive": 1,
        "temporal": 2,
    }


def test_build_video_level_pilot_is_reproducible() -> None:
    records = load_nextqa_csv(FIXTURE)
    first = build_video_level_pilot(records, num_videos=2, seed=42)
    second = build_video_level_pilot(records, num_videos=2, seed=42)
    assert [record.sample_id for record in first] == [
        record.sample_id for record in second
    ]
    assert len({record.video_id for record in first}) == 2


def test_build_video_level_pilot_caps_questions_per_video() -> None:
    records = load_nextqa_csv(FIXTURE)
    pilot = build_video_level_pilot(
        records,
        num_videos=3,
        seed=42,
        max_questions_per_video=1,
    )
    assert len(pilot) == 3
    assert len({record.video_id for record in pilot}) == 3


def test_write_nextqa_manifest(tmp_path: Path) -> None:
    records = load_nextqa_csv(FIXTURE)[:2]
    output = write_nextqa_manifest(
        records,
        tmp_path / "pilot.jsonl",
        split="val",
        video_id_map={"video_001": "mapped_001"},
    )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["video_relpath"] == "mapped_001.mp4"
    assert rows[0]["question_category"] == "causal"
    assert rows[1]["answer_text"] == "they walked"


def test_load_nextqa_csv_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_nextqa_csv("does-not-exist.csv")
