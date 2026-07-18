import json
from pathlib import Path

from longvideoguard.video import VideoMetadata
from longvideoguard.video_validation import (
    load_jsonl,
    unique_video_rows,
    validate_manifest_videos,
)


def write_manifest(path: Path) -> None:
    rows = [
        {
            "video_id": "v1",
            "video_relpath": "mapped_v1.mp4",
            "source_metadata": {"frame_count": 100},
        },
        {
            "video_id": "v1",
            "video_relpath": "mapped_v1.mp4",
            "source_metadata": {"frame_count": 100},
        },
        {
            "video_id": "v2",
            "video_relpath": "mapped_v2.mp4",
            "source_metadata": {"frame_count": 80},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_jsonl_and_deduplicate(tmp_path: Path) -> None:
    manifest = tmp_path / "pilot.jsonl"
    write_manifest(manifest)
    rows = load_jsonl(manifest)
    assert len(rows) == 3
    assert len(unique_video_rows(rows)) == 2


def test_missing_videos_are_reported(tmp_path: Path) -> None:
    manifest = tmp_path / "pilot.jsonl"
    write_manifest(manifest)
    results, summary = validate_manifest_videos(manifest, tmp_path / "videos")
    assert len(results) == 2
    assert summary["status_counts"] == {"missing": 2}
    assert summary["all_videos_ready"] is False


def test_readable_videos_are_reported(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "pilot.jsonl"
    write_manifest(manifest)
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "mapped_v1.mp4").touch()
    (video_root / "mapped_v2.mp4").touch()

    def fake_probe(path: Path) -> VideoMetadata:
        frames = 100 if path.name == "mapped_v1.mp4" else 80
        return VideoMetadata(
            path=path,
            frame_count=frames,
            fps=25.0,
            width=640,
            height=480,
            duration_seconds=frames / 25.0,
        )

    monkeypatch.setattr(
        "longvideoguard.video_validation.probe_video",
        fake_probe,
    )
    results, summary = validate_manifest_videos(manifest, video_root)
    assert summary["status_counts"] == {"ok": 2}
    assert summary["all_videos_ready"] is True
    assert all(item.frame_count_relative_error == 0 for item in results)
