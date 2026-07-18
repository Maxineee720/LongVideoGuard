from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from longvideoguard.video import probe_video


@dataclass(frozen=True)
class VideoValidationResult:
    video_id: str
    video_relpath: str
    resolved_path: str
    status: str
    error: str | None = None
    frame_count: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    annotated_frame_count: int | None = None
    frame_count_relative_error: float | None = None


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSONL manifest not found: {source}")

    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Line {line_number} of {source} must contain a JSON object."
                )
            rows.append(payload)

    if not rows:
        raise ValueError(f"No records found in JSONL manifest: {source}")
    return rows


def unique_video_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    unique: dict[str, Mapping[str, object]] = {}
    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        video_relpath = str(row.get("video_relpath", "")).strip()
        if not video_id or not video_relpath:
            raise ValueError("Every manifest row must include video_id and video_relpath.")
        unique.setdefault(video_id, row)
    return list(unique.values())


def validate_manifest_videos(
    manifest_path: str | Path,
    video_root: str | Path,
) -> tuple[list[VideoValidationResult], dict[str, object]]:
    rows = unique_video_rows(load_jsonl(manifest_path))
    root = Path(video_root).expanduser().resolve()
    results: list[VideoValidationResult] = []

    for row in rows:
        video_id = str(row["video_id"])
        video_relpath = str(row["video_relpath"])
        video_path = root / video_relpath

        metadata_field = row.get("source_metadata")
        annotated_frames = None
        if isinstance(metadata_field, Mapping):
            try:
                annotated_frames = int(metadata_field.get("frame_count"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                annotated_frames = None

        if not video_path.is_file():
            results.append(
                VideoValidationResult(
                    video_id=video_id,
                    video_relpath=video_relpath,
                    resolved_path=str(video_path),
                    status="missing",
                    error="File does not exist.",
                    annotated_frame_count=annotated_frames,
                )
            )
            continue

        try:
            metadata = probe_video(video_path)
        except Exception as exc:  # noqa: BLE001
            results.append(
                VideoValidationResult(
                    video_id=video_id,
                    video_relpath=video_relpath,
                    resolved_path=str(video_path),
                    status="unreadable",
                    error=str(exc),
                    annotated_frame_count=annotated_frames,
                )
            )
            continue

        relative_error = None
        if annotated_frames and annotated_frames > 0:
            relative_error = abs(metadata.frame_count - annotated_frames) / annotated_frames

        results.append(
            VideoValidationResult(
                video_id=video_id,
                video_relpath=video_relpath,
                resolved_path=str(video_path),
                status="ok",
                frame_count=metadata.frame_count,
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
                duration_seconds=metadata.duration_seconds,
                annotated_frame_count=annotated_frames,
                frame_count_relative_error=relative_error,
            )
        )

    status_counts = Counter(item.status for item in results)
    summary = {
        "num_unique_videos": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "all_videos_ready": status_counts.get("ok", 0) == len(results),
    }
    return results, summary


def write_video_validation_report(
    results: Iterable[VideoValidationResult],
    summary: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "summary": dict(summary),
                "videos": [asdict(item) for item in results],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
