from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float


def probe_video(path: str | Path) -> VideoMetadata:
    """Read basic metadata from a local video file."""
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count <= 0:
            raise ValueError(f"Invalid frame count for video: {video_path}")
        if fps <= 0:
            raise ValueError(f"Invalid FPS for video: {video_path}")

        return VideoMetadata(
            path=video_path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=frame_count / fps,
        )
    finally:
        capture.release()
