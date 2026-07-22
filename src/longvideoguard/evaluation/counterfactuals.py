from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

VIDEO_CONDITIONS = (
    "original",
    "reversed",
    "shuffled",
    "black",
    "relevant_mask",
    "random_mask",
)
ALL_CONDITIONS = VIDEO_CONDITIONS + ("question_only",)


def stable_seed(sample_id: object, *, base_seed: int = 42) -> int:
    digest = hashlib.sha256(
        f"{base_seed}::{sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def read_video_frames(video_path: str | Path) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required. Install opencv-python-headless."
        ) from exc

    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")

    frames: list[np.ndarray] = []
    try:
        while True:
            success, bgr = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames decoded from: {source}")
    return frames


def write_h264_video(
    rgb_frames: Sequence[np.ndarray],
    output_path: str | Path,
    *,
    fps: float = 1.0,
) -> Path:
    if not rgb_frames:
        raise ValueError("rgb_frames must be non-empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg was not found.")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required.") from exc

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="longvideoguard_counterfactual_"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        for index, frame in enumerate(rgb_frames):
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("Frames must be RGB HxWx3 arrays.")
            Image.fromarray(array.astype(np.uint8)).save(
                temporary_root / f"{index:06d}.png"
            )

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(temporary_root / "%06d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg failed to create counterfactual video:\n"
                + result.stderr
            )

    if not destination.is_file() or destination.stat().st_size == 0:
        raise ValueError(f"Video write failed: {destination}")
    return destination


def top_relevant_indices(
    query_scores: Sequence[float],
    *,
    count: int,
) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not query_scores:
        raise ValueError("query_scores must be non-empty")
    count = min(count, len(query_scores))
    return sorted(
        range(len(query_scores)),
        key=lambda index: (-float(query_scores[index]), index),
    )[:count]


def random_mask_indices(
    frame_count: int,
    *,
    count: int,
    sample_id: object,
    seed: int = 42,
) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    count = min(count, frame_count)
    rng = random.Random(stable_seed(sample_id, base_seed=seed))
    return sorted(rng.sample(range(frame_count), count))


def perturb_frames(
    rgb_frames: Sequence[np.ndarray],
    *,
    condition: str,
    sample_id: object,
    query_scores: Sequence[float] | None = None,
    mask_count: int = 2,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[int]]:
    if not rgb_frames:
        raise ValueError("rgb_frames must be non-empty")
    frames = [np.asarray(frame).copy() for frame in rgb_frames]
    frame_count = len(frames)

    if condition == "original":
        return frames, []
    if condition == "reversed":
        return list(reversed(frames)), []
    if condition == "shuffled":
        indices = list(range(frame_count))
        random.Random(
            stable_seed(sample_id, base_seed=seed)
        ).shuffle(indices)
        return [frames[index] for index in indices], indices
    if condition == "black":
        return [np.zeros_like(frame) for frame in frames], list(
            range(frame_count)
        )
    if condition == "relevant_mask":
        if query_scores is None:
            raise ValueError(
                "relevant_mask requires query_scores."
            )
        if len(query_scores) != frame_count:
            raise ValueError(
                "query_scores length must match frame count."
            )
        masked = top_relevant_indices(
            query_scores,
            count=mask_count,
        )
        for index in masked:
            frames[index] = np.zeros_like(frames[index])
        return frames, masked
    if condition == "random_mask":
        masked = random_mask_indices(
            frame_count,
            count=mask_count,
            sample_id=sample_id,
            seed=seed,
        )
        for index in masked:
            frames[index] = np.zeros_like(frames[index])
        return frames, masked

    raise ValueError(f"Unsupported condition: {condition!r}")


def subset_by_category(
    rows: Sequence[Mapping[str, object]],
    category: str,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if str(row.get("question_category", "unknown")) == category
    ]


def accuracy(rows: Sequence[Mapping[str, object]]) -> float:
    if not rows:
        raise ValueError("rows must be non-empty")
    return sum(bool(row["is_correct"]) for row in rows) / len(rows)


def diagnostic_summary(
    predictions_by_condition: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
) -> dict[str, object]:
    required = set(ALL_CONDITIONS)
    missing = sorted(required - set(predictions_by_condition))
    if missing:
        raise ValueError(f"Missing conditions: {missing}")

    original = predictions_by_condition["original"]
    original_accuracy = accuracy(original)

    condition_accuracies = {
        condition: accuracy(rows)
        for condition, rows in predictions_by_condition.items()
    }

    temporal_original = subset_by_category(original, "temporal")
    temporal_reversed = subset_by_category(
        predictions_by_condition["reversed"],
        "temporal",
    )
    temporal_shuffled = subset_by_category(
        predictions_by_condition["shuffled"],
        "temporal",
    )

    relevant_drop = (
        original_accuracy
        - condition_accuracies["relevant_mask"]
    )
    random_drop = (
        original_accuracy
        - condition_accuracies["random_mask"]
    )

    return {
        "condition_accuracies": condition_accuracies,
        "visual_dependence": {
            "original_minus_question_only": (
                original_accuracy
                - condition_accuracies["question_only"]
            ),
            "original_minus_black": (
                original_accuracy
                - condition_accuracies["black"]
            ),
        },
        "temporal_order_sensitivity": {
            "temporal_original_accuracy": accuracy(temporal_original),
            "temporal_reversed_accuracy": accuracy(temporal_reversed),
            "temporal_shuffled_accuracy": accuracy(temporal_shuffled),
            "drop_after_reversal": (
                accuracy(temporal_original)
                - accuracy(temporal_reversed)
            ),
            "drop_after_shuffle": (
                accuracy(temporal_original)
                - accuracy(temporal_shuffled)
            ),
        },
        "evidence_removal": {
            "relevant_mask_accuracy_drop": relevant_drop,
            "random_mask_accuracy_drop": random_drop,
            "relevant_minus_random_drop": relevant_drop - random_drop,
        },
    }


def write_jsonl(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return destination


def write_json(
    payload: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
