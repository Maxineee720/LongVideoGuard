from __future__ import annotations

import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

FAILURE_TYPES = (
    "sampling_miss",
    "temporal_order_error",
    "causal_reasoning_error",
    "object_confusion",
    "action_confusion",
    "language_prior_guess",
    "ambiguous_question",
    "annotation_issue",
    "other",
)


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSONL file not found: {source}")

    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
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
        raise ValueError(f"No records found in {source}.")
    return rows


def read_uniform_video_frames(
    video_path: str | Path,
    *,
    num_frames: int,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Decode deterministic uniformly sampled RGB frames from a local video."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    # Imported lazily so pure review/report utilities remain lightweight.
    from longvideoguard.sampling import (
        indices_to_timestamps,
        uniform_sample_indices,
    )
    from longvideoguard.video import probe_video

    metadata = probe_video(video_path)
    indices = uniform_sample_indices(metadata.frame_count, num_frames)
    timestamps = indices_to_timestamps(indices, metadata.fps)

    capture = cv2.VideoCapture(str(Path(video_path).expanduser().resolve()))
    frames: list[np.ndarray] = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise ValueError(
                    f"Could not decode frame {frame_index} from {video_path}"
                )
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    return frames, indices, timestamps


def _resize_preserving_aspect(
    frame_rgb: np.ndarray,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        frame_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return canvas


def make_contact_sheet(
    frames_rgb: Sequence[np.ndarray],
    *,
    frame_indices: Sequence[int],
    timestamps: Sequence[float],
    columns: int = 4,
    tile_width: int = 320,
    tile_height: int = 220,
) -> np.ndarray:
    """Create an RGB contact sheet with frame index and timestamp labels."""
    if not frames_rgb:
        raise ValueError("frames_rgb must be non-empty")
    if not (
        len(frames_rgb) == len(frame_indices) == len(timestamps)
    ):
        raise ValueError("frames, frame_indices, and timestamps must align")
    if columns <= 0 or tile_width <= 0 or tile_height <= 0:
        raise ValueError("sheet dimensions must be positive")

    label_height = 34
    rows = math.ceil(len(frames_rgb) / columns)
    sheet = np.full(
        (rows * (tile_height + label_height), columns * tile_width, 3),
        255,
        dtype=np.uint8,
    )

    for position, (frame, frame_index, timestamp) in enumerate(
        zip(frames_rgb, frame_indices, timestamps, strict=True)
    ):
        row = position // columns
        column = position % columns
        x0 = column * tile_width
        y0 = row * (tile_height + label_height)

        tile = _resize_preserving_aspect(
            frame,
            target_width=tile_width,
            target_height=tile_height,
        )
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile

        label = f"#{frame_index}  t={timestamp:.2f}s"
        cv2.putText(
            sheet,
            label,
            (x0 + 8, y0 + tile_height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return sheet


def write_rgb_image(image_rgb: np.ndarray, output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(destination),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )
    if not ok:
        raise ValueError(f"Could not write image: {destination}")
    return destination


def build_review_rows(
    error_rows: Sequence[Mapping[str, object]],
    *,
    contact_sheet_by_video: Mapping[str, str],
) -> list[dict[str, object]]:
    """Create a manual-review template without inventing failure labels."""
    review_rows: list[dict[str, object]] = []
    for row in error_rows:
        video_id = str(row["video_id"])
        review_rows.append(
            {
                "sample_id": row["sample_id"],
                "video_id": video_id,
                "question_category": row.get("question_category"),
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "options": row.get("options"),
                "gold_answer_letter": row.get("gold_answer_letter"),
                "predicted_letter": row.get("predicted_letter"),
                "raw_output": row.get("raw_output"),
                "contact_sheet": contact_sheet_by_video[video_id],
                "evidence_covered": None,
                "failure_type": None,
                "attribution_confidence": None,
                "notes": "",
                "proposed_fix": "",
            }
        )
    return review_rows


def write_jsonl(rows: Iterable[Mapping[str, object]], output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("Refusing to write an empty JSONL file.")
    return destination


def write_review_html(
    review_rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    title: str = "LongVideoGuard Failure Review",
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    for row in review_rows:
        options = row.get("options")
        if isinstance(options, list):
            options_html = "<ol type='A'>" + "".join(
                f"<li>{html.escape(str(option))}</li>"
                for option in options
            ) + "</ol>"
        else:
            options_html = "<p>No options available.</p>"

        image_path = html.escape(str(row["contact_sheet"]))
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(str(row['sample_id']))}</h2>
              <p><b>Category:</b> {html.escape(str(row.get('question_category')))}</p>
              <p><b>Question:</b> {html.escape(str(row.get('question')))}</p>
              {options_html}
              <p><b>Gold:</b> {html.escape(str(row.get('gold_answer_letter')))}
                 &nbsp; <b>Prediction:</b> {html.escape(str(row.get('predicted_letter')))}</p>
              <img src="{image_path}" alt="uniform sampled frames">
              <p><b>Manual fields:</b> evidence_covered, failure_type,
                 attribution_confidence, notes, proposed_fix</p>
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: sans-serif; max-width: 1400px; margin: 0 auto; padding: 24px; }}
.card {{ border: 1px solid #ccc; border-radius: 12px; padding: 18px; margin: 20px 0; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
ol {{ columns: 2; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p>Allowed failure types: <code>{", ".join(FAILURE_TYPES)}</code></p>
{"".join(cards)}
</body>
</html>
"""
    destination.write_text(document, encoding="utf-8")
    return destination


def validate_review_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate completed manual labels and return a coverage summary."""
    labelled = 0
    incomplete_sample_ids: list[str] = []
    failure_counts: Counter[str] = Counter()

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        failure_type = row.get("failure_type")
        evidence_covered = row.get("evidence_covered")
        confidence = row.get("attribution_confidence")

        complete = (
            failure_type in FAILURE_TYPES
            and isinstance(evidence_covered, bool)
            and confidence in {"low", "medium", "high"}
        )
        if complete:
            labelled += 1
            failure_counts[str(failure_type)] += 1
        else:
            incomplete_sample_ids.append(sample_id)

    return {
        "total": len(rows),
        "labelled": labelled,
        "completion_rate": labelled / len(rows) if rows else None,
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "incomplete_sample_ids": incomplete_sample_ids,
    }
