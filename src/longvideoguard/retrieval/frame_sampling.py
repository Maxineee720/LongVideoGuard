from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateFrames:
    rgb_frames: list[np.ndarray]
    source_frame_indices: list[int]
    timestamps_seconds: list[float]
    source_fps: float
    total_source_frames: int

    def __post_init__(self) -> None:
        n = len(self.rgb_frames)
        if n == 0 or n != len(self.source_frame_indices) or n != len(self.timestamps_seconds):
            raise ValueError("Candidate frame fields must be non-empty and aligned.")


def safe_sample_id(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "sample"


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        raise ValueError("total and count must be positive")
    if count >= total:
        return list(range(total))
    indices = np.rint(np.linspace(0, total - 1, num=count)).astype(int).tolist()
    result: list[int] = []
    for index in indices:
        if index not in result:
            result.append(index)
    for index in range(total):
        if len(result) == count:
            break
        if index not in result:
            result.append(index)
    return sorted(result)


def frame_change_scores(frames: Sequence[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("frames must be non-empty")
    histograms = []
    for frame in frames:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("frames must be RGB HxWx3 arrays")
        channels = []
        for channel in range(3):
            hist, _ = np.histogram(array[..., channel], bins=32, range=(0, 256))
            hist = hist.astype(np.float64)
            hist /= max(hist.sum(), 1.0)
            channels.append(hist)
        histograms.append(np.concatenate(channels))
    scores = np.zeros(len(histograms), dtype=np.float64)
    for index in range(1, len(histograms)):
        scores[index] = 0.5 * np.abs(histograms[index] - histograms[index - 1]).sum()
    return scores


def _farthest_fill(selected: set[int], total: int, count: int) -> list[int]:
    while len(selected) < min(total, count):
        remaining = [i for i in range(total) if i not in selected]
        selected.add(max(remaining, key=lambda i: min(abs(i - j) for j in selected)))
    return sorted(selected)[:count]


def scene_aware_indices(scores: Sequence[float], count: int, min_gap: int = 1) -> list[int]:
    values = np.asarray(scores, dtype=np.float64)
    total = len(values)
    if total == 0 or count <= 0 or min_gap < 0:
        raise ValueError("invalid scene selection arguments")
    if count >= total:
        return list(range(total))
    selected = {0, total - 1}
    for index in np.argsort(-values).tolist():
        if len(selected) >= count:
            break
        if all(abs(index - other) >= min_gap for other in selected):
            selected.add(int(index))
    return _farthest_fill(selected, total, count)


def l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=axis, keepdims=True), 1e-12)


def cosine_scores(image_embeddings: np.ndarray, text_embedding: np.ndarray) -> np.ndarray:
    images = l2_normalize(image_embeddings, axis=1)
    text = l2_normalize(np.asarray(text_embedding).reshape(1, -1), axis=1)[0]
    if images.shape[1] != text.shape[0]:
        raise ValueError("embedding dimensions differ")
    return images @ text


def query_aware_indices(
    image_embeddings: np.ndarray,
    text_embedding: np.ndarray,
    count: int,
    relevance_weight: float = 0.8,
    min_gap: int = 1,
) -> tuple[list[int], list[float]]:
    embeddings = l2_normalize(image_embeddings, axis=1)
    total = len(embeddings)
    if total == 0 or count <= 0 or not 0 <= relevance_weight <= 1 or min_gap < 0:
        raise ValueError("invalid query-aware selection arguments")
    relevance = cosine_scores(embeddings, text_embedding)
    if count >= total:
        return list(range(total)), relevance.tolist()
    selected: list[int] = []
    remaining = set(range(total))
    while remaining and len(selected) < count:
        candidates = [i for i in remaining if all(abs(i - j) >= min_gap for j in selected)] or list(remaining)
        def score(index: int) -> float:
            if not selected:
                diversity = 1.0
            else:
                visual = 1.0 - max(float(embeddings[index] @ embeddings[j]) for j in selected)
                temporal = min(abs(index - j) / max(total - 1, 1) for j in selected)
                diversity = 0.5 * (visual + temporal)
            return relevance_weight * float(relevance[index]) + (1 - relevance_weight) * diversity
        best = max(candidates, key=score)
        selected.append(best)
        remaining.remove(best)
    return sorted(selected), relevance.tolist()


def build_query_text(row: Mapping[str, object], mode: str = "question") -> str:
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    if mode == "question":
        return question
    if mode == "question_options":
        options = row.get("options")
        if not isinstance(options, list) or len(options) != 5:
            raise ValueError("five options are required")
        suffix = " ".join(f"{letter}: {option}" for letter, option in zip("ABCDE", options, strict=True))
        return f"{question} {suffix}"
    raise ValueError("mode must be question or question_options")


def decode_uniform_candidates(video_path: str | Path, candidate_count: int) -> CandidateFrames:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless") from exc
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open {source}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total <= 0:
            raise ValueError(f"No frames in {source}")
        if not math.isfinite(fps) or fps <= 0:
            fps = 1.0
        indices = evenly_spaced_indices(total, min(candidate_count, total))
        frames, decoded, timestamps = [], [], []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if ok and bgr is not None:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                decoded.append(index)
                timestamps.append(index / fps)
        return CandidateFrames(frames, decoded, timestamps, fps, total)
    finally:
        capture.release()


def write_selected_video(frames: Sequence[np.ndarray], output_path: str | Path, fps: float = 1.0) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless") from exc
    if not frames:
        raise ValueError("frames must be non-empty")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = np.asarray(frames[0]).shape[:2]
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not create {destination}")
    try:
        for frame in frames:
            array = np.asarray(frame)
            if array.shape[:2] != (height, width):
                array = cv2.resize(array, (width, height))
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return destination


def pairwise_redundancy(embeddings: np.ndarray, indices: Sequence[int]) -> float:
    if len(indices) < 2:
        return 0.0
    normalized = l2_normalize(embeddings, axis=1)
    values = [float(normalized[a] @ normalized[b]) for i, a in enumerate(indices) for b in indices[i + 1 :]]
    return float(sum(values) / len(values))


def temporal_coverage(indices: Sequence[int], total_candidates: int) -> float:
    if not indices:
        return 0.0
    return 1.0 if total_candidates <= 1 else (max(indices) - min(indices)) / (total_candidates - 1)


class CLIPFrameEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    def encode_images(self, frames: Sequence[np.ndarray], batch_size: int = 16) -> np.ndarray:
        from PIL import Image
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                images = [Image.fromarray(np.asarray(frame).astype(np.uint8)) for frame in frames[start : start + batch_size]]
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                outputs.append(features.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def encode_text(self, text: str) -> np.ndarray:
        with self.torch.inference_mode():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].float().cpu().numpy()


def write_jsonl(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return destination
