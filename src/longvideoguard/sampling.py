from __future__ import annotations

import numpy as np


def uniform_sample_indices(total_frames: int, num_samples: int) -> list[int]:
    """Return deterministic, uniformly spaced frame indices."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    sample_count = min(total_frames, num_samples)
    if sample_count == 1:
        return [0]

    indices = np.linspace(0, total_frames - 1, sample_count)
    return sorted({int(round(index)) for index in indices})


def indices_to_timestamps(indices: list[int], fps: float) -> list[float]:
    """Convert frame indices into timestamps in seconds."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if any(index < 0 for index in indices):
        raise ValueError("frame indices must be non-negative")
    return [index / fps for index in indices]
