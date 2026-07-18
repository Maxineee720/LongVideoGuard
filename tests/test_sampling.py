import pytest

from longvideoguard.sampling import indices_to_timestamps, uniform_sample_indices


def test_uniform_sampling_includes_boundaries() -> None:
    assert uniform_sample_indices(total_frames=100, num_samples=5) == [0, 25, 50, 74, 99]


def test_uniform_sampling_caps_at_video_length() -> None:
    assert uniform_sample_indices(total_frames=3, num_samples=10) == [0, 1, 2]


def test_uniform_sampling_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        uniform_sample_indices(total_frames=0, num_samples=4)
    with pytest.raises(ValueError):
        uniform_sample_indices(total_frames=10, num_samples=0)


def test_indices_to_timestamps() -> None:
    assert indices_to_timestamps([0, 10, 25], fps=5.0) == [0.0, 2.0, 5.0]


def test_indices_to_timestamps_rejects_invalid_fps() -> None:
    with pytest.raises(ValueError):
        indices_to_timestamps([0], fps=0.0)
