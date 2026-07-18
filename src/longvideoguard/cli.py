from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from longvideoguard.sampling import indices_to_timestamps, uniform_sample_indices
from longvideoguard.video import probe_video

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def probe(video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Print video metadata as JSON."""
    metadata = probe_video(video)
    console.print_json(
        json.dumps(
            {
                "path": str(metadata.path),
                "frame_count": metadata.frame_count,
                "fps": metadata.fps,
                "width": metadata.width,
                "height": metadata.height,
                "duration_seconds": metadata.duration_seconds,
            }
        )
    )


@app.command("sample-uniform")
def sample_uniform(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    num_frames: Annotated[int, typer.Option(min=1)] = 16,
) -> None:
    """Print deterministic uniform frame indices and timestamps."""
    metadata = probe_video(video)
    indices = uniform_sample_indices(metadata.frame_count, num_frames)
    timestamps = indices_to_timestamps(indices, metadata.fps)

    rows = [
        {"sample_id": sample_id, "frame_index": index, "timestamp_seconds": timestamp}
        for sample_id, (index, timestamp) in enumerate(zip(indices, timestamps, strict=True))
    ]
    console.print_json(json.dumps(rows))


if __name__ == "__main__":
    app()
