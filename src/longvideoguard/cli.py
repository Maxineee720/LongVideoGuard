from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from longvideoguard.datasets.nextqa import (
    build_video_level_pilot,
    compute_nextqa_stats,
    load_nextqa_csv,
    load_video_id_map,
    write_nextqa_manifest,
    write_stats,
)
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


@app.command("nextqa-stats")
def nextqa_stats(
    annotations: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Validate a NExT-QA CSV split and print dataset statistics."""
    records = load_nextqa_csv(annotations)
    stats = compute_nextqa_stats(records)
    console.print_json(json.dumps(stats))

    if output is not None:
        destination = write_stats(stats, output)
        console.print(f"[green]Saved statistics:[/green] {destination}")


@app.command("nextqa-build-pilot")
def nextqa_build_pilot(
    annotations: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    split: Annotated[str, typer.Option()] = "val",
    num_videos: Annotated[int, typer.Option(min=1)] = 12,
    max_questions_per_video: Annotated[int | None, typer.Option(min=1)] = 4,
    seed: Annotated[int, typer.Option()] = 42,
    video_id_map: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Build a reproducible, video-level NExT-QA pilot JSONL manifest."""
    records = load_nextqa_csv(annotations)
    pilot = build_video_level_pilot(
        records,
        num_videos=num_videos,
        seed=seed,
        max_questions_per_video=max_questions_per_video,
    )
    mapping = load_video_id_map(video_id_map)
    manifest_path = write_nextqa_manifest(
        pilot,
        output,
        split=split,
        video_id_map=mapping,
    )

    stats = compute_nextqa_stats(pilot)
    stats_path = manifest_path.with_suffix(".stats.json")
    write_stats(stats, stats_path)

    console.print(f"[green]Pilot manifest:[/green] {manifest_path}")
    console.print(f"[green]Pilot statistics:[/green] {stats_path}")
    console.print_json(json.dumps(stats))


if __name__ == "__main__":
    app()
