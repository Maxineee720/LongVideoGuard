from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from longvideoguard.nextqa_prompt import (
    answer_index_to_letter,
    format_nextqa_prompt,
    parse_option_letter,
    validate_manifest_row,
)
from longvideoguard.qwen3vl_runner import Qwen3VLRunner


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            validate_manifest_row(payload)
            rows.append(payload)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def completed_sample_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id")
            if sample_id:
                completed.add(str(sample_id))
    return completed


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def select_rows(
    rows: list[dict[str, object]],
    *,
    limit: int | None,
    sample_id: str | None,
) -> Iterable[dict[str, object]]:
    selected = rows
    if sample_id is not None:
        selected = [row for row in rows if str(row["sample_id"]) == sample_id]
        if not selected:
            raise ValueError(f"sample_id not found in manifest: {sample_id}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL zero-shot inference on a NExT-QA manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/predictions/"
            "nextqa_qwen3vl2b_zero_shot_uniform16.jsonl"
        ),
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    args = parser.parse_args()

    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    manifest = args.manifest.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if args.overwrite:
        output.unlink(missing_ok=True)

    rows = load_jsonl(manifest)
    done = completed_sample_ids(output)
    selected = list(
        select_rows(rows, limit=args.limit, sample_id=args.sample_id)
    )

    print(f"Model: {args.model_name}")
    print(f"Manifest rows selected: {len(selected)}")
    print(f"Frame budget: {args.num_frames}")
    print(f"Output: {output}")

    runner = Qwen3VLRunner(
        args.model_name,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    processed = 0
    for position, row in enumerate(selected, start=1):
        sample_id = str(row["sample_id"])
        if sample_id in done:
            print(f"[{position}/{len(selected)}] SKIP {sample_id}")
            continue

        video_path = video_root / str(row["video_relpath"])
        prompt = format_nextqa_prompt(
            str(row["question"]),
            [str(option) for option in row["options"]],  # type: ignore[union-attr]
        )

        print(f"[{position}/{len(selected)}] RUN  {sample_id}")
        error: str | None = None
        raw_output = ""
        predicted_letter: str | None = None
        latency_seconds: float | None = None
        input_token_count: int | None = None
        generated_token_count: int | None = None
        peak_gpu_memory_mb: float | None = None

        try:
            result = runner.generate_video_answer(
                video_path,
                prompt,
                num_frames=args.num_frames,
                max_new_tokens=args.max_new_tokens,
            )
            raw_output = result.raw_output
            predicted_letter = parse_option_letter(raw_output)
            latency_seconds = result.latency_seconds
            input_token_count = result.input_token_count
            generated_token_count = result.generated_token_count
            peak_gpu_memory_mb = result.peak_gpu_memory_mb
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        gold_index = int(row["answer_index"])
        gold_letter = answer_index_to_letter(gold_index)
        prediction = {
            "schema_version": "1.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": {
                "model_name": args.model_name,
                "num_frames": args.num_frames,
                "dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "attn_implementation": args.attn_implementation,
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "sample_id": sample_id,
            "video_id": row["video_id"],
            "video_relpath": row["video_relpath"],
            "question": row["question"],
            "options": row["options"],
            "question_type": row.get("question_type"),
            "question_category": row["question_category"],
            "gold_answer_index": gold_index,
            "gold_answer_letter": gold_letter,
            "raw_output": raw_output,
            "predicted_letter": predicted_letter,
            "is_valid_prediction": predicted_letter is not None,
            "is_correct": (
                predicted_letter == gold_letter
                if predicted_letter is not None
                else False
            ),
            "latency_seconds": latency_seconds,
            "input_token_count": input_token_count,
            "generated_token_count": generated_token_count,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "error": error,
        }
        append_jsonl(output, prediction)
        processed += 1

        print(
            f"  output={raw_output!r} parsed={predicted_letter!r} "
            f"gold={gold_letter} error={error!r}"
        )

    print(f"New predictions written: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
