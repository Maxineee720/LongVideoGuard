from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from longvideoguard.video_validation import load_jsonl, unique_video_rows


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def build_member_index(
    archive: zipfile.ZipFile,
) -> dict[str, list[zipfile.ZipInfo]]:
    index: dict[str, list[zipfile.ZipInfo]] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        member = PurePosixPath(info.filename)
        if member.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        index.setdefault(member.stem, []).append(info)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract only NExT-QA pilot videos from a local NExTVideo ZIP."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/nextqa/videos"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = unique_video_rows(load_jsonl(args.manifest))
    expected = {
        Path(str(row["video_relpath"])).stem: Path(str(row["video_relpath"])).name
        for row in rows
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as archive:
        member_index = build_member_index(archive)
        missing: list[str] = []
        ambiguous: list[str] = []

        for number, (stem, output_name) in enumerate(expected.items(), start=1):
            matches = member_index.get(stem, [])
            if not matches:
                missing.append(output_name)
                continue
            if len(matches) > 1:
                exact = [
                    item
                    for item in matches
                    if PurePosixPath(item.filename).name == output_name
                ]
                if len(exact) != 1:
                    ambiguous.append(output_name)
                    continue
                info = exact[0]
            else:
                info = matches[0]

            destination = args.output_dir / output_name
            if destination.exists() and not args.overwrite:
                print(f"[{number}/{len(expected)}] SKIP {destination}")
                continue

            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.unlink(missing_ok=True)
            print(f"[{number}/{len(expected)}] GET  {info.filename}")
            with archive.open(info) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary.replace(destination)
            print(f"[{number}/{len(expected)}] OK   {destination}")

    if missing:
        print("\nMissing archive members:")
        for item in missing:
            print(f"  - {item}")
    if ambiguous:
        print("\nAmbiguous archive members:")
        for item in ambiguous:
            print(f"  - {item}")

    return 1 if missing or ambiguous else 0


if __name__ == "__main__":
    raise SystemExit(main())
