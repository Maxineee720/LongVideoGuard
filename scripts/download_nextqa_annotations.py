from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/doc-doc/NExT-QA/"
    "refs/heads/main/dataset/nextqa"
)
FILES = ("train.csv", "val.csv", "test.csv", "map_vid_vidorID.json")


def download_file(url: str, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print(f"[SKIP] {destination} already exists")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[GET]  {url}")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[OK]   {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download official NExT-QA multiple-choice annotations."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/nextqa/annotations"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for filename in FILES:
        download_file(
            f"{BASE_URL}/{filename}",
            args.output_dir / filename,
            overwrite=args.overwrite,
        )

    print(
        "\nAnnotations downloaded. Raw videos are distributed separately by "
        "the dataset authors; see docs/NEXTQA_SETUP.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
