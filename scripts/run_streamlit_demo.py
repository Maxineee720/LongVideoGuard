from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the LongVideoGuard Streamlit demo."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
    )
    parser.add_argument(
        "--address",
        default="0.0.0.0",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not existing
        else source_path + os.pathsep + existing
    )

    command = [
        "streamlit",
        "run",
        str(project_root / "streamlit_app.py"),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
        "--server.headless",
        "true",
    ]
    return subprocess.call(
        command,
        cwd=project_root,
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
