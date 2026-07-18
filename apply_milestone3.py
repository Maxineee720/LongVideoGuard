from __future__ import annotations

from pathlib import Path

pyproject = Path("pyproject.toml")
text = pyproject.read_text(encoding="utf-8")

old = 'vlm = [\n  "torch>=2.4",\n  "transformers>=4.57",\n  "accelerate>=1.0",\n  "qwen-vl-utils>=0.0.14",\n]\n'
new = 'vlm = [\n  "torch>=2.4",\n  "torchvision>=0.19",\n  "transformers>=4.57.0",\n  "accelerate>=1.0",\n  "qwen-vl-utils>=0.0.14",\n]\n'

if new in text:
    print("pyproject.toml already contains Milestone 3 VLM dependencies.")
elif old in text:
    pyproject.write_text(text.replace(old, new), encoding="utf-8")
    print("Updated pyproject.toml VLM dependencies.")
else:
    raise SystemExit(
        "Could not find the expected [project.optional-dependencies] "
        "vlm block. Update it manually using requirements-colab.txt."
    )
