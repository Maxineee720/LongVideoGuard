from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = ("cv2", "numpy", "pydantic", "yaml", "rich", "typer")


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    failed: list[str] = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {package}: {version}")
        except Exception as exc:  # noqa: BLE001
            failed.append(package)
            print(f"[FAIL] {package}: {exc}")

    if failed:
        print(f"\nMissing or broken packages: {', '.join(failed)}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
