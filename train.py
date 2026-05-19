from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parent / "4_学習" / "wm_cnn_color_v5.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    sys.argv = [str(script_path)] + sys.argv[1:]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
