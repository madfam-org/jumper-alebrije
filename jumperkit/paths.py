"""Where things live: this repository, KingKong's checkouts beside it, outputs.

KingKong's `jumper` and `jumper-design` repositories are expected next to this
one (see README). Set `JUMPERKIT_WORKSPACE` to point somewhere else.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("JUMPERKIT_WORKSPACE", REPO.parent)).resolve()

JUMPER = WORKSPACE / "jumper"
DESIGN = WORKSPACE / "jumper-design"

BUNDLE = JUMPER / "out" / "bundle_example" / "jumper"
ROBOT_XML = JUMPER / "assets" / "jumper" / "jumper.xml"
VISUAL_MESHES = JUMPER / "assets" / "jumper" / "urdf" / "jumper" / "meshes" / "visual"
PROFILE = DESIGN / "robots" / "jumper" / "profile.json"

CHOREO = REPO / "choreo"
MEDIA = REPO / "media"
BUILD = REPO / "build"
DIST = REPO / "dist"
TAKES = BUILD / "takes"
TEXTURES = BUILD / "textures"


def build_dir(character_id: str) -> Path:
    """Generated outputs for one character: skin, map, previews, video."""
    return BUILD / character_id


def media_dir(character_id: str) -> Path:
    """Committed images for one character (previews embedded in its packages)."""
    return MEDIA / character_id


def use_shellflow() -> None:
    """Make jumper-design's `shellflow` package importable."""
    src = str(DESIGN / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def run_shellflow(*args: str) -> dict:
    """Run the `shellflow` CLI and return its JSON result; exit loudly on failure."""
    r = subprocess.run([sys.executable, str(DESIGN / "scripts" / "shellflow.py"), *args],
                       capture_output=True, text=True, cwd=DESIGN)
    try:
        result = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"shellflow {args[0]} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        raise SystemExit(f"shellflow {args[0]} failed ({r.returncode}):\n"
                         f"{json.dumps(result, indent=1)[:3000]}")
    return result
