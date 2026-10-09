"""Package the Día de Muertos plaza as a `.map` with KingKong's jumper-design tools.

1. Export the plaza world (`dia_de_muertos.world()`, no robot) as `scene.xml`
   with its meshes as OBJ files and its textures as PNG files under `assets/`.
2. Write a `kk-scene-package/1` manifest (file inventory, counts, spawn).
3. `shellflow.map_package.export_environment` upgrades it to `kk-scene-package/2`
   with the alebrije skin as the bundled default robot.
4. `shellflow verify-package --mujoco` checks the result.

    python scene/build_map.py [--skin skin/build/alebrije-jumper.skin]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parents[1]
ROOT = A.parent
DESIGN = ROOT / "jumper-design"
PROFILE = DESIGN / "robots" / "jumper" / "profile.json"
sys.path[:0] = [str(A / "scene"), str(DESIGN / "src")]
from dia_de_muertos import world  # noqa: E402

MAP_ID = "dia-de-muertos-plaza"
TITLE = "Day of the Dead Plaza"
DESCRIPTION = ("A Mexican Day of the Dead plaza at dusk: a marigold petal path and ring, "
               "a three-tier ofrenda with candles, sugar skulls and pan de muerto, "
               "papel picado overhead and bougainvillea on a cobalt adobe wall.")
OUT = A / "scene" / "build"


def _obj(vertices: np.ndarray, faces: np.ndarray, texcoords: np.ndarray | None) -> bytes:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    if texcoords is not None:
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in texcoords]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in faces]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return ("\n".join(lines) + "\n").encode()


def export_world() -> dict[str, bytes]:
    spec = world()
    root = ET.fromstring(spec.to_xml())
    files: dict[str, bytes] = {}
    # Hosts own the physics options; a map may not carry them.
    for tag in ("option", "size", "statistic", "extension", "include"):
        for e in root.findall(tag):
            root.remove(e)
    root.set("model", MAP_ID)
    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        v = np.array(mesh.get("vertex").split(), float).reshape(-1, 3)
        f = np.array(mesh.get("face").split(), int).reshape(-1, 3)
        tc = mesh.get("texcoord")
        tc = np.array(tc.split(), float).reshape(-1, 2) if tc else None
        if tc is not None:
            tc[:, 1] = 1.0 - tc[:, 1]  # OBJ v runs up; MuJoCo flips it back on load
        path = f"assets/{name}.obj"
        files[path] = _obj(v, f, tc)
        for k in ("vertex", "face", "texcoord", "normal"):
            mesh.attrib.pop(k, None)
        mesh.set("file", path)
    for tex in asset.findall("texture"):
        src = Path(tex.get("file"))
        path = f"assets/{src.name}"
        files[path] = src.read_bytes()
        tex.set("file", path)
    ET.indent(root)
    files["scene.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return files


def manifest(files: dict[str, bytes]) -> dict:
    root = ET.fromstring(files["scene.xml"])
    wb = root.findall("worldbody")
    asset = root.find("asset")
    counts = {
        "geoms": sum(1 for w in wb for _ in w.iter("geom")),
        "bodies": sum(1 for w in wb for _ in w.iter("body")),
        "lights": sum(1 for w in wb for _ in w.iter("light")),
        "textures": sum(1 for _ in asset.iter("texture")),
        "materials": sum(1 for _ in asset.iter("material")),
        "meshes": sum(1 for _ in asset.iter("mesh")),
    }
    return {
        "schema": "kk-scene-package/1",
        "id": MAP_ID,
        "title": TITLE,
        "description": DESCRIPTION,
        "version": "1.0.0",
        "use": "watching",
        "exportedBy": {"repo": "jumper-alebrije", "tool": "scene/build_map.py"},
        "source": {"repository": "jumper-alebrije", "authoredGround": True},
        "requiredCapabilities": ["rigid"],
        "world": {"file": "scene.xml", "assetDir": "", "terrainType": "plane",
                  "ground": {"geom": "floor"}, "counts": counts, "attach": {"prefix": "scn_"}},
        "props": [], "flex": [],
        "lights": [{"name": l.get("name")} for l in root.iter("light") if l.get("name")],
        "cameras": [],
        "spawn": {"position": [0.0, 0.0, 0.0], "yaw": 0},
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in sorted(files.items())},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skin", type=Path, default=A / "skin" / "build" / f"alebrije-jumper.skin")
    p.add_argument("--preview", type=Path, default=A / "media" / "map-preview.png")
    a = p.parse_args()

    from shellflow.map_package import export_environment
    from shellflow.web_appearance import complete_appearance

    OUT.mkdir(parents=True, exist_ok=True)
    env = export_world()
    env["scene-package.json"] = json.dumps(manifest(env), indent=1).encode()
    # Give every visible geom a named material for the BE UNLIMITED web viewer.
    env = complete_appearance(env)
    m = json.loads(env["scene-package.json"])
    m["files"] = {name: {"bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()}
                  for name, d in sorted(env.items()) if name != "scene-package.json"}
    root = ET.fromstring(env["scene.xml"])
    m["world"]["counts"]["materials"] = sum(1 for _ in root.find("asset").iter("material"))
    env["scene-package.json"] = json.dumps(m, indent=1).encode()

    out = OUT / f"{MAP_ID}.map"
    out.unlink(missing_ok=True)
    result = export_environment(env, out, default_skin=a.skin, profile=PROFILE,
                                preview=a.preview if a.preview.is_file() else None)
    print("export:", json.dumps(result)[:400])
    r = subprocess.run([sys.executable, str(DESIGN / "scripts" / "shellflow.py"), "verify-package",
                        str(out), "--profile", str(PROFILE), "--capability", "rigid",
                        "--capability", "jumper", "--mujoco"],
                       capture_output=True, text=True, cwd=DESIGN)
    print("verify:", r.returncode, r.stdout[:600], r.stderr[-1500:])


if __name__ == "__main__":
    main()
