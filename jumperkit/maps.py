"""Package the Día de Muertos plaza as a `.map` with KingKong's jumper-design tools.

1. Export the plaza world (`scene.world()`, no robot) as `scene.xml` with its
   meshes as OBJ files and its textures as PNG files under `assets/`.
2. Write a `kk-scene-package/1` manifest (file inventory, counts, spawn).
3. `shellflow.map_package.export_environment` upgrades it to
   `kk-scene-package/2` with the character's skin as the bundled default robot.
4. `shellflow verify-package --mujoco` checks the result.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .character import Character
from .paths import PROFILE, build_dir, media_dir, run_shellflow, use_shellflow
from .scene import DESCRIPTION, TITLE, world

#: Written into every manifest as provenance.
EXPORTED_BY = {"repo": "jumper-alebrije", "tool": "jumperkit/maps.py"}


def _obj(vertices: np.ndarray, faces: np.ndarray, texcoords: np.ndarray | None) -> bytes:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    if texcoords is not None:
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in texcoords]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in faces]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return ("\n".join(lines) + "\n").encode()


def export_world(map_id: str) -> dict[str, bytes]:
    """The plaza as package files: `scene.xml` plus `assets/` meshes and textures."""
    spec = world()
    root = ET.fromstring(spec.to_xml())
    files: dict[str, bytes] = {}
    # Hosts own the physics options; a map may not carry them.
    for tag in ("option", "size", "statistic", "extension", "include"):
        for e in root.findall(tag):
            root.remove(e)
    root.set("model", map_id)
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


def _inventory(files: dict[str, bytes]) -> dict:
    return {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(files.items()) if name != "scene-package.json"}


def manifest(files: dict[str, bytes], map_id: str) -> dict:
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
        "id": map_id,
        "title": TITLE,
        "description": DESCRIPTION,
        "version": "1.0.0",
        "use": "watching",
        "exportedBy": EXPORTED_BY,
        "source": {"repository": "jumper-alebrije", "authoredGround": True},
        "requiredCapabilities": ["rigid"],
        "world": {"file": "scene.xml", "assetDir": "", "terrainType": "plane",
                  "ground": {"geom": "floor"}, "counts": counts, "attach": {"prefix": "scn_"}},
        "props": [], "flex": [],
        "lights": [{"name": l.get("name")} for l in root.iter("light") if l.get("name")],
        "cameras": [],
        "spawn": {"position": [0.0, 0.0, 0.0], "yaw": 0},
        "files": _inventory(files),
    }


def map_path(character: Character) -> Path:
    return build_dir(character.id) / f"{character.plaza_map_id}.map"


def build_map(character: Character, skin: Path, *, preview: Path | None = None,
              log=print) -> Path:
    """Package the plaza with `skin` as its default robot, verify; return the path."""
    use_shellflow()
    from shellflow.map_package import export_environment
    from shellflow.web_appearance import complete_appearance

    map_id = character.plaza_map_id
    env = export_world(map_id)
    env["scene-package.json"] = json.dumps(manifest(env, map_id), indent=1).encode()
    # Give every visible geom a named material for the BE UNLIMITED web viewer.
    env = complete_appearance(env)
    m = json.loads(env["scene-package.json"])
    m["files"] = _inventory(env)
    root = ET.fromstring(env["scene.xml"])
    m["world"]["counts"]["materials"] = sum(1 for _ in root.find("asset").iter("material"))
    env["scene-package.json"] = json.dumps(m, indent=1).encode()

    out = map_path(character)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    preview = preview or media_dir(character.id) / "map-preview.png"
    result = export_environment(env, out, default_skin=skin, profile=PROFILE,
                                preview=preview if preview.is_file() else None)
    log(f"export: ok={result.get('ok')} schema={result.get('schema')} id={result.get('id')}")
    verify_map(out, log=log)
    return out


def verify_map(path: Path, log=print) -> dict:
    result = run_shellflow("verify-package", str(path), "--profile", str(PROFILE),
                           "--capability", "rigid", "--capability", "jumper", "--mujoco")
    log(f"verify {path.name}: ok={result.get('ok')} sha256={result.get('sha256')}")
    return result
