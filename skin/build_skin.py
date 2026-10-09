"""Build `alebrije-jumper.skin` with KingKong's own jumper-design toolchain.

1. The new upper shell = the original upper shell + the alebrije decorations,
   written as one STL in the profile's original CAD millimeter frame.
2. A Bambu-style `ams.3mf` beside it paints every triangle with one of the four
   shell colors, triangle for triangle, as `shellflow.ams_colors` reads it.
3. `shellflow.simulation.assemble` exports the whole robot (MJCF + URDF) with
   that shell and the limb color overrides.
4. `shellflow export-skin` packages it; `verify-package --mujoco` checks it
   against the trusted Jumper profile.

The result is a display-only `kk-skin-package/3`: collision, inertia and joints
stay the baseline's. It is not a printable shell: no mounting interface was
engineered and nothing was fit-tested.

    python skin/build_skin.py [--version 1.0.0]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh

A = Path(__file__).resolve().parents[1]
ROOT = A.parent
DESIGN = ROOT / "jumper-design"
PROFILE = DESIGN / "robots" / "jumper" / "profile.json"
sys.path[:0] = [str(A / "design"), str(DESIGN / "src")]
from alebrije import PALETTE, RGBA, SHELL_SLOTS, Shell, decorations, part_color  # noqa: E402

OUT = A / "skin" / "build"
SKIN_ID = "alebrije-jumper"
TITLE = "Alebrije Jumper"
PAINT = ["4", "8", "0C", "1C"]  # slot 0..3, as shellflow.ams_colors.PAINT_CODES


def shell_in_cad_mm() -> tuple[trimesh.Trimesh, np.ndarray]:
    """The new shell and each face's color slot, in CAD millimeters."""
    profile = json.loads(PROFILE.read_text())
    tf = profile["cad_mm_to_parent_m"]
    R = np.asarray(tf["rotation_matrix"])
    t = np.asarray(tf["translation_m"])
    shell = Shell()
    pieces = [(shell.mesh, 0)]
    for color, mesh in decorations(shell).items():
        pieces.append((mesh, SHELL_SLOTS.index(color)))
    # Slot order keeps each color contiguous: shell first, then by slot.
    pieces.sort(key=lambda p: p[1])
    meshes, slots = [], []
    for mesh, slot in pieces:
        m = mesh.copy()
        # p_parent = R @ (p_mm * s) + t  ->  p_mm = R.T @ (p_parent - t) / s
        m.vertices = ((np.asarray(m.vertices) - t) @ R) / tf["scale"]
        meshes.append(m)
        slots.append(np.full(len(m.faces), slot, dtype=np.uint8))
    return trimesh.util.concatenate(meshes), np.concatenate(slots)


def write_ams(path: Path, mesh: trimesh.Trimesh, slots: np.ndarray) -> None:
    """A minimal Bambu Studio style 3MF: one component, painted triangles."""
    core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    prod = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
    verts = "\n".join(f'<vertex x="{x!r}" y="{y!r}" z="{z!r}"/>' for x, y, z in mesh.vertices.tolist())
    tris = "\n".join(f'<triangle v1="{a}" v2="{b}" v3="{c}" paint_color="{PAINT[s]}"/>'
                     for (a, b, c), s in zip(mesh.faces.tolist(), slots.tolist()))
    obj = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xmlns="{core}">'
           f'<resources><object id="1" type="model"><mesh><vertices>\n{verts}\n</vertices>'
           f'<triangles>\n{tris}\n</triangles></mesh></object></resources></model>\n')
    main = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xmlns="{core}" '
            f'xmlns:p="{prod}" requiredextensions="p"><resources><object id="2" type="model">'
            f'<components><component p:path="/3D/Objects/alebrije_shell.model" objectid="1" '
            f'transform="1 0 0 0 1 0 0 0 1 0 0 0"/></components></object></resources>'
            f'<build><item objectid="2"/></build></model>\n')
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/'
                     'package/2006/content-types"><Default Extension="rels" ContentType="application/'
                     'vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" '
                     'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>\n')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>\n')
    settings = {"filament_colour": [PALETTE[c] for c in SHELL_SLOTS]}
    members = [("[Content_Types].xml", content_types), ("_rels/.rels", rels),
               ("3D/3dmodel.model", main), ("3D/Objects/alebrije_shell.model", obj),
               ("Metadata/project_settings.config", json.dumps(settings))]
    with zipfile.ZipFile(path, "w") as z:
        for name, text in members:
            # A fixed timestamp keeps the archive, and every hash recorded from it,
            # identical from build to build.
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, text)


def visual_overrides() -> list[dict]:
    import mujoco
    baseline = json.loads(PROFILE.read_text())["baseline_mjcf"]
    spec = mujoco.MjSpec.from_file(str(PROFILE.parent / baseline))
    out = []
    for g in spec.geoms:
        if not g.name.endswith("_visual") or g.name == "upper_shell_link_visual":
            continue
        link = g.name[: -len("_visual")]
        color = part_color(link)
        if color:
            out.append({"link": link, "geom": g.name, "rgba": RGBA[color]})
    return out


def shellflow(*args: str) -> dict:
    r = subprocess.run([sys.executable, str(DESIGN / "scripts" / "shellflow.py"), *args],
                       capture_output=True, text=True, cwd=DESIGN)
    try:
        result = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"shellflow {args[0]} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        raise SystemExit(f"shellflow {args[0]} failed ({r.returncode}):\n{json.dumps(result, indent=1)[:3000]}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--author", default="MADFAM (Innovaciones MADFAM S.A.S. de C.V.)")
    p.add_argument("--license", default="Apache-2.0")
    p.add_argument("--preview", type=Path, default=A / "media" / "skin-preview.png")
    a = p.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "print").mkdir(parents=True)
    mesh, slots = shell_in_cad_mm()
    stl = OUT / "print" / "shell.stl"
    mesh.export(stl)
    # Paint against the mesh exactly as the exporter will load it.
    loaded = trimesh.load_mesh(stl, process=True)
    assert len(loaded.faces) == len(slots), (len(loaded.faces), len(slots))
    write_ams(OUT / "print" / "ams.3mf", loaded, slots)
    print(f"shell: {len(loaded.faces)} faces, extents {np.round(loaded.extents, 1).tolist()} mm")

    from shellflow.ams_colors import colors_from_ams
    _, palette, report = colors_from_ams(OUT / "print" / "ams.3mf", loaded)
    print("ams check:", palette, report["matched_faces"], "faces,",
          f"residual {report['max_coordinate_residual_mm']:.2g} mm")

    from shellflow.simulation import assemble
    assemble(stl, OUT / "simulation", PROFILE, visual_overrides=visual_overrides())
    sim_report = json.loads((OUT / "simulation" / "simulation-report.json").read_text())
    print("assembled:", {k: sim_report[k] for k in ("whole_robot",) if k in sim_report},
          "colors:", sim_report.get("color", {}).get("mode") if isinstance(sim_report.get("color"), dict) else "?")

    skin = OUT / f"{SKIN_ID}.skin"
    args = ["export-skin", "--simulation", str(OUT / "simulation"), "--profile", str(PROFILE),
            "--id", SKIN_ID, "--title", TITLE, "--version", a.version,
            "--author", a.author, "--license", a.license, "--output", str(skin)]
    if a.preview.is_file():
        args += ["--preview", str(a.preview)]
    print("export:", json.dumps(shellflow(*args))[:400])
    verify = shellflow("verify-package", str(skin), "--profile", str(PROFILE), "--mujoco")
    print("verify:", json.dumps(verify)[:600])


if __name__ == "__main__":
    main()
