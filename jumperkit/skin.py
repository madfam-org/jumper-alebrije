"""Build a character's `.skin` with KingKong's own jumper-design toolchain.

1. The new upper shell = the original upper shell + the character's decorations,
   written as one STL in the profile's original CAD millimeter frame.
2. A Bambu-style `ams.3mf` beside it paints every triangle with one of the
   (at most four) shell colors, triangle for triangle, as
   `shellflow.ams_colors` reads it.
3. `shellflow.simulation.assemble` exports the whole robot (MJCF + URDF) with
   that shell and the character's per-part color overrides.
4. `shellflow export-skin` packages it; `verify-package --mujoco` checks it
   against the trusted Jumper profile.

The result is a display-only `kk-skin-package/3`: collision, inertia and joints
stay the baseline's. It is not a printable shell: no mounting interface was
engineered and nothing was fit-tested. Builds are byte-reproducible.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from .character import Character
from .parts import Shell
from .paths import PROFILE, build_dir, media_dir, run_shellflow, use_shellflow

PAINT = ["4", "8", "0C", "1C"]  # slot 0..3, as shellflow.ams_colors.PAINT_CODES
AUTHOR = "MADFAM (Innovaciones MADFAM S.A.S. de C.V.)"
LICENSE = "Apache-2.0"


def shell_in_cad_mm(character: Character) -> tuple[trimesh.Trimesh, np.ndarray]:
    """The new shell and each face's color slot, in CAD millimeters."""
    profile = json.loads(PROFILE.read_text())
    tf = profile["cad_mm_to_parent_m"]
    R = np.asarray(tf["rotation_matrix"])
    t = np.asarray(tf["translation_m"])
    shell = Shell()
    pieces = [(shell.mesh, 0)]
    for color, mesh in character.decorations(shell).items():
        pieces.append((mesh, character.shell_slots.index(color)))
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


def write_ams(path: Path, mesh: trimesh.Trimesh, slots: np.ndarray, colors: list[str],
              model_name: str) -> None:
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
            f'<components><component p:path="/3D/Objects/{model_name}" objectid="1" '
            f'transform="1 0 0 0 1 0 0 0 1 0 0 0"/></components></object></resources>'
            f'<build><item objectid="2"/></build></model>\n')
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/'
                     'package/2006/content-types"><Default Extension="rels" ContentType="application/'
                     'vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" '
                     'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>\n')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>\n')
    settings = {"filament_colour": colors}
    members = [("[Content_Types].xml", content_types), ("_rels/.rels", rels),
               ("3D/3dmodel.model", main), (f"3D/Objects/{model_name}", obj),
               ("Metadata/project_settings.config", json.dumps(settings))]
    with zipfile.ZipFile(path, "w") as z:
        for name, text in members:
            # A fixed timestamp keeps the archive, and every hash recorded from it,
            # identical from build to build.
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, text)


def visual_overrides(character: Character) -> list[dict]:
    """Per-part colors for every recolored baseline visual except the shell."""
    import mujoco
    baseline = json.loads(PROFILE.read_text())["baseline_mjcf"]
    spec = mujoco.MjSpec.from_file(str(PROFILE.parent / baseline))
    out = []
    for g in spec.geoms:
        if not g.name.endswith("_visual") or g.name == "upper_shell_link_visual":
            continue
        link = g.name[: -len("_visual")]
        color = character.part_color(link)
        if color:
            out.append({"link": link, "geom": g.name, "rgba": character.rgba(color)})
    return out


def skin_path(character: Character) -> Path:
    return build_dir(character.id) / "skin" / f"{character.skin_id}.skin"


def build_skin(character: Character, *, version: str | None = None, author: str = AUTHOR,
               license_id: str = LICENSE, preview: Path | None = None,
               log=print) -> Path:
    """Build, package and verify the character's skin; return the `.skin` path."""
    use_shellflow()
    from shellflow.ams_colors import colors_from_ams
    from shellflow.simulation import assemble

    out = build_dir(character.id) / "skin"
    if out.exists():
        shutil.rmtree(out)
    (out / "print").mkdir(parents=True)
    mesh, slots = shell_in_cad_mm(character)
    stl = out / "print" / "shell.stl"
    mesh.export(stl)
    # Paint against the mesh exactly as the exporter will load it.
    loaded = trimesh.load_mesh(stl, process=True)
    assert len(loaded.faces) == len(slots), (len(loaded.faces), len(slots))
    write_ams(out / "print" / "ams.3mf", loaded, slots,
              [character.palette[c] for c in character.shell_slots],
              f"{character.id}_shell.model")
    log(f"shell: {len(loaded.faces)} faces, extents {np.round(loaded.extents, 1).tolist()} mm")
    _, palette, report = colors_from_ams(out / "print" / "ams.3mf", loaded)
    log(f"ams check: {palette} {report['matched_faces']} faces, "
        f"residual {report['max_coordinate_residual_mm']:.2g} mm")

    assemble(stl, out / "simulation", PROFILE, visual_overrides=visual_overrides(character))

    skin = skin_path(character)
    preview = preview or media_dir(character.id) / "skin-preview.png"
    args = ["export-skin", "--simulation", str(out / "simulation"), "--profile", str(PROFILE),
            "--id", character.skin_id, "--title", character.title,
            "--version", version or character.version,
            "--author", author, "--license", license_id, "--output", str(skin)]
    if preview.is_file():
        args += ["--preview", str(preview)]
    exported = run_shellflow(*args)
    log(f"export: ok={exported.get('ok')} schema={exported.get('schema')} id={exported.get('id')}")
    verify_skin(skin, log=log)
    return skin


def verify_skin(path: Path, log=print) -> dict:
    result = run_shellflow("verify-package", str(path), "--profile", str(PROFILE), "--mujoco")
    log(f"verify {path.name}: ok={result.get('ok')} sha256={result.get('sha256')}")
    return result
