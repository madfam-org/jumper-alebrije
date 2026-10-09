"""Alebrije look for Jumper: palette, per-part colors and shell decorations.

Alebrijes are the painted fantastical creatures of Oaxaca and Mexico City
folk art: saturated lacquer colors, dots and rings, zigzags, spines, wings.
This module builds that look as display-only geometry on `upper_shell_link`,
grouped one mesh per color so it maps onto multi-color printing slots, and
recolors the remaining visual parts.

Everything is in the `upper_shell_link` body frame, in meters, as in
`jumper/assets/jumper/jumper.xml`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
VISUAL = ROOT / "jumper" / "assets" / "jumper" / "urdf" / "jumper" / "meshes" / "visual"


def hex_rgba(h: str) -> list[float]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] + [1.0]


PALETTE = {
    "rosa": "#E4007C",      # rosa mexicano
    "turquesa": "#00B3A6",
    "amarillo": "#FFC20E",
    "naranja": "#FF6A13",
    "cobalto": "#1F4FD8",
    "lima": "#7DC242",
    "morado": "#6A2FB8",
    "blanco": "#FFF6E0",
    "negro": "#141414",
}
RGBA = {k: hex_rgba(v) for k, v in PALETTE.items()}

SHELL_COLOR = "rosa"
# The shell is one multi-material print in the .skin format, which carries at
# most four filament colors: the shell and its decorations share these four.
SHELL_SLOTS = ("rosa", "amarillo", "turquesa", "negro")
BASE_COLOR = "morado"

# Per-part colors for the rest of the robot, by link-name suffix. The front legs
# are arms (shoulder, upper arm, forearm, palm, finger); the other four are
# legs (hip, thigh, calf, foot tip).
PART_COLORS = {
    "base_link": BASE_COLOR,
    "shoulder_link": "amarillo",
    "upper_arm_link": "turquesa",
    "forearm_link": "naranja",
    "palm_link": "cobalto",
    "palm_pad_f_link": "amarillo",
    "palm_pad_b_link": "amarillo",
    "finger_link": "lima",
    "finger_tip_link": "amarillo",
    "finger_grip_insert_link": "blanco",
    "palm_grip_insert_link": "blanco",
    "hip_link": "amarillo",
    "thigh_link": "turquesa",
    "calf_link": "naranja",
    "foot_tip_link": "lima",
}
# The middle legs swap two colors so neighbouring legs never match.
MIDDLE_SWAP = {"turquesa": "cobalto", "naranja": "lima", "lima": "amarillo"}


def part_color(link: str) -> str | None:
    if link == "base_link":
        return BASE_COLOR
    leg, _, rest = link.partition("_")
    color = PART_COLORS.get(rest)
    if color and leg in ("LM", "RM"):
        color = MIDDLE_SWAP.get(color, color)
    return color


# ── geometry helpers ─────────────────────────────────────────────────────────

def _align(mesh: trimesh.Trimesh, normal: np.ndarray, point: np.ndarray,
           spin: float = 0.0) -> trimesh.Trimesh:
    """Rotate a +Z-up primitive so +Z follows `normal`, then move it to `point`."""
    m = mesh.copy()
    if spin:
        m.apply_transform(trimesh.transformations.rotation_matrix(spin, [0, 0, 1]))
    z = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    axis = np.cross(z, n)
    if np.linalg.norm(axis) > 1e-9:
        angle = math.acos(max(-1.0, min(1.0, float(z @ n))))
        m.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))
    m.apply_translation(point)
    return m


def _disc(radius: float, thickness: float) -> trimesh.Trimesh:
    d = trimesh.creation.cylinder(radius=radius, height=thickness, sections=40)
    d.apply_translation([0, 0, thickness / 2 - thickness * 0.35])
    return d


def _ring(r_in: float, r_out: float, thickness: float) -> trimesh.Trimesh:
    a = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=thickness, sections=48)
    a.apply_translation([0, 0, thickness / 2 - thickness * 0.35])
    return a


def _spine(radius: float, height: float, lean: float) -> trimesh.Trimesh:
    c = trimesh.creation.cone(radius=radius, height=height, sections=32)
    c.apply_translation([0, 0, -height * 0.12])
    c.apply_transform(trimesh.transformations.rotation_matrix(lean, [0, 1, 0]))
    return c


def _feather(length: float, width: float, thick: float) -> trimesh.Trimesh:
    f = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    f.apply_scale([width / 2, thick / 2, length / 2])
    f.apply_translation([0, 0, length / 2])
    return f


class Shell:
    """The upper shell, with surface queries from above."""

    def __init__(self) -> None:
        self.mesh = trimesh.load(VISUAL / "upper_shell_link.stl")
        self.ray = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

    def top(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray] | None:
        loc, _, tri = self.ray.intersects_location(
            [[x, y, 0.2]], [[0.0, 0.0, -1.0]], multiple_hits=False)
        if len(loc) == 0:
            return None
        n = self.mesh.face_normals[tri[0]]
        if n[2] < 0:
            n = -n
        return loc[0], n


def decorations(shell: Shell | None = None) -> dict[str, trimesh.Trimesh]:
    """Shell decorations, one mesh per color, using only `SHELL_SLOTS`."""
    shell = shell or Shell()
    parts: dict[str, list[trimesh.Trimesh]] = {}

    def add(color: str, m: trimesh.Trimesh) -> None:
        parts.setdefault(color, []).append(m)

    cx = -0.012  # visual centre of the dome, behind the face
    # Rings of dots around the centre, colors cycling per ring.
    rings = [
        # (rx, ry, count, dot radius, colors, ring outline color or None)
        (0.030, 0.026, 8, 0.0062, ["amarillo", "turquesa"], None),
        (0.052, 0.046, 12, 0.0072, ["amarillo"], "negro"),
        (0.074, 0.066, 18, 0.0050, ["turquesa"], None),
    ]
    for ri, (rx, ry, count, r, colors, outline) in enumerate(rings):
        for i in range(count):
            a = 2 * math.pi * (i + 0.5 * (ri % 2)) / count
            x, y = cx + rx * math.cos(a), ry * math.sin(a)
            if x > 0.058:  # keep clear of the face and the eyestalks
                continue
            hit = shell.top(x, y)
            if hit is None or hit[1][2] < 0.45:
                continue
            p, n = hit
            add(colors[i % len(colors)], _align(_disc(r, 0.0022), n, p))
            if outline:
                add(outline, _align(_ring(r * 1.05, r * 1.55, 0.0016), n, p))
    # A centre medallion: sun of concentric rings.
    hit = shell.top(cx, 0.0)
    if hit:
        p, n = hit
        add("amarillo", _align(_disc(0.013, 0.0026), n, p))
        add("negro", _align(_ring(0.0135, 0.0175, 0.0020), n, p))
        add("turquesa", _align(_disc(0.0055, 0.0032), n, p))

    # The crest: spines down the midline, leaning back, tallest in the middle.
    spine_colors = ["amarillo", "turquesa"]
    xs = np.linspace(-0.088, -0.040, 5)
    for i, x in enumerate(xs):
        hit = shell.top(x, 0.0)
        if hit is None:
            continue
        p, n = hit
        h = 0.020 + 0.012 * math.sin(math.pi * (i + 0.5) / len(xs))
        add(spine_colors[i % len(spine_colors)],
            _align(_spine(0.0065, h, math.radians(-28)), n, p))

    # The zigzag crown: teeth around the rim, alternating colors.
    teeth = 34
    for i in range(teeth):
        a = 2 * math.pi * i / teeth
        x, y = cx + 0.088 * math.cos(a), 0.083 * math.sin(a)
        if x > 0.062:
            continue
        hit = shell.top(x, y)
        if hit is None:
            continue
        p, n = hit
        out = np.array([math.cos(a), math.sin(a), 0.0])
        tooth_n = n + 0.6 * out
        add("amarillo" if i % 2 else "turquesa",
            _align(_spine(0.0055, 0.011, 0.0), tooth_n, p - 0.001 * n))

    # Wings: a fan of feathers each side at the back, rising up and out.
    feather_colors = ["turquesa", "amarillo", "rosa", "turquesa", "amarillo", "negro"]
    for side in (1, -1):
        hit = shell.top(-0.058, 0.052 * side)
        if hit is None:
            continue
        p, _ = hit
        for k, color in enumerate(feather_colors):
            f = _feather(0.078 - 0.006 * k, 0.019, 0.0045)
            # Fan in the plane that leans out from the body: from pointing back
            # (k=0) to pointing up (last), then tipped outwards.
            f.apply_transform(trimesh.transformations.rotation_matrix(
                math.radians(-78 + 15 * k), [0, 1, 0]))
            f.apply_transform(trimesh.transformations.rotation_matrix(
                side * math.radians(-38), [1, 0, 0]))
            f.apply_translation(p + np.array([0.0, 0.0, -0.003]))
            add(color, f)
        # A painted dot at each feather root.
        add("amarillo", _align(_disc(0.0075, 0.003), np.array([0, 0.35 * side, 1.0]), p))

    # Eyestalks above the face: stalk, big yellow alebrije eye, black pupil, lid.
    for side in (1, -1):
        hit = shell.top(0.040, 0.030 * side)
        if hit is None:
            continue
        p, _ = hit
        stalk_h = 0.026
        stalk = trimesh.creation.cylinder(radius=0.0045, height=stalk_h, sections=24)
        stalk.apply_translation(p + [0.004, 0.0, stalk_h / 2 - 0.002])
        add("turquesa", stalk)
        eye_c = p + np.array([0.006, 0.0, stalk_h + 0.006])
        eye = trimesh.creation.icosphere(subdivisions=3, radius=0.0115)
        eye.apply_translation(eye_c)
        add("amarillo", eye)
        pupil = trimesh.creation.icosphere(subdivisions=3, radius=0.0058)
        pupil.apply_translation(eye_c + np.array([0.0082, 0.0012 * side, 0.0012]))
        add("negro", pupil)
        lid = trimesh.creation.torus(major_radius=0.0098, minor_radius=0.0026,
                                     major_sections=32, minor_sections=12)
        lid.apply_transform(trimesh.transformations.rotation_matrix(math.radians(70), [0, 1, 0]))
        lid.apply_translation(eye_c + np.array([0.0005, 0.0, 0.0035]))
        add("turquesa", lid)

    assert set(parts) <= set(SHELL_SLOTS), set(parts) - set(SHELL_SLOTS)
    return {c: trimesh.util.concatenate(ms) for c, ms in parts.items()}


def apply_to_spec(spec, decor: dict[str, trimesh.Trimesh] | None = None,
                  gloss: bool = True) -> None:
    """Recolor a robot MjSpec (before attach) and add the shell decorations.

    Visual-only: every added geom has contype=conaffinity=0 and density 0, so
    mass, inertia and collision stay the baseline's.
    """
    import mujoco

    for name, rgba in RGBA.items():
        spec.add_material(name=f"alebrije_{name}", rgba=rgba,
                          specular=0.55 if gloss else 0.2,
                          shininess=0.7 if gloss else 0.3, reflectance=0.0)
    for g in spec.geoms:
        if not g.name.endswith("_visual"):
            continue
        link = g.name[: -len("_visual")]
        color = SHELL_COLOR if link == "upper_shell_link" else part_color(link)
        if color:
            g.material = f"alebrije_{color}"
            g.rgba = RGBA[color]
    body = spec.body("upper_shell_link")
    for color, mesh in (decor or decorations()).items():
        spec.add_mesh(name=f"alebrije_decor_{color}",
                      uservert=np.asarray(mesh.vertices, dtype=float).ravel().tolist(),
                      userface=np.asarray(mesh.faces, dtype=int).ravel().tolist())
        body.add_geom(name=f"alebrije_decor_{color}", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=f"alebrije_decor_{color}", material=f"alebrije_{color}",
                      contype=0, conaffinity=0, density=0, group=2)


if __name__ == "__main__":
    out = decorations()
    for c, m in out.items():
        print(f"{c:9s} faces={len(m.faces):6d} bounds={np.round(m.bounds, 3).tolist()}")
