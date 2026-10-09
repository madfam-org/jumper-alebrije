"""Shell-decoration library: the parts characters are made of.

Every part is display-only geometry in the `upper_shell_link` body frame, in
meters (as in `jumper/assets/jumper/jumper.xml`). Parts find the shell surface
by casting rays down onto the real upper-shell mesh and orient themselves along
its normals. Each part takes the color names it paints with; `Decor` collects
the pieces and merges them into one mesh per color, which is how the `.skin`
format maps a shell onto its (at most four) filament colors.

Defaults reproduce the alebrije, the first character; other characters pass
their own colors and, where they need to, their own sizes and positions.
Characters may also build custom parts from the primitives (`disc`, `ring`,
`spine`, `feather`, `align`) and add them with `Decor.add`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import trimesh

from .paths import VISUAL_MESHES

#: The dome's visual centre: behind the face (the display module at +X).
DOME_CX = -0.012


# ── primitives (+Z up, base at the origin) ───────────────────────────────────

def align(mesh: trimesh.Trimesh, normal: np.ndarray, point: np.ndarray,
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


def disc(radius: float, thickness: float) -> trimesh.Trimesh:
    """A flat painted dot, sunk a third of its thickness into the surface."""
    d = trimesh.creation.cylinder(radius=radius, height=thickness, sections=40)
    d.apply_translation([0, 0, thickness / 2 - thickness * 0.35])
    return d


def ring(r_in: float, r_out: float, thickness: float) -> trimesh.Trimesh:
    """A flat painted ring, sunk like `disc`."""
    a = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=thickness, sections=48)
    a.apply_translation([0, 0, thickness / 2 - thickness * 0.35])
    return a


def spine(radius: float, height: float, lean: float) -> trimesh.Trimesh:
    """A cone leaning `lean` radians about Y (negative leans back, toward -X)."""
    c = trimesh.creation.cone(radius=radius, height=height, sections=32)
    c.apply_translation([0, 0, -height * 0.12])
    c.apply_transform(trimesh.transformations.rotation_matrix(lean, [0, 1, 0]))
    return c


def feather(length: float, width: float, thick: float) -> trimesh.Trimesh:
    """A flattened ellipsoid standing on its base, pointing +Z."""
    f = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    f.apply_scale([width / 2, thick / 2, length / 2])
    f.apply_translation([0, 0, length / 2])
    return f


# ── the shell and the collector ──────────────────────────────────────────────

class Shell:
    """The upper shell, with surface queries from above."""

    def __init__(self) -> None:
        self.mesh = trimesh.load(VISUAL_MESHES / "upper_shell_link.stl")
        self.ray = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

    def top(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray] | None:
        """The topmost shell point above (x, y) and its upward normal, or None."""
        loc, _, tri = self.ray.intersects_location(
            [[x, y, 0.2]], [[0.0, 0.0, -1.0]], multiple_hits=False)
        if len(loc) == 0:
            return None
        n = self.mesh.face_normals[tri[0]]
        if n[2] < 0:
            n = -n
        return loc[0], n


class Decor:
    """Collects decoration pieces by color, in the order they are added."""

    def __init__(self) -> None:
        self.pieces: dict[str, list[trimesh.Trimesh]] = {}

    def add(self, color: str, mesh: trimesh.Trimesh) -> None:
        self.pieces.setdefault(color, []).append(mesh)

    def meshes(self) -> dict[str, trimesh.Trimesh]:
        """One merged mesh per color."""
        return {c: trimesh.util.concatenate(ms) for c, ms in self.pieces.items()}


# ── parts ────────────────────────────────────────────────────────────────────

#: A ring of dots: (x radius, y radius, count, dot radius, dot colors, outline color or None).
DotRing = tuple[float, float, int, float, Sequence[str], "str | None"]


def dot_rings(d: Decor, shell: Shell, rings: Sequence[DotRing], *, cx: float = DOME_CX,
              face_clear_x: float = 0.058, min_up: float = 0.45) -> None:
    """Concentric elliptical rings of dots around the dome; odd rings are offset
    half a step. Dots cycle through their ring's colors; an outline color adds a
    painted ring around each dot. Steep spots (normal z < `min_up`) and the face
    (x > `face_clear_x`) are skipped."""
    for ri, (rx, ry, count, r, colors, outline) in enumerate(rings):
        for i in range(count):
            a = 2 * math.pi * (i + 0.5 * (ri % 2)) / count
            x, y = cx + rx * math.cos(a), ry * math.sin(a)
            if x > face_clear_x:
                continue
            hit = shell.top(x, y)
            if hit is None or hit[1][2] < min_up:
                continue
            p, n = hit
            d.add(colors[i % len(colors)], align(disc(r, 0.0022), n, p))
            if outline:
                d.add(outline, align(ring(r * 1.05, r * 1.55, 0.0016), n, p))


def medallion(d: Decor, shell: Shell, disc_color: str, ring_color: str, core_color: str,
              *, cx: float = DOME_CX) -> None:
    """A sun medallion at the dome centre: disc, ring around it, raised core."""
    hit = shell.top(cx, 0.0)
    if hit:
        p, n = hit
        d.add(disc_color, align(disc(0.013, 0.0026), n, p))
        d.add(ring_color, align(ring(0.0135, 0.0175, 0.0020), n, p))
        d.add(core_color, align(disc(0.0055, 0.0032), n, p))


def crest(d: Decor, shell: Shell, colors: Sequence[str], *, x_from: float = -0.088,
          x_to: float = -0.040, count: int = 5, radius: float = 0.0065, base: float = 0.020,
          extra: float = 0.012, lean_deg: float = -28) -> None:
    """Spines down the midline, leaning back, tallest in the middle."""
    xs = np.linspace(x_from, x_to, count)
    for i, x in enumerate(xs):
        hit = shell.top(x, 0.0)
        if hit is None:
            continue
        p, n = hit
        h = base + extra * math.sin(math.pi * (i + 0.5) / len(xs))
        d.add(colors[i % len(colors)], align(spine(radius, h, math.radians(lean_deg)), n, p))


def zigzag_crown(d: Decor, shell: Shell, colors: Sequence[str], *, teeth: int = 34,
                 cx: float = DOME_CX, rx: float = 0.088, ry: float = 0.083,
                 face_clear_x: float = 0.062, radius: float = 0.0055,
                 height: float = 0.011) -> None:
    """Teeth around the rim, tipped outwards, cycling through `colors`."""
    for i in range(teeth):
        a = 2 * math.pi * i / teeth
        x, y = cx + rx * math.cos(a), ry * math.sin(a)
        if x > face_clear_x:
            continue
        hit = shell.top(x, y)
        if hit is None:
            continue
        p, n = hit
        out = np.array([math.cos(a), math.sin(a), 0.0])
        tooth_n = n + 0.6 * out
        d.add(colors[i % len(colors)], align(spine(radius, height, 0.0), tooth_n, p - 0.001 * n))


def wings(d: Decor, shell: Shell, feather_colors: Sequence[str], root_color: str, *,
          x: float = -0.058, y: float = 0.052, length: float = 0.078, taper: float = 0.006,
          width: float = 0.019, thick: float = 0.0045, fan_from_deg: float = -78,
          fan_step_deg: float = 15, tilt_deg: float = -38) -> None:
    """A fan of feathers each side at the back, from pointing back (first color)
    to pointing up (last), tipped outwards, with a painted dot at the root."""
    for side in (1, -1):
        hit = shell.top(x, y * side)
        if hit is None:
            continue
        p, _ = hit
        for k, color in enumerate(feather_colors):
            f = feather(length - taper * k, width, thick)
            f.apply_transform(trimesh.transformations.rotation_matrix(
                math.radians(fan_from_deg + fan_step_deg * k), [0, 1, 0]))
            f.apply_transform(trimesh.transformations.rotation_matrix(
                side * math.radians(tilt_deg), [1, 0, 0]))
            f.apply_translation(p + np.array([0.0, 0.0, -0.003]))
            d.add(color, f)
        d.add(root_color, align(disc(0.0075, 0.003), np.array([0, 0.35 * side, 1.0]), p))


def eyestalks(d: Decor, shell: Shell, stalk: str, eye: str, pupil: str, lid: str, *,
              x: float = 0.040, y: float = 0.030, stalk_height: float = 0.026,
              eye_radius: float = 0.0115, pupil_radius: float = 0.0058,
              pupil_forward: float = 0.0082, lid_radius: float = 0.0098) -> None:
    """Two crab eyestalks above the face: stalk, eye, pupil looking forward, lid."""
    for side in (1, -1):
        hit = shell.top(x, y * side)
        if hit is None:
            continue
        p, _ = hit
        s = trimesh.creation.cylinder(radius=0.0045, height=stalk_height, sections=24)
        s.apply_translation(p + [0.004, 0.0, stalk_height / 2 - 0.002])
        d.add(stalk, s)
        eye_c = p + np.array([0.006, 0.0, stalk_height + 0.006])
        e = trimesh.creation.icosphere(subdivisions=3, radius=eye_radius)
        e.apply_translation(eye_c)
        d.add(eye, e)
        pp = trimesh.creation.icosphere(subdivisions=3, radius=pupil_radius)
        pp.apply_translation(eye_c + np.array([pupil_forward, 0.0012 * side, 0.0012]))
        d.add(pupil, pp)
        lid_m = trimesh.creation.torus(major_radius=lid_radius, minor_radius=0.0026,
                                       major_sections=32, minor_sections=12)
        lid_m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(70), [0, 1, 0]))
        lid_m.apply_translation(eye_c + np.array([0.0005, 0.0, 0.0035]))
        d.add(lid, lid_m)
