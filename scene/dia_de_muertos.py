"""A Día de Muertos plaza for Jumper: marigold path, ofrenda, papel picado.

Built as a MuJoCo world (MjSpec) at robot scale: Jumper stands about 0.2 m
tall, so the ofrenda is roughly twice that. Only the floor and the altar tiers
collide; petals, flowers and decorations are visual (contype=conaffinity=0).

Coordinates: Z up, the robot spawns at the origin facing +X; the ofrenda
stands behind it at -X, the camera mostly sits on the +X side.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
TEX = HERE / "textures"

GEOM = mujoco.mjtGeom


def _hex(h: str, a: float = 1.0) -> list[float]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] + [a]


C = {
    "cempasuchil": "#FF8A00", "cempasuchil2": "#FFB000", "cempasuchil3": "#F26A0A",
    "terciopelo": "#B0125B", "rosa": "#E4007C", "morado": "#6A2FB8",
    "turquesa": "#00B3A6", "amarillo": "#FFC20E", "naranja": "#FF6A13",
    "lima": "#7DC242", "cobalto": "#1F4FD8", "blanco": "#FFF6E0",
    "cera": "#FFF1C9", "llama": "#FFB347", "pan": "#C8843E", "pan2": "#A9652B",
    "barro": "#B5562F", "mantel": "#F8F3EA", "verde": "#2E7D32", "verde2": "#43A047",
    "adobe": "#2A5DB0", "negro": "#141414",
}


# ── textures ─────────────────────────────────────────────────────────────────

def make_textures() -> dict[str, Path]:
    TEX.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    out = {}

    # Terracotta floor tiles with lighter grout and per-tile variation.
    n, px = 4, 512
    img = Image.new("RGB", (px, px), (222, 196, 160))
    d = ImageDraw.Draw(img)
    cell = px // n
    for i in range(n):
        for j in range(n):
            base = np.array([181, 86, 47]) + rng.integers(-14, 15, 3)
            d.rectangle([i * cell + 5, j * cell + 5, (i + 1) * cell - 5, (j + 1) * cell - 5],
                        fill=tuple(int(v) for v in np.clip(base, 0, 255)))
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-8, 9, arr.shape[:2])[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    out["floor"] = TEX / "barro_tiles.png"
    img.save(out["floor"])

    # Papel picado: one sheet per color, the cut pattern as transparent holes.
    for name in ("rosa", "morado", "turquesa", "amarillo", "naranja", "lima", "cobalto"):
        w, h = 240, 320
        sheet = Image.new("RGBA", (w, h), tuple(int(c * 255) for c in _hex(C[name])[:3]) + (255,))
        d = ImageDraw.Draw(sheet)
        hole = (255, 236, 200, 0)
        # Border of small diamonds.
        for x in range(14, w - 10, 22):
            for y in (14, h - 18):
                d.polygon([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)], fill=hole)
        for y in range(36, h - 30, 22):
            for x in (14, w - 14):
                d.polygon([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)], fill=hole)
        # Central calavera: skull outline with eye holes, carved as negative space.
        cx, cy = w // 2, h // 2 - 10
        d.ellipse([cx - 62, cy - 70, cx + 62, cy + 48], fill=hole)
        d.rounded_rectangle([cx - 36, cy + 30, cx + 36, cy + 78], radius=12, fill=hole)
        col = tuple(int(c * 255) for c in _hex(C[name])[:3]) + (255,)
        d.ellipse([cx - 44, cy - 24, cx - 8, cy + 14], fill=col)
        d.ellipse([cx + 8, cy - 24, cx + 44, cy + 14], fill=col)
        d.polygon([(cx, cy + 18), (cx - 9, cy + 34), (cx + 9, cy + 34)], fill=col)
        for k in range(-2, 3):
            d.rectangle([cx + k * 13 - 4, cy + 44, cx + k * 13 + 4, cy + 66], fill=col)
        # Flowers on the forehead.
        for fx in (cx - 22, cx, cx + 22):
            d.ellipse([fx - 7, cy - 58, fx + 7, cy - 44], fill=col)
        # Scalloped bottom edge.
        for x in range(0, w, 30):
            d.pieslice([x, h - 22, x + 30, h + 8], 180, 360, fill=hole)
        out[f"papel_{name}"] = TEX / f"papel_{name}.png"
        sheet.save(out[f"papel_{name}"])

    # Dusk skybox as a 4x3 cube cross: purple overhead, a pink-orange glow at
    # the horizon (the middle row of each side face), warm dark below, stars.
    f = 256
    sky = np.zeros((3 * f, 4 * f, 3), np.float32)
    top, mid, glow, below = (np.array(c, np.float32) for c in
                             ([40, 22, 92], [196, 72, 128], [255, 168, 92], [70, 36, 40]))
    rows = np.linspace(0, 1, f)[:, None]  # 0 at the top edge of a side face
    def side(v):
        up = np.clip(v / 0.5, 0, 1)
        c = np.where(v < 0.5, top + (mid - top) * up ** 1.6,
                     np.where(v < 0.56, mid + (glow - mid) * ((v - 0.5) / 0.06), below))
        return c
    band = np.repeat(side(rows)[:, None, :], f, axis=1).reshape(f, f, 3)
    for col in range(4):
        sky[f:2 * f, col * f:(col + 1) * f] = band
    sky[0:f, f:2 * f] = top          # up
    sky[2 * f:3 * f, f:2 * f] = below  # down
    stars = rng.random((3 * f, 4 * f)) > 0.9985
    mask = np.zeros_like(stars); mask[0:f, f:2 * f] = True; mask[f:f + f // 3, :] = True
    sky[stars & mask] = [255, 244, 220]
    out["sky"] = TEX / "dusk_sky.png"
    Image.fromarray(sky.astype(np.uint8)).save(out["sky"])
    return out


# ── meshes built from many small pieces ──────────────────────────────────────

def petal_carpet(rng: np.random.Generator) -> dict[str, trimesh.Trimesh]:
    """A marigold path: from the camera side to the ofrenda, widening into a
    cross where the robot dances. One mesh per petal color."""
    groups: dict[str, list] = {}
    base = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

    def petal(x, y, color, r):
        p = base.copy()
        p.apply_scale([r, r * rng.uniform(0.6, 1.0), r * 0.28])
        p.apply_transform(trimesh.transformations.rotation_matrix(rng.uniform(0, math.pi), [0, 0, 1]))
        p.apply_translation([x, y, r * 0.18])
        groups.setdefault(color, []).append(p)

    def inside(x, y):
        path = abs(y) < 0.16 + 0.03 * math.sin(6 * x) and -0.75 < x < 1.6
        cross = abs(x) < 0.16 and abs(y) < 0.55
        circle = x * x + y * y < 0.36 ** 2
        return path or cross or circle

    count = 0
    while count < 5200:
        x, y = rng.uniform(-0.8, 1.6), rng.uniform(-0.6, 0.6)
        if not inside(x, y):
            continue
        ring = x * x + y * y
        if 0.30 ** 2 < ring < 0.36 ** 2 and rng.random() < 0.7:
            color = "terciopelo"
        else:
            color = rng.choice(["cempasuchil", "cempasuchil2", "cempasuchil3"], p=[0.5, 0.3, 0.2])
        petal(x, y, color, rng.uniform(0.006, 0.011))
        count += 1
    return {c: trimesh.util.concatenate(v) for c, v in groups.items()}


def marigold_bunch(center, radius, rng, count=40) -> dict[str, trimesh.Trimesh]:
    groups: dict[str, list] = {}
    for _ in range(count):
        v = rng.normal(size=3)
        v[2] = abs(v[2]) * 0.8
        v /= np.linalg.norm(v)
        r = rng.uniform(0.010, 0.016)
        s = trimesh.creation.icosphere(subdivisions=2, radius=r)
        s.apply_translation(np.asarray(center) + v * radius)
        groups.setdefault(rng.choice(["cempasuchil", "cempasuchil2", "cempasuchil3"]), []).append(s)
    return {c: trimesh.util.concatenate(v) for c, v in groups.items()}


def bougainvillea(x0, y0, z0, width, height, rng) -> dict[str, trimesh.Trimesh]:
    groups: dict[str, list] = {}
    for _ in range(900):
        x = x0 + rng.normal(0, 0.03)
        y = y0 + rng.uniform(-width / 2, width / 2)
        z = z0 + height * (1 - rng.power(2.2))
        if rng.random() < 0.7:
            color, r = rng.choice(["rosa", "terciopelo"]), rng.uniform(0.008, 0.014)
        else:
            color, r = rng.choice(["verde", "verde2"]), rng.uniform(0.009, 0.016)
        s = trimesh.creation.icosphere(subdivisions=1, radius=r)
        s.apply_translation([x, y, z])
        groups.setdefault(color, []).append(s)
    return {c: trimesh.util.concatenate(v) for c, v in groups.items()}


# ── the world ────────────────────────────────────────────────────────────────

class Builder:
    def __init__(self, spec: mujoco.MjSpec) -> None:
        self.s = spec
        self.n = 0
        self.mats: set[str] = set()

    def mat(self, color: str, emission: float = 0.0, specular: float = 0.3,
            shininess: float = 0.4) -> str:
        name = f"m_{color}_{emission:g}_{specular:g}"
        if name not in self.mats:
            self.s.add_material(name=name, rgba=_hex(C[color]), emission=emission,
                                specular=specular, shininess=shininess)
            self.mats.add(name)
        return name

    def geom(self, type_, size, pos, color, quat=(1, 0, 0, 0), collide=False, **mat):
        self.n += 1
        return self.s.worldbody.add_geom(
            name=f"deco_{self.n}", type=type_, size=list(size) + [0] * (3 - len(size)),
            pos=list(pos), quat=list(quat), material=self.mat(color, **mat),
            contype=1 if collide else 0, conaffinity=1 if collide else 0, group=0)

    def meshes(self, groups: dict[str, trimesh.Trimesh], tag: str, **mat) -> None:
        for color, m in groups.items():
            name = f"{tag}_{color}"
            self.s.add_mesh(name=name,
                            uservert=np.asarray(m.vertices, float).ravel().tolist(),
                            userface=np.asarray(m.faces, int).ravel().tolist())
            self.n += 1
            self.s.worldbody.add_geom(name=f"deco_{self.n}", type=GEOM.mjGEOM_MESH, meshname=name,
                                      material=self.mat(color, **mat), contype=0, conaffinity=0)


def _quat_z(deg: float):
    h = math.radians(deg) / 2
    return (math.cos(h), 0, 0, math.sin(h))


def world(width: int = 1080, height: int = 1920) -> mujoco.MjSpec:
    rng = np.random.default_rng(11)
    tex = make_textures()
    s = mujoco.MjSpec()
    s.option.timestep = 0.001
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.ccd_iterations = 50
    s.visual.global_.offwidth = width
    s.visual.global_.offheight = height
    s.visual.headlight.ambient = [0.30, 0.24, 0.26]
    s.visual.headlight.diffuse = [0.30, 0.27, 0.25]
    s.visual.headlight.specular = [0.05, 0.05, 0.05]
    s.visual.quality.shadowsize = 4096
    s.visual.map.znear = 0.005
    s.visual.rgba.haze = [0.98, 0.62, 0.42, 1]

    b = Builder(s)

    # Dusk sky.
    s.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, file=str(tex["sky"]),
                  gridsize=[3, 4], gridlayout=".U..LFRB.D..")

    # Floor: terracotta tiles. Contact like the training ground plane.
    s.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tex["floor"]))
    s.add_material(name="floor", textures=["", "floor"], texrepeat=[3, 3], texuniform=True,
                   specular=0.15, shininess=0.2)
    s.worldbody.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=[4, 4, 0.1], material="floor",
                         contype=1, conaffinity=1, condim=3, friction=[1, 0.005, 0.0001])

    # Marigold petal path, cross and ring.
    b.meshes(petal_carpet(rng), "petals", specular=0.1)

    # Ofrenda: three tiers with a white cloth and magenta skirts, behind the robot.
    ox = -0.78
    tiers = [(0.30, 0.10, 0.60), (0.22, 0.20, 0.48), (0.14, 0.30, 0.36)]  # (depth/2, top z, width/2)
    for i, (dx, top, wy) in enumerate(tiers):
        x = ox - 0.10 * i
        b.geom(GEOM.mjGEOM_BOX, (dx * 0.5, wy, top / 2), (x, 0, top / 2), "mantel", collide=True)
        b.geom(GEOM.mjGEOM_BOX, (0.004, wy + 0.004, 0.025), (x + dx * 0.5 + 0.002, 0, top - 0.025),
               ["rosa", "morado", "naranja"][i])
        # Scalloped trim: little half-discs along the skirt edge.
        for k in range(int(wy * 2 / 0.05)):
            y = -wy + 0.025 + k * 0.05
            b.geom(GEOM.mjGEOM_CYLINDER, (0.016, 0.003), (x + dx * 0.5 + 0.004, y, top - 0.05),
                   ["amarillo", "turquesa", "lima"][(i + k) % 3], quat=(0.7071, 0, 0.7071, 0))
    # Candles on every tier, with emissive flames and a few warm point lights.
    candle_spots = []
    for i, (dx, top, wy) in enumerate(tiers):
        x = ox - 0.10 * i + 0.02
        for y in np.linspace(-wy + 0.06, wy - 0.06, 5 - i):
            h = rng.uniform(0.05, 0.10)
            b.geom(GEOM.mjGEOM_CYLINDER, (0.011, h / 2), (x, y, top + h / 2), "cera", specular=0.2)
            b.geom(GEOM.mjGEOM_ELLIPSOID, (0.006, 0.006, 0.013), (x, y, top + h + 0.012), "llama",
                   emission=1.0)
            candle_spots.append((x, y, top + h + 0.03))
    for k, (x, y, z) in enumerate(candle_spots[::3][:5]):
        s.worldbody.add_light(name=f"candle_{k}", pos=[x + 0.05, y, z + 0.04], dir=[1, 0, -0.4],
                              diffuse=[0.55, 0.32, 0.12], specular=[0.1, 0.06, 0.02],
                              attenuation=[0.3, 1.2, 4.0], castshadow=False, cutoff=80)

    # Sugar skulls on the first tier.
    for y in (-0.36, -0.12, 0.12, 0.36):
        x, z = ox + 0.07, tiers[0][1] + 0.030
        b.geom(GEOM.mjGEOM_ELLIPSOID, (0.030, 0.034, 0.030), (x, y, z), "blanco", specular=0.5)
        b.geom(GEOM.mjGEOM_BOX, (0.018, 0.022, 0.012), (x + 0.012, y, z - 0.026), "blanco", specular=0.5)
        for side in (-1, 1):
            b.geom(GEOM.mjGEOM_SPHERE, (0.0095,), (x + 0.026, y + side * 0.0125, z + 0.002),
                   rng.choice(["turquesa", "rosa", "amarillo", "lima"]))
        b.geom(GEOM.mjGEOM_SPHERE, (0.006,), (x + 0.029, y, z - 0.010), "negro")
        for k in range(3):
            b.geom(GEOM.mjGEOM_SPHERE, (0.005,), (x + 0.020, y + (k - 1) * 0.011, z + 0.024),
                   ["rosa", "amarillo", "turquesa"][k])
    # Pan de muerto on the second tier.
    for y in (-0.26, 0.26):
        x, z = ox - 0.10 + 0.02, tiers[1][1]
        b.geom(GEOM.mjGEOM_ELLIPSOID, (0.045, 0.045, 0.026), (x, y, z + 0.01), "pan", specular=0.4)
        for q in ((0.7071, 0, 0.7071, 0), (0.7071, -0.7071, 0, 0)):
            b.geom(GEOM.mjGEOM_CAPSULE, (0.006, 0.035), (x, y, z + 0.034), "pan2", quat=q)
        b.geom(GEOM.mjGEOM_SPHERE, (0.011,), (x, y, z + 0.045), "pan2")
    # Clay pots with marigold bunches on the top tier and on the floor.
    pots = [(ox - 0.20 + 0.01, -0.22, tiers[2][1]), (ox - 0.20 + 0.01, 0.22, tiers[2][1]),
            (ox + 0.24, -0.62, 0.0), (ox + 0.24, 0.62, 0.0), (0.55, -0.52, 0.0), (0.55, 0.52, 0.0)]
    for x, y, z in pots:
        b.geom(GEOM.mjGEOM_CYLINDER, (0.04, 0.045), (x, y, z + 0.045), "barro", specular=0.2)
        b.geom(GEOM.mjGEOM_CYLINDER, (0.045, 0.006), (x, y, z + 0.09), "barro", specular=0.2)
        b.meshes(marigold_bunch((x, y, z + 0.12), 0.05, rng), f"bunch_{b.n}", specular=0.1)

    # Back wall in cobalt adobe with bougainvillea spilling over it.
    b.geom(GEOM.mjGEOM_BOX, (0.05, 1.6, 0.31), (-1.25, 0, 0.31), "adobe", collide=False, specular=0.05)
    b.geom(GEOM.mjGEOM_BOX, (0.06, 1.6, 0.02), (-1.24, 0, 0.63), "amarillo", specular=0.05)
    for y0 in (-0.9, 0.0, 0.9):
        b.meshes(bougainvillea(-1.18, y0, 0.30, 0.7, 0.45, rng), f"bugambilia_{b.n}", specular=0.1)

    # Papel picado: strings of flags over the scene.
    colors = ["rosa", "morado", "turquesa", "amarillo", "naranja", "lima", "cobalto"]
    for name in colors:
        s.add_texture(name=f"papel_{name}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      file=str(tex[f"papel_{name}"]))
        s.add_material(name=f"papel_{name}", textures=["", f"papel_{name}"],
                       rgba=[1, 1, 1, 0.999], specular=0.0, emission=0.15)
    s.add_mesh(name="papel_quad",
               uservert=[0, -0.055, -0.072, 0, 0.055, -0.072, 0, 0.055, 0.072, 0, -0.055, 0.072],
               userface=[0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2],
               usertexcoord=[0, 1, 1, 1, 1, 0, 0, 0],
               inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    for row, (x, z, sag) in enumerate([(-0.55, 0.80, 0.05), (-0.12, 0.88, 0.07)]):
        n = 11
        for k in range(n):
            y = -0.85 + 1.7 * k / (n - 1)
            zz = z - sag * (1 - (y / 0.85) ** 2)
            b.n += 1
            s.worldbody.add_geom(name=f"papel_{b.n}", type=GEOM.mjGEOM_MESH, meshname="papel_quad",
                                 pos=[x, y, zz - 0.072],
                                 material=f"papel_{colors[(k + 2 * row) % len(colors)]}",
                                 contype=0, conaffinity=0)
        # The string.
        for k in range(n - 1):
            y0 = -0.85 + 1.7 * k / (n - 1)
            y1 = -0.85 + 1.7 * (k + 1) / (n - 1)
            z0 = z - sag * (1 - (y0 / 0.85) ** 2)
            z1 = z - sag * (1 - (y1 / 0.85) ** 2)
            b.n += 1
            s.worldbody.add_geom(name=f"cuerda_{b.n}", type=GEOM.mjGEOM_CAPSULE,
                                 fromto=[x, y0, z0, x, y1, z1], size=[0.0012, 0, 0],
                                 material=b.mat("blanco"), contype=0, conaffinity=0)

    # Warm low key light, like the last sun of the day, plus a soft fill.
    s.worldbody.add_light(name="sun", pos=[1.6, 1.2, 1.4], dir=[-0.75, -0.55, -0.6],
                          type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True,
                          diffuse=[0.85, 0.62, 0.45], specular=[0.3, 0.25, 0.2])
    s.worldbody.add_light(name="fill", pos=[1.0, -1.5, 1.2], dir=[-0.4, 0.8, -0.6],
                          type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False,
                          diffuse=[0.22, 0.20, 0.32], specular=[0, 0, 0])
    return s
