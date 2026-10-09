"""Still images: a character's look, the packaged skin and map, and lineups.

- `look_sheet`: the look on a studio floor from four angles (design review)
- `plaza_stills`: the look in the Día de Muertos plaza from three angles
- `skin_preview` / `map_preview`: render the *packaged* skin or map itself, as
  shellflow composes it; these become the thumbnails embedded in the packages
- `lineup`: every character at the same frame of a recorded dance, same camera,
  tiled with titles, for picking between them
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .character import Character, apply_to_spec
from .paths import PROFILE, use_shellflow
from .sim import PREFIX, App, attach_robot, robot_spec

FONT = "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"


def _hide_collision() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0  # collision meshes overlap the visuals
    return opt


def _camera(lookat, azimuth: float, elevation: float, distance: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    return cam


def studio(size: int = 1600) -> mujoco.MjSpec:
    """A plain warm-checker floor and one shadowing light, for design review."""
    w = mujoco.MjSpec()
    w.option.timestep = 0.001
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.visual.global_.offwidth = size
    w.visual.global_.offheight = size
    w.visual.headlight.ambient = [0.35, 0.35, 0.38]
    w.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    w.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                  rgb1=[0.93, 0.89, 0.82], rgb2=[0.88, 0.83, 0.75], width=512, height=512)
    w.add_material(name="floor", textures=["", "grid"], texrepeat=[6, 6], texuniform=True)
    w.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                         material="floor", contype=1, conaffinity=1)
    w.worldbody.add_light(pos=[0.6, -0.8, 1.6], dir=[-0.35, 0.45, -1], castshadow=True,
                          diffuse=[0.7, 0.7, 0.68])
    return w


def _model_with(world: mujoco.MjSpec, character: Character | None):
    r = robot_spec()
    if character is not None:
        apply_to_spec(r, character)
    attach_robot(world, r)
    m = world.compile()
    d = mujoco.MjData(m)
    App(m).reset(d)
    return m, d


def look_sheet(character: Character | None, out: Path, tile: int = 800) -> Path:
    """Four views on the studio floor: front 3/4 both sides, top, rear."""
    m, d = _model_with(studio(2 * tile), character)
    ren = mujoco.Renderer(m, height=tile, width=tile)
    frames = []
    for az, el, dist in [(150, -25, 0.75), (210, -20, 0.75), (90, -60, 0.8), (0, -12, 0.7)]:
        ren.update_scene(d, camera=_camera([0.0, 0, 0.11], az, el, dist),
                         scene_option=_hide_collision())
        frames.append(ren.render())
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])]))
    return out


def plaza_stills(character: Character | None, out: Path) -> Path:
    """Three views in the Día de Muertos plaza, side by side."""
    from .scene import world
    m, d = _model_with(world(), character)
    ren = mujoco.Renderer(m, height=960, width=540)
    frames = []
    for az, el, dist, look in [(180, -8, 1.05, (0, 0, 0.22)), (150, -14, 0.75, (0, 0, 0.12)),
                               (205, -30, 1.6, (-0.3, 0, 0.25))]:
        ren.update_scene(d, camera=_camera(look, az, el, dist), scene_option=_hide_collision())
        frames.append(ren.render())
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, np.hstack(frames))
    return out


def skin_preview(simulation_dir: Path, out: Path) -> Path:
    """The packaged robot from its own scene.xml at its display pose, 800x600."""
    m = mujoco.MjModel.from_xml_path(str(simulation_dir / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("display_home").id)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=600, width=800)
    r.update_scene(d, camera=_camera([0.0, 0, 0.12], 150, -22, 0.78), scene_option=_hide_collision())
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, r.render())
    return out


def map_preview(map_file: Path, out: Path) -> Path:
    """The packaged map composed with its bundled robot, as shellflow composes it."""
    use_shellflow()
    from shellflow.map_package import compose_default
    from shellflow.package_io import read_archive

    m, d, _ = compose_default(read_archive(map_file), profile=PROFILE)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1000, 750
    m.vis.quality.shadowsize = 4096
    m.vis.headlight.ambient[:] = [0.30, 0.24, 0.26]
    m.vis.headlight.diffuse[:] = [0.30, 0.27, 0.25]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=750, width=1000)
    r.update_scene(d, camera=_camera([-0.25, 0, 0.18], 200, -24, 1.9), scene_option=_hide_collision())
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, r.render())
    return out


#: Standing after the bow in take1: claws down, face, eyes and shell in view.
LINEUP_AT = 41.5
LINEUP_CAMERA = (160, -26, 0.8, 0.10)  # azimuth, elevation, distance, look-at height


def lineup(characters: list[Character], take: Path, out: Path, *, at: float = LINEUP_AT,
           tile: tuple[int, int] = (480, 640), columns: int = 4,
           camera: tuple[float, float, float, float] = LINEUP_CAMERA) -> Path:
    """Each character at the same moment of a recorded take, same camera, titled.

    Physics never depends on the look, so one recorded take poses everyone.
    """
    from .scene import world
    z = np.load(take)
    qpos, fps = z["qpos"], float(z["fps"])
    i = min(len(qpos) - 1, round(at * fps))
    w_px, h_px = tile
    font = ImageFont.truetype(FONT, max(14, h_px // 22))
    tiles = []
    for character in characters:
        m, d = _model_with(world(w_px, h_px), character)
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        root = m.jnt_qposadr[m.joint(PREFIX + "floating_base").id]
        base = qpos[i][root:root + 3]
        az, el, dist, lz = camera
        ren = mujoco.Renderer(m, height=h_px, width=w_px)
        ren.update_scene(d, camera=_camera([base[0], base[1], lz], az, el, dist),
                         scene_option=_hide_collision())
        img = Image.fromarray(ren.render())
        ren.close()
        draw = ImageDraw.Draw(img)
        label = f"{character.title}  ({character.id})"
        draw.rectangle([0, h_px - h_px // 11, w_px, h_px], fill=(20, 10, 30))
        draw.text((12, h_px - h_px // 11 + 6), label, font=font, fill=(255, 245, 225))
        tiles.append(img)
    rows = math.ceil(len(tiles) / columns)
    cols = min(columns, len(tiles))
    sheet = Image.new("RGB", (cols * w_px, rows * h_px), (20, 10, 30))
    for k, img in enumerate(tiles):
        sheet.paste(img, ((k % cols) * w_px, (k // cols) * h_px))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
