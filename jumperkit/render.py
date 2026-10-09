"""Render a recorded take in the Día de Muertos plaza, as a vertical video.

`shots` (a JSON file) holds camera keyframes and captions on the video timeline:
    {"camera": [[t, azimuth, elevation, distance, lookat_z], ...],
     "captions": [[t0, t1, "text", "top"|"middle"|"bottom"], ...],
     "badge": "SIMULACIÓN | MuJoCo"}

Camera keyframes are interpolated with a smoothstep; the look-at point follows
the robot's base, low-pass filtered so the camera does not jitter with steps.
Arial Rounded Bold has no middle dot (·); use "|" in captions and badges.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .character import Character, apply_to_spec
from .scene import world
from .sim import PREFIX, attach_robot, robot_spec

FONT = "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"


def smooth(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3 - 2 * a)


def camera_at(keys: list[list[float]], t: float) -> list[float]:
    if t <= keys[0][0]:
        return keys[0][1:]
    for k0, k1 in zip(keys, keys[1:]):
        if k0[0] <= t <= k1[0]:
            s = smooth((t - k0[0]) / max(1e-6, k1[0] - k0[0]))
            return [v0 + (v1 - v0) * s for v0, v1 in zip(k0[1:], k1[1:])]
    return keys[-1][1:]


def caption(img: Image.Image, text: str, where: str, alpha: float, size: int) -> None:
    if alpha <= 0:
        return
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    lines = text.split("\n")
    # Shrink to fit 86% of the width, the safe area of a vertical player.
    while size > 12:
        font = ImageFont.truetype(FONT, size)
        if max(d.textbbox((0, 0), ln, font=font)[2] for ln in lines) <= W * 0.86:
            break
        size -= 2
    heights = [d.textbbox((0, 0), ln, font=font)[3] for ln in lines]
    total = sum(heights) + (len(lines) - 1) * size * 0.25
    y = {"top": H * 0.10, "middle": H * 0.42 - total / 2, "bottom": H * 0.66 - total}[where]
    for ln, h in zip(lines, heights):
        w = d.textbbox((0, 0), ln, font=font)[2]
        d.text(((W - w) / 2, y), ln, font=font, fill=(255, 250, 235, int(255 * alpha)),
               stroke_width=max(2, size // 12), stroke_fill=(40, 10, 50, int(230 * alpha)))
        y += h + size * 0.25
    img.alpha_composite(layer)


def badge(img: Image.Image, text: str, size: int) -> None:
    font = ImageFont.truetype(FONT, size)
    d = ImageDraw.Draw(img)
    w = d.textbbox((0, 0), text, font=font)[2]
    pad = size * 0.5
    x, y = img.size[0] - w - 2.4 * pad, img.size[1] * 0.035
    d.rounded_rectangle([x, y, x + w + 2 * pad, y + size + 1.4 * pad], radius=size * 0.6,
                        fill=(20, 10, 30, 150))
    d.text((x + pad, y + 0.6 * pad), text, font=font, fill=(255, 245, 225, 235))


def render_take(take: Path, shots: Path, out: Path, character: Character | None, *,
                scale: float = 1.0, step: int = 1, stills: list[float] | None = None,
                log=print) -> Path:
    """Render `take` (from `sim.simulate`) to an MP4, or to PNG stills at `stills` seconds.

    `scale` 1.0 renders 1080x1920; `step` renders every Nth frame (fps / step).
    Stills are written beside `out` as `<stem>_t<seconds>.png`.
    """
    z = np.load(take)
    qpos, fps = z["qpos"], float(z["fps"])
    shot = json.loads(Path(shots).read_text())
    W, H = int(1080 * scale), int(1920 * scale)

    w = world(1080, 1920)
    r = robot_spec()
    if character is not None:
        apply_to_spec(r, character)
    attach_robot(w, r)
    m = w.compile()
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0  # collision meshes stay hidden
    root = m.jnt_qposadr[m.joint(PREFIX + "floating_base").id]

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None if stills else imageio.get_writer(
        out, fps=fps / step, codec="libx264", quality=None, macro_block_size=8,
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "medium"])
    look = qpos[0][root:root + 3].copy()
    t0 = time.time()
    indices = [min(len(qpos) - 1, round(t * fps)) for t in stills] if stills \
        else range(0, len(qpos), step)
    for n, i in enumerate(indices):
        t = i / fps
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        base = qpos[i][root:root + 3]
        look += (base - look) * (1.0 if stills else min(1.0, 2.5 * step / fps))
        az, el, dist, lz = camera_at(shot["camera"], t)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [look[0], look[1], lz]
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        ren.update_scene(d, camera=cam, scene_option=opt)
        img = Image.fromarray(ren.render()).convert("RGBA")
        for c0, c1, text, where in shot.get("captions", []):
            fade = min(1.0, (t - c0) / 0.25, (c1 - t) / 0.25)
            caption(img, text, where, fade, int(H * 0.052))
        if shot.get("badge"):
            badge(img, shot["badge"], int(H * 0.016))
        frame = np.asarray(img.convert("RGB"))
        if stills:
            Image.fromarray(frame).save(out.with_name(f"{out.stem}_t{t:05.1f}.png"))
        else:
            writer.append_data(frame)
        if n % 60 == 0:
            log(f"  frame {n}/{len(indices)}  t={t:5.1f}s  {time.time() - t0:5.0f}s")
    if writer:
        writer.close()
    log(f"wrote {out} ({len(indices)} frames, {time.time() - t0:.0f}s)")
    return out
