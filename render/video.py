"""Render a recorded choreography in the Día de Muertos plaza, as a vertical video.

    python render/video.py out/take.npz shots.json out/take.mp4 [--scale 0.5] [--step 1]

`shots.json` holds camera keyframes and captions on the video timeline:
    {"camera": [[t, azimuth, elevation, distance, lookat_z], ...],
     "captions": [[t0, t1, "text", "top"|"middle"|"bottom"], ...],
     "badge": "SIMULATION · MuJoCo"}

Camera keyframes are interpolated with a smoothstep; the look-at point follows
the robot's base, low-pass filtered so the camera does not jitter with steps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

A = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(A / "sim"), str(A / "design"), str(A / "scene")]
from alebrije import apply_to_spec  # noqa: E402
from dia_de_muertos import world  # noqa: E402
from jumper_sim import PREFIX, attach_robot, robot_spec  # noqa: E402

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("take", type=Path)
    p.add_argument("shots", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--scale", type=float, default=1.0, help="0.5 renders 540x960")
    p.add_argument("--step", type=int, default=1, help="render every Nth frame")
    p.add_argument("--frames", type=str, default="", help="comma list of times to save as PNG")
    a = p.parse_args()

    take = np.load(a.take)
    qpos, fps = take["qpos"], float(take["fps"])
    shots = json.loads(a.shots.read_text())
    W, H = int(1080 * a.scale), int(1920 * a.scale)

    w = world(1080, 1920)
    r = robot_spec()
    apply_to_spec(r)
    attach_robot(w, r)
    m = w.compile()
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0  # collision meshes stay hidden
    root = m.jnt_qposadr[m.joint(PREFIX + "floating_base").id]

    stills = [float(x) for x in a.frames.split(",") if x]
    writer = None if stills else imageio.get_writer(
        a.out, fps=fps / a.step, codec="libx264", quality=None, macro_block_size=8,
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "medium"])
    look = qpos[0][root:root + 3].copy()
    t0 = time.time()
    indices = [min(len(qpos) - 1, round(t * fps)) for t in stills] or range(0, len(qpos), a.step)
    for n, i in enumerate(indices):
        t = i / fps
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        base = qpos[i][root:root + 3]
        look += (base - look) * (1.0 if stills else min(1.0, 2.5 * a.step / fps))
        az, el, dist, lz = camera_at(shots["camera"], t)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [look[0], look[1], lz]
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        ren.update_scene(d, camera=cam, scene_option=opt)
        img = Image.fromarray(ren.render()).convert("RGBA")
        for c0, c1, text, where in shots.get("captions", []):
            fade = min(1.0, (t - c0) / 0.25, (c1 - t) / 0.25)
            caption(img, text, where, fade, int(H * 0.052))
        if shots.get("badge"):
            badge(img, shots["badge"], int(H * 0.016))
        frame = np.asarray(img.convert("RGB"))
        if stills:
            Image.fromarray(frame).save(a.out.with_name(f"{a.out.stem}_t{t:05.1f}.png"))
        else:
            writer.append_data(frame)
        if n % 60 == 0:
            print(f"  frame {n}/{len(indices)}  t={t:5.1f}s  {time.time() - t0:5.0f}s", flush=True)
    if writer:
        writer.close()
    print(f"wrote {a.out} ({len(indices)} frames, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
