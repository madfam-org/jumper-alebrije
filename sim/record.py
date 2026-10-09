"""Simulate a choreography once and save the robot pose at video rate.

    python sim/record.py choreography.json out.npz [--stock]

The choreography is a list of key events at times in seconds:
    {"seconds": 46, "fps": 30, "events": [[0.6, "down", "key_1"], ...]}

Saved per video frame: the full qpos, the controller's mode, the base
position and tilt, so a renderer can replay it with any camera.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

A = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(A / "sim"), str(A / "design"), str(A / "scene")]
from jumper_sim import App, attach_robot, base_state, robot_spec  # noqa: E402


def plain_world() -> mujoco.MjSpec:
    w = mujoco.MjSpec()
    w.option.timestep = 0.001
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.option.ccd_iterations = 50
    w.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1],
                         contype=1, conaffinity=1, condim=3, friction=[1, 0.005, 0.0001])
    return w


def simulate(choreo: dict, alebrije: bool = True) -> dict:
    w = plain_world()
    r = robot_spec()
    if alebrije:
        from alebrije import apply_to_spec
        apply_to_spec(r)
    attach_robot(w, r)
    m = w.compile()
    d = mujoco.MjData(m)
    app = App(m)
    app.reset(d)
    hz = app.control_hz
    sub = round(1.0 / (hz * m.opt.timestep))
    fps = choreo.get("fps", 30)
    events: dict[int, tuple[list, list]] = {}
    for t, kind, key in choreo["events"]:
        dn, up = events.setdefault(round(t * hz), ([], []))
        (dn if kind == "down" else up).append(key)
    n_ticks = int(choreo["seconds"] * hz)
    every = hz / fps
    qpos, modes, tilts = [], [], []
    next_frame = 0.0
    for k in range(n_ticks):
        dn, up = events.get(k, ([], []))
        mode = app.tick(d, int(round(k * 1e6 / hz)), dn, up)
        for _ in range(sub):
            app.apply_torque(d)
            mujoco.mj_step(m, d)
        if k + 1 >= next_frame:
            qpos.append(d.qpos.copy())
            modes.append(mode)
            tilts.append(base_state(app, d)[1])
            next_frame += every
    return {"qpos": np.array(qpos), "modes": np.array(modes), "tilt": np.array(tilts),
            "fps": fps, "log": [tuple(map(str, e)) for e in app.log]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("choreography", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--stock", action="store_true", help="the stock look, for comparison")
    a = p.parse_args()
    t0 = time.time()
    res = simulate(json.loads(a.choreography.read_text()), alebrije=not a.stock)
    np.savez_compressed(a.out, qpos=res["qpos"], modes=res["modes"], tilt=res["tilt"],
                        fps=res["fps"], log=np.array(res["log"]))
    print(f"{a.out.name}: {len(res['qpos'])} frames, worst tilt {res['tilt'].max():.1f} deg, "
          f"{time.time() - t0:.0f}s wall")
    for e in res["log"]:
        print("  fsm:", e)
