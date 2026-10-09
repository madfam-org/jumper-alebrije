"""Still frames of the alebrije Jumper in the Día de Muertos plaza."""
import sys
from pathlib import Path
import mujoco, numpy as np, imageio.v3 as iio
A = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(A / "sim"), str(A / "design"), str(A / "scene")]
from jumper_sim import App, attach_robot, robot_spec
from alebrije import apply_to_spec
from dia_de_muertos import world

if __name__ == "__main__":
    w = world(); r = robot_spec(); apply_to_spec(r); attach_robot(w, r)
    m = w.compile(); d = mujoco.MjData(m); App(m).reset(d)
    print("ngeom", m.ngeom, "nlight", m.nlight)
    ren = mujoco.Renderer(m, height=960, width=540)
    opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
    frames = []
    for az, el, dist, look in [(180, -8, 1.05, (0, 0, 0.22)), (150, -14, 0.75, (0, 0, 0.12)), (205, -30, 1.6, (-0.3, 0, 0.25))]:
        cam = mujoco.MjvCamera(); cam.lookat[:] = look; cam.distance = dist; cam.azimuth = az; cam.elevation = el
        ren.update_scene(d, camera=cam, scene_option=opt); frames.append(ren.render())
    iio.imwrite(sys.argv[1], np.hstack(frames)); print("wrote", sys.argv[1])
