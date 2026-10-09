"""Render the packaged skin itself (its scene.xml at the display pose) as the preview."""
import sys
from pathlib import Path
import mujoco, imageio.v3 as iio

sim = Path(sys.argv[1]); out = Path(sys.argv[2])
m = mujoco.MjModel.from_xml_path(str(sim / "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("display_home").id)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=600, width=800)
opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.0, 0, 0.12]; cam.distance = 0.78; cam.azimuth = 150; cam.elevation = -22
r.update_scene(d, camera=cam, scene_option=opt)
iio.imwrite(out, r.render()); print("wrote", out)
