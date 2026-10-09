"""Render the packaged .map itself (composed as shellflow composes it) as its thumbnail."""
import sys
from pathlib import Path
A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A.parent / "jumper-design" / "src"))
import mujoco, imageio.v3 as iio
from shellflow.map_package import compose_default
from shellflow.package_io import read_archive

files = read_archive(Path(sys.argv[1]))
m, d, report = compose_default(files, profile=A.parent / "jumper-design" / "robots" / "jumper" / "profile.json")
m.vis.global_.offwidth, m.vis.global_.offheight = 1000, 750
m.vis.quality.shadowsize = 4096
m.vis.headlight.ambient[:] = [0.30, 0.24, 0.26]; m.vis.headlight.diffuse[:] = [0.30, 0.27, 0.25]
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=750, width=1000)
opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
cam = mujoco.MjvCamera(); cam.lookat[:] = [-0.25, 0, 0.18]; cam.distance = 1.9; cam.azimuth = 200; cam.elevation = -24
r.update_scene(d, camera=cam, scene_option=opt)
iio.imwrite(sys.argv[2], r.render()); print("wrote", sys.argv[2], "spawn:", report.get("spawn_support"))
