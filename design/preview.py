"""Render still previews of the alebrije Jumper from several angles."""
import sys
from pathlib import Path
import mujoco, numpy as np, imageio.v3 as iio
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from jumper_sim import App, attach_robot, robot_spec
from alebrije import apply_to_spec

def studio():
    w = mujoco.MjSpec()
    w.option.timestep = 0.001; w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.visual.global_.offwidth = 1600; w.visual.global_.offheight = 1600
    w.visual.headlight.ambient = [0.35, 0.35, 0.38]; w.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    w.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                  rgb1=[0.93, 0.89, 0.82], rgb2=[0.88, 0.83, 0.75], width=512, height=512)
    w.add_material(name="floor", textures=["", "grid"], texrepeat=[6, 6], texuniform=True)
    w.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], material="floor",
                         contype=1, conaffinity=1)
    w.worldbody.add_light(pos=[0.6, -0.8, 1.6], dir=[-0.35, 0.45, -1], castshadow=True, diffuse=[0.7, 0.7, 0.68])
    return w

if __name__ == "__main__":
    out = Path(sys.argv[1])
    w = studio(); r = robot_spec(); apply_to_spec(r); attach_robot(w, r)
    m = w.compile(); d = mujoco.MjData(m); App(m).reset(d)
    ren = mujoco.Renderer(m, height=800, width=800)
    frames = []
    for az, el, dist in [(150, -25, 0.75), (210, -20, 0.75), (90, -60, 0.8), (0, -12, 0.7)]:
        cam = mujoco.MjvCamera(); cam.lookat[:] = [0.0, 0, 0.11]; cam.distance = dist; cam.azimuth = az; cam.elevation = el
        opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
        ren.update_scene(d, camera=cam, scene_option=opt); frames.append(ren.render())
    grid = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
    iio.imwrite(out, grid); print("wrote", out, grid.shape)
