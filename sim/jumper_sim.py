"""Run Jumper's exported app (controller + ONNX policies) in native MuJoCo.

KingKong's own `scripts/play.py --app` goes through mjlab, which needs torch>=2.7.
There is no such wheel for an Intel Mac, so this drives the same pieces directly:

- the bundle's compiled controller (`runtime/mjlab/<platform>/controller.so`),
  which builds every observation, switches modes and decodes actions;
- the bundle's ONNX policies, through onnxruntime;
- the robot MJCF from `jumper/assets/jumper/jumper.xml`, with the collision,
  contact and armature settings `tasks/jumper/common/constants.py` applies;
- the servo's measured torque-speed curve from `assets/jumper/motor/motor_config.yaml`.

The physics is MuJoCo at 1 kHz, the controller ticks at the app's fastest mode
rate (200 Hz), each mode infers at its own rate, as in `mjrl/app_play.py`.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
JUMPER = ROOT / "jumper"
BUNDLE = JUMPER / "out" / "bundle_example" / "jumper"
ROBOT_XML = JUMPER / "assets" / "jumper" / "jumper.xml"

# tasks/jumper/common/constants.py
LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")
FEET = (
    "LF_palm_pad_b_link", "RF_palm_pad_b_link",
    "LM_foot_tip_link", "RM_foot_tip_link",
    "LR_foot_tip_link", "RR_foot_tip_link",
)
STAND_Z = 0.10647
ARMATURE = 0.0015
FOOT_FRICTION = (1.2, 0.01, 0.01)
CONTACT_SOLREF = (0.02, 1.0)
CONTACT_SOLIMP = (0.8, 0.90, 0.002, 0.5, 2.0)
GROUND_BIT = 1
LEG_BIT = {leg: 1 << (i + 1) for i, leg in enumerate(LEGS)}
ALL_LEG_BITS = sum(LEG_BIT.values())

# assets/jumper/motor/motor_config.yaml, joint_servo.torque_speed
RPM = 2.0 * math.pi / 60.0
PLATEAU = 1.7464
CORNER = 293.5 * RPM
DECAY = 240.5 * RPM
CUTOFF = 611.0 * RPM

PREFIX = "robot/"


def servo_torque_limit(speed: np.ndarray) -> np.ndarray:
    w = np.abs(speed)
    limit = PLATEAU * np.exp(-np.clip(w - CORNER, 0.0, None) / DECAY)
    return np.where(w >= CUTOFF, 0.0, limit)


def robot_spec(xml: Path = ROBOT_XML) -> mujoco.MjSpec:
    """The robot as training sees it: contact classes, friction, armature."""
    spec = mujoco.MjSpec.from_file(str(xml))
    feet = {f"{f}_meshcol" for f in FEET}
    for g in spec.geoms:
        if not g.name.endswith("_meshcol"):
            continue
        leg = g.name[:2] if g.name[2:3] == "_" and g.name[:2] in LEGS else None
        g.contype = GROUND_BIT | (LEG_BIT[leg] if leg else 0)
        g.conaffinity = (ALL_LEG_BITS & ~LEG_BIT[leg]) if leg else 0
        foot = g.name in feet
        g.condim = 3 if foot else 1
        g.priority = 1 if foot else 0
        if foot:
            g.friction = FOOT_FRICTION
        g.solref = CONTACT_SOLREF
        g.solimp = CONTACT_SOLIMP
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.armature = ARMATURE
    return spec


def attach_robot(world: mujoco.MjSpec, robot: mujoco.MjSpec,
                 pos=(0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> None:
    """Attach the robot under a world frame; its names gain `robot/`."""
    half = math.radians(yaw_deg) / 2.0
    frame = world.worldbody.add_frame(
        pos=list(pos), quat=[math.cos(half), 0.0, 0.0, math.sin(half)])
    world.attach(robot, frame=frame, prefix=PREFIX)


def _load_controller(bundle: Path):
    manifest = json.loads((bundle / "bundle.json").read_text("utf-8"))
    key = "macosx-universal2" if platform.system() == "Darwin" else "linux-x86_64"
    path = bundle / manifest["runtimes"]["mjlab"]["extensions"][key]["file"]
    loader = importlib.machinery.ExtensionFileLoader("mjrl_fsm", str(path))
    spec = importlib.util.spec_from_file_location("mjrl_fsm", path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module, manifest


@dataclass
class App:
    """The deployment controller and its policies, ticked like the board ticks it."""

    model: mujoco.MjModel
    bundle: Path = BUNDLE
    control_hz: float = field(init=False)

    def __post_init__(self) -> None:
        mjrl_fsm, manifest = _load_controller(self.bundle)
        b = self.bundle
        contracts = {m: (b / e["contract"]).read_text("utf-8") for m, e in manifest["modes"].items()}
        trajectories = {m: (b / e["trajectory"]).read_text("utf-8")
                        for m, e in manifest["modes"].items() if e.get("trajectory")}
        parsed = {m: json.loads(t) for m, t in contracts.items()}
        self.wire: list[str] = next(iter(parsed.values()))["wire_joint_order"]
        self.contracts = parsed
        self.control_hz = max(float(c["control"]["control_hz"]) for c in parsed.values())

        m = self.model
        self.jnt = [m.joint(PREFIX + n).id for n in self.wire]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jnt])
        self.vadr = np.array([m.jnt_dofadr[j] for j in self.jnt])
        free = m.joint(PREFIX + "floating_base").id
        self.root_q = m.jnt_qposadr[free]
        self.root_v = m.jnt_dofadr[free]
        lo = [float(m.jnt_range[j][0]) for j in self.jnt]
        hi = [float(m.jnt_range[j][1]) for j in self.jnt]

        self.fsm = mjrl_fsm.Fsm(
            (b / "controller.toml").read_text("utf-8"),
            contracts,
            {"joint_names": self.wire, "joint_pos_lo": lo, "joint_pos_hi": hi,
             "output_rate_hz": self.control_hz, "gait_period": None,
             "gait_gate_threshold": None},
            trajectories=trajectories,
            operators=True,
        )
        self.sessions = {
            mode: ort.InferenceSession(str(b / e["models"]["onnx"]),
                                       providers=["CPUExecutionProvider"])
            for mode, e in manifest["modes"].items()
        }
        home = parsed["locomotion"]["default_joint_pos"]
        self.home = np.array([home[n] for n in self.wire])
        self.targets = self.home.copy()
        self.kp = np.full(len(self.wire), 10.0)
        self.kd = np.full(len(self.wire), 0.5)
        self.tau = np.zeros(len(self.wire))
        self.tick_count = 0
        self.log: list[tuple] = []

    def reset(self, data: mujoco.MjData, pos=(0.0, 0.0), z: float = STAND_Z) -> None:
        q0 = self.root_q
        data.qpos[q0:q0 + 2] += np.asarray(pos)  # relative to the attach frame
        data.qpos[q0 + 2] += z
        data.qpos[self.qadr] = self.home
        mujoco.mj_forward(self.model, data)

    def tick(self, data: mujoco.MjData, now_us: int,
             downs: list[str] = (), ups: list[str] = ()) -> str | None:
        q = data.qpos[self.qadr]
        qd = data.qvel[self.vadr]
        quat = data.qpos[self.root_q + 3:self.root_q + 7]
        ang_b = data.qvel[self.root_v + 3:self.root_v + 6]  # free joint: body frame
        self.fsm.set_state(q.tolist(), qd.tolist(), self.tau.tolist(),
                           quat.tolist(), ang_b.tolist(), now_us)
        for k in downs:
            self.fsm.set_key(k, True, now_us)
        for k in ups:
            self.fsm.set_key(k, False, now_us)
        self.fsm.set_pad_frame([], {}, connected=False, now_us=now_us)
        mode = self.fsm.tick(now_us)
        if mode is not None:
            s = self.sessions[mode]
            obs = np.asarray([self.fsm.observation()], dtype=np.float32)
            action = s.run(None, {s.get_inputs()[0].name: obs})[0][0]
            self.fsm.resume([float(v) for v in action])
        self.log.extend(self.fsm.take_log())
        self.targets = np.asarray(self.fsm.positions())
        self.kp = np.asarray(self.fsm.kp())
        self.kd = np.asarray(self.fsm.kd())
        self.tick_count += 1
        return self.fsm.mode

    def apply_torque(self, data: mujoco.MjData) -> None:
        q = data.qpos[self.qadr]
        qd = data.qvel[self.vadr]
        tau = self.kp * (self.targets - q) - self.kd * qd
        lim = servo_torque_limit(qd)
        self.tau = np.clip(tau, -lim, lim)
        data.qfrc_applied[self.vadr] = self.tau


def base_state(app: App, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    """Root position and tilt from vertical, in degrees."""
    q0 = app.root_q
    pos = data.qpos[q0:q0 + 3].copy()
    w, x, y, z = data.qpos[q0 + 3:q0 + 7]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return pos, math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
