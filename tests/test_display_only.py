"""A look never changes the physics: same masses, inertias, joints and colliders.

This is the static form of the dynamic check (stock and alebrije runs of a
40 s dance gave bit-identical trajectories): every geom a look adds must be
non-colliding and massless, and nothing the physics reads may differ.
"""

import importlib

import mujoco
import numpy as np
import pytest

from jumperkit.character import apply_to_spec, available, load
from jumperkit.sim import robot_spec

SPECS = available() + ["_template"]


def _spec(character=None) -> mujoco.MjSpec:
    spec = robot_spec()
    if character is not None:
        apply_to_spec(spec, character)
    return spec


def _compile(character=None) -> mujoco.MjModel:
    return _spec(character).compile()


@pytest.fixture(scope="module")
def stock() -> mujoco.MjModel:
    return _compile()


def _colliders(m: mujoco.MjModel) -> dict:
    out = {}
    for i in range(m.ngeom):
        if m.geom_contype[i] or m.geom_conaffinity[i]:
            out[m.geom(i).name] = (m.geom_type[i], m.geom_bodyid[i], tuple(m.geom_size[i]),
                                   tuple(m.geom_friction[i]), tuple(m.geom_pos[i]),
                                   tuple(m.geom_quat[i]), m.geom_condim[i])
    return out


@pytest.mark.parametrize("cid", SPECS)
def test_look_is_display_only(cid, stock):
    c = importlib.import_module("characters._template").CHARACTER if cid == "_template" else load(cid)
    spec = _spec(c)
    m = spec.compile()
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos",
                  "jnt_range", "dof_armature", "dof_damping", "dof_frictionloss"):
        assert np.array_equal(getattr(m, field), getattr(stock, field)), field
    assert _colliders(m) == _colliders(stock)
    stock_names = {stock.geom(i).name for i in range(stock.ngeom)}
    added = [i for i in range(m.ngeom) if m.geom(i).name not in stock_names]
    assert added, "a look should add its decorations"
    for i in added:
        name = m.geom(i).name
        assert m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0, name
        # Massless even on a body without a declared <inertial>, where geom
        # density would otherwise feed the body's mass.
        assert spec.geom(name).density == 0, name
