"""The alebrije, released as v1.0.0, must not drift.

`golden/alebrije.json` was captured from the v1.0.0 code before the refactor
into jumperkit: the decoration meshes (hash of vertices and faces per color),
the per-part color overrides packaged in the skin, and the colors of every
rendered part. The slow tests rebuild the packages themselves.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jumperkit.character import apply_to_spec, load
from jumperkit.sim import robot_spec
from jumperkit.skin import visual_overrides

GOLDEN = json.loads((Path(__file__).parent / "golden" / "alebrije.json").read_text())


def mesh_hash(m) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(m.vertices, dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(m.faces, dtype=np.int64)).tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def alebrije():
    return load("alebrije")


def test_decorations_match(alebrije):
    meshes = alebrije.decorations()
    assert list(meshes) == GOLDEN["decoration_order"]
    for color, mesh in meshes.items():
        assert mesh_hash(mesh) == GOLDEN["decorations"][color]["sha256"], color


def test_packaged_part_colors_match(alebrije):
    assert visual_overrides(alebrije) == GOLDEN["visual_overrides"]


def test_rendered_colors_match(alebrije):
    spec = robot_spec()
    apply_to_spec(spec, alebrije)
    m = spec.compile()
    look = {m.geom(i).name: [round(float(x), 6) for x in m.geom_rgba[i]] for i in range(m.ngeom)
            if m.geom(i).name.endswith("_visual") or "decor" in m.geom(i).name}
    assert look == GOLDEN["render_rgba"]


def test_identity_is_the_released_one(alebrije):
    assert alebrije.skin_id == "alebrije-jumper"
    assert alebrije.plaza_map_id == "dia-de-muertos-plaza"
