"""Minutes-long checks: rebuild the released packages and run every mode.

    pytest -m slow        (or: make regress)

Needs KingKong's `jumper` and `jumper-design` checkouts beside this repository.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from jumperkit.character import load
from jumperkit.maps import EXPORTED_BY, build_map
from jumperkit.sim import smoke
from jumperkit.skin import build_skin

GOLDEN = json.loads((Path(__file__).parent / "golden" / "alebrije.json").read_text())
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def alebrije_skin() -> Path:
    return build_skin(load("alebrije"), log=lambda *_: None)


def test_skin_rebuilds_byte_identical(alebrije_skin):
    digest = hashlib.sha256(alebrije_skin.read_bytes()).hexdigest()
    assert digest == GOLDEN["packages_sha256"]["alebrije-jumper.skin"]


def test_map_rebuilds_identical_but_for_its_producer(alebrije_skin):
    """v1.0.0 named its producer `scene/build_map.py`; now it is `jumperkit/maps.py`.
    That provenance field is the only difference allowed."""
    path = build_map(load("alebrije"), alebrije_skin, log=lambda *_: None)
    z = zipfile.ZipFile(path)
    members = {n: hashlib.sha256(z.read(n)).hexdigest()
               for n in sorted(z.namelist()) if n != "scene-package.json"}
    released = GOLDEN["released_map"]
    assert members == released["members"]
    manifest = json.loads(z.read("scene-package.json"))
    assert manifest.pop("exportedBy") == EXPORTED_BY
    assert manifest == released["manifest_without_exportedBy"]


def test_every_mode_runs_without_a_fall():
    results = smoke("alebrije")
    assert all(r["ok"] for r in results), [r for r in results if not r["ok"]]
