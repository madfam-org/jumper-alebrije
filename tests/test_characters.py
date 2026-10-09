"""Every character spec loads, keeps the .skin format's rules, and fails loudly when it doesn't."""

import importlib
import re

import pytest

from jumperkit.character import MAX_SHELL_COLORS, Character, available, load
from jumperkit.parts import Decor, Shell

PACKAGE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")  # shellflow's package id rule
SPECS = available() + ["_template"]


def _spec(cid: str) -> Character:
    if cid == "_template":
        return importlib.import_module("characters._template").CHARACTER
    return load(cid)


@pytest.fixture(scope="module")
def shell() -> Shell:
    return Shell()


def test_alebrije_is_available():
    assert "alebrije" in available()
    assert "_template" not in available()


@pytest.mark.parametrize("cid", SPECS)
def test_spec_is_valid(cid, shell):
    c = _spec(cid)
    assert 1 <= len(c.shell_slots) <= MAX_SHELL_COLORS
    assert PACKAGE_ID.fullmatch(c.skin_id) and PACKAGE_ID.fullmatch(c.plaza_map_id)
    assert c.title.strip() and c.description.strip()
    meshes = c.decorations(shell)
    assert set(meshes) <= set(c.shell_slots)
    assert all(len(m.faces) for m in meshes.values())


def test_alebrije_part_colors():
    c = load("alebrije")
    assert c.part_color("base_link") == "morado"
    assert c.part_color("LR_thigh_link") == "turquesa"
    assert c.part_color("LM_thigh_link") == "cobalto"  # middle legs swap
    assert c.part_color("display_module_link") is None  # kept as is


def _kwargs(**over):
    base = dict(id="probe", title="Probe", description="A probe.",
                palette={"a": "#000000", "b": "#FFFFFF"}, shell_slots=("a", "b"),
                limb_colors={"thigh_link": "a"}, decorate=lambda d, s: None)
    base.update(over)
    return base


@pytest.mark.parametrize("over, message", [
    (dict(id="Bad-Id"), "snake_case"),
    (dict(palette={"a": "red", "b": "#FFFFFF"}), "#RRGGBB"),
    (dict(palette={c: "#000000" for c in "abcde"}, shell_slots=tuple("abcde")), "1..4"),
    (dict(limb_colors={"thigh_link": "missing"}), "not in the palette"),
])
def test_bad_specs_are_rejected(over, message):
    with pytest.raises(ValueError, match=message):
        Character(**_kwargs(**over))


def test_decorations_outside_the_shell_colors_are_rejected(shell):
    from jumperkit import parts as P

    def decorate(d: Decor, s: Shell):
        P.medallion(d, s, "a", "b", "c")  # "c" is a limb color, not a shell slot

    c = Character(**_kwargs(palette={"a": "#000000", "b": "#FFFFFF", "c": "#FF0000"},
                            decorate=decorate))
    with pytest.raises(ValueError, match="outside shell_slots"):
        c.decorations(shell)
