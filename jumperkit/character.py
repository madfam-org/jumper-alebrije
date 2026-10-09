"""A Jumper character: its colors, its shell decorations, and how they apply.

A character is a spec module `characters/<id>.py` that defines `CHARACTER`.
The `.skin` format fixes two rules every character must keep:

- the shell is one multi-material print of at most four filament colors, so
  the shell body and all its decorations share `shell_slots` (first = body);
- everything else is recolored per part (`limb_colors`), never reshaped.

Looks are display-only: added geoms never collide and weigh nothing, so a
character never changes the physics (see tests/test_display_only.py).
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .parts import Decor, Shell
from .paths import REPO

_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_HEX = re.compile(r"#[0-9A-Fa-f]{6}\Z")
MAX_SHELL_COLORS = 4

#: The middle legs, which `middle_swap` recolors so neighbouring legs differ.
MIDDLE_LEGS = ("LM", "RM")


def hex_rgba(h: str) -> list[float]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] + [1.0]


@dataclass(frozen=True)
class Character:
    id: str
    title: str
    description: str
    #: color name -> "#RRGGBB"
    palette: Mapping[str, str]
    #: shell body color first, then up to three decoration colors
    shell_slots: tuple[str, ...]
    #: link-name suffix (or "base_link") -> color name; e.g. "thigh_link": "turquesa"
    limb_colors: Mapping[str, str]
    #: decorate(decor, shell): add the shell decorations
    decorate: Callable[[Decor, Shell], None]
    #: color -> color on the middle legs
    middle_swap: Mapping[str, str] = field(default_factory=dict)
    version: str = "1.0.0"
    #: the plaza .map id; a released map keeps the id it shipped with
    map_id: str | None = None

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id):
            raise ValueError(f"character id must be lowercase snake_case: {self.id!r}")
        for name, value in self.palette.items():
            if not _HEX.fullmatch(value):
                raise ValueError(f"{self.id}: palette[{name!r}] is not #RRGGBB: {value!r}")
        if not 1 <= len(self.shell_slots) <= MAX_SHELL_COLORS:
            raise ValueError(f"{self.id}: the shell takes 1..{MAX_SHELL_COLORS} colors, "
                             f"got {len(self.shell_slots)}")
        used = list(self.shell_slots) + list(self.limb_colors.values()) + \
            list(self.middle_swap) + list(self.middle_swap.values())
        missing = sorted({c for c in used if c not in self.palette})
        if missing:
            raise ValueError(f"{self.id}: colors not in the palette: {missing}")

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def skin_id(self) -> str:
        return f"{self.id}-jumper"

    @property
    def plaza_map_id(self) -> str:
        return self.map_id or f"dia-de-muertos-plaza-{self.id}"

    # ── colors ──────────────────────────────────────────────────────────────
    @property
    def shell_color(self) -> str:
        return self.shell_slots[0]

    def rgba(self, color: str) -> list[float]:
        return hex_rgba(self.palette[color])

    def part_color(self, link: str) -> str | None:
        """The color of a visual part by its link name, or None to keep it."""
        if link == "base_link":
            return self.limb_colors.get("base_link")
        leg, _, rest = link.partition("_")
        color = self.limb_colors.get(rest)
        if color and leg in MIDDLE_LEGS:
            color = self.middle_swap.get(color, color)
        return color

    # ── shell decorations ───────────────────────────────────────────────────
    def decorations(self, shell: Shell | None = None) -> dict[str, trimesh.Trimesh]:
        """One merged mesh per color; every color is one of `shell_slots`."""
        d = Decor()
        self.decorate(d, shell or Shell())
        meshes = d.meshes()
        stray = set(meshes) - set(self.shell_slots)
        if stray:
            raise ValueError(f"{self.id}: decorations use colors outside shell_slots: "
                             f"{sorted(stray)}")
        return meshes


def available() -> list[str]:
    """Ids of the character specs in `characters/`."""
    return sorted(p.stem for p in (REPO / "characters").glob("*.py")
                  if not p.stem.startswith("_"))


def load(character_id: str) -> Character:
    module = importlib.import_module(f"characters.{character_id}")
    character = getattr(module, "CHARACTER", None)
    if not isinstance(character, Character):
        raise ValueError(f"characters/{character_id}.py must define CHARACTER = Character(...)")
    if character.id != character_id:
        raise ValueError(f"characters/{character_id}.py defines id {character.id!r}")
    return character


def apply_to_spec(spec, character: Character, decor: dict[str, trimesh.Trimesh] | None = None,
                  gloss: bool = True) -> None:
    """Recolor a robot MjSpec (before attach) and add the shell decorations.

    Visual-only: every added geom has contype=conaffinity=0 and density 0, so
    mass, inertia and collision stay the baseline's. `gloss` gives the colors a
    lacquer finish for rendering; packaged skins carry plain colors.
    """
    import mujoco

    cid = character.id
    for name in character.palette:
        spec.add_material(name=f"{cid}_{name}", rgba=character.rgba(name),
                          specular=0.55 if gloss else 0.2,
                          shininess=0.7 if gloss else 0.3, reflectance=0.0)
    for g in spec.geoms:
        if not g.name.endswith("_visual"):
            continue
        link = g.name[: -len("_visual")]
        color = character.shell_color if link == "upper_shell_link" else character.part_color(link)
        if color:
            g.material = f"{cid}_{color}"
            g.rgba = character.rgba(color)
    body = spec.body("upper_shell_link")
    for color, mesh in (decor or character.decorations()).items():
        spec.add_mesh(name=f"{cid}_decor_{color}",
                      uservert=np.asarray(mesh.vertices, dtype=float).ravel().tolist(),
                      userface=np.asarray(mesh.faces, dtype=int).ravel().tolist())
        body.add_geom(name=f"{cid}_decor_{color}", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=f"{cid}_decor_{color}", material=f"{cid}_{color}",
                      contype=0, conaffinity=0, density=0, group=2)
