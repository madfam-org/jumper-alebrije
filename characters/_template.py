"""Template for a new character: copy to `characters/<id>.py` and edit.

Rules the `.skin` format sets (checked by `python -m jumperkit check`):
- `shell_slots`: the shell body color first, then at most three decoration
  colors; every decoration must use one of them;
- `limb_colors` recolor whole parts by link-name suffix; they can use any
  palette color, including ones not on the shell.

Parts live in `jumperkit/parts.py`; pass your own colors and, if you need to,
sizes and positions. For a custom part, build it from `P.disc`, `P.ring`,
`P.spine`, `P.feather` and `P.align`, find the surface with `shell.top(x, y)`,
and add it with `d.add(color, mesh)`. Coordinates are meters in the
`upper_shell_link` frame: +X is the face, the dome spans about x -0.10..0.10,
y -0.09..0.09, and peaks near z 0.04.

Then:  python -m jumperkit look --character <id>   (4-view preview)
       python -m jumperkit skin --character <id>   (build + verify the .skin)
"""

from jumperkit import parts as P
from jumperkit.character import Character

PALETTE = {
    "body": "#2E86DE",
    "accent": "#F6C90E",
    "light": "#FFFFFF",
    "dark": "#1B1B1B",
}


def decorate(d: P.Decor, shell: P.Shell) -> None:
    P.dot_rings(d, shell, [(0.045, 0.040, 10, 0.006, ["accent"], "dark")])
    P.eyestalks(d, shell, stalk="accent", eye="light", pupil="dark", lid="accent")


CHARACTER = Character(
    id="template",
    title="Template Jumper",
    description="A starting point: one ring of dots and a pair of eyes.",
    palette=PALETTE,
    shell_slots=("body", "accent", "light", "dark"),
    limb_colors={"base_link": "dark", "thigh_link": "body", "calf_link": "accent"},
    decorate=decorate,
)
