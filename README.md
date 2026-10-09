# Jumper Alebrije · Día de Muertos edition

A Mexican folk-art look for [Jumper](https://github.com/KingKongRobotics/jumper), KingKong
Robotics' open-source crab robot, and a Day of the Dead plaza for it to dance in. Our entry
to the [Crab Robot Open Motion Challenge](https://beunlimited.me/en/events/crab-robot-challenge-2026),
and `jumperkit`, the small toolkit that builds it, so new characters take one spec file each.

By [MADFAM](https://madfam.io), Cuernavaca, Mexico.

![The alebrije skin, rendered from the packaged .skin](media/alebrije/skin-preview.png)

## What is here

| Deliverable | What it is | Checked with |
|---|---|---|
| `alebrije-jumper.skin` | Display-only appearance, `kk-skin-package/3`: a pink shell painted with yellow, turquoise and black (dot rings, a sun medallion, a spiny crest, a zigzag crown, feathered wings, eyestalks with big yellow eyes), plus a lacquered rainbow palette on every leg segment | `shellflow verify-package --profile robots/jumper/profile.json --mujoco` → `ok` |
| `dia-de-muertos-plaza.map` | Scene, `kk-scene-package/2`: terracotta tiles, a marigold petal path and ring, a three-tier ofrenda with candles, sugar skulls and pan de muerto, papel picado, bougainvillea on a cobalt adobe wall, a dusk sky. Bundles the alebrije Jumper as its default robot | `shellflow verify-package --capability rigid --capability jumper --mujoco` → `ok` |
| `jumper-alebrije-dia-de-muertos.mp4` | 43 s vertical video: the alebrije Jumper dancing KingKong's own trained *Brazilian* and *Dream Wings* policies in the plaza, then bowing | — |

The `.skin` and `.map` are built only with KingKong's own
[jumper-design](https://github.com/KingKongRobotics/jumper-design) toolchain
(`shellflow.simulation.assemble`, `export-skin`, `export_environment`, `verify-package`).
Download them from this repository's [Releases](https://github.com/madfam-org/jumper-alebrije/releases),
or rebuild them with the steps below.

## How it works

**Looks are generated, not hand-modeled.** A character spec (`characters/<id>.py`) picks colors
and calls parts from `jumperkit/parts.py`: rings of dots, a medallion, a crest of spines, a zigzag
crown, feathered wings, eyestalks. Parts are ray-cast onto the real upper-shell mesh and oriented
along its surface normals, then merged one mesh per color. The `.skin` format paints a shell
from at most four filament colors (one multi-material print); the legs are recolored through
the format's per-part `visual_overrides`.

**Looks are display-only.** Every added geom is massless and non-colliding. Running the same 40 s
dance with the stock robot and with the alebrije gives bit-identical trajectories (maximum
`|Δqpos|` = 0.0 over 1,200 frames), and `tests/test_display_only.py` checks every character
statically: same masses, inertias, joints and colliders as the stock robot.

**The motion is KingKong's.** `jumperkit/sim.py` runs the bundle in `jumper/out/bundle_example/`
without PyTorch, so it works on an Intel Mac or any CPU:

- the bundle's compiled controller (`runtime/mjlab/<platform>/controller.so`) builds every
  observation, switches modes on the bundle's own key bindings and decodes actions;
- its ONNX policies run through onnxruntime;
- MuJoCo steps the robot at 1 kHz with the collision, friction, contact and armature settings
  of `tasks/jumper/common/constants.py` and the servo's measured torque–speed curve from
  `assets/jumper/motor/motor_config.yaml`.

All eleven trigger-able modes (four dances, four gestures, jump, both claws) run without a fall.

**Simulate once, render any look.** Because looks never touch the physics, a choreography is
simulated once (`build/takes/take1.npz`) and every character renders from it: the video, stills,
and a lineup of all characters at the same moment.

## Layout

```text
characters/     one spec per look: alebrije.py, _template.py (copy it to start)
jumperkit/      the toolkit: parts, character, sim, scene, skin, maps, previews, render, cli
choreo/         take1.json (key events) and take1.shots.json (cameras, captions)
media/<id>/     committed images; skin-preview.png and map-preview.png are embedded in packages
tests/          fast checks (`make test`) and slow rebuilds (`make regress`)
build/          generated, per character (ignored)       dist/   release copies (ignored)
```

## Setup

Python 3.10+. Put KingKong's two repositories next to this one (or set `JUMPERKIT_WORKSPACE`):

```text
workspace/
  jumper/          git clone https://github.com/KingKongRobotics/jumper
  jumper-design/   git clone https://github.com/KingKongRobotics/jumper-design  (with Git LFS files)
  alebrije/        this repository
```

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e "alebrije[dev]"
pip install -e "jumper-design[sim,dev]"
```

Built against `jumper` @ `61d0652` and `jumper-design` @ `918d192`. MuJoCo 3.10 was used
because it is the last release with Intel-Mac wheels; KingKong's training pins 3.11.

## Commands

From `alebrije/`, `make help` lists the targets; each takes `CHAR=<id>` (default `alebrije`).
They wrap `python -m jumperkit <command>` (`--help` for options).

```sh
make test                # seconds: specs valid, looks display-only, alebrije unchanged
make regress             # minutes: rebuild the released alebrije byte-for-byte, run every mode
make look CHAR=<id>      # 4-view studio sheet -> build/<id>/look.png
make package CHAR=<id>   # .skin + .map, thumbnails, dist/, KingKong's verify-package
make video CHAR=<id>     # the 43 s vertical video -> build/<id>/video.mp4 (~6 min on this Mac)
make lineup              # every character at the same moment -> build/lineup/lineup.png
```

## Add a character

1. `cp characters/_template.py characters/<id>.py`, set `id="<id>"`, a title and description.
2. Pick a palette. `shell_slots` is the shell body color plus at most three decoration colors;
   `limb_colors` can use any palette color. Stay clear of licensed characters: the challenge
   disqualifies IP infringement.
3. Write `decorate(d, shell)` with parts from `jumperkit/parts.py`, or custom parts built from
   its primitives (`disc`, `ring`, `spine`, `feather`, `align`, `shell.top(x, y)`).
4. `make check` and `make test` (the new spec is picked up automatically), then `make look CHAR=<id>`
   and iterate. Compare with the others in `make lineup`.
5. For a release: `make package CHAR=<id>` (renders the thumbnails into `media/<id>/` and embeds
   them), then `make video CHAR=<id>`.

Thumbnails in `media/<id>/` are part of a package's bytes, so `skin-preview` and `map-preview`
keep an existing one unless given `--force`.

## Limits, stated plainly

- Everything here is simulation. Nothing ran on a physical Jumper.
- Skins are display-only, like every `/3` skin: no mounting interface was engineered and
  nothing was printed or fit-tested (`physical_fit_tested=false` in the package).
- The motions are KingKong's trained policies from the example bundle; we did not train new ones.
- The video has no soundtrack.

## License

Apache-2.0 for the code, the generated meshes and the packages built from them. KingKong's
toolkit, robot model and policies keep their own Apache-2.0 terms and notices; see their
repositories. MuJoCo is Apache-2.0.
