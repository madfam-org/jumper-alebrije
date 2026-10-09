"""jumperkit: looks, scenes and videos for KingKong Robotics' Jumper crab robot.

Each look is a character spec in `characters/<id>.py`; everything else here is
shared machinery:

- `parts`: the shell-decoration library (dots, rings, spines, feathers, eyes)
- `character`: the `Character` spec, its loader, and applying a look to a model
- `sim`: KingKong's exported app (controller + ONNX policies) in native MuJoCo
- `scene`: the Día de Muertos plaza
- `skin`, `maps`: `.skin` / `.map` packages through jumper-design's toolchain
- `previews`, `render`: stills, lineups and the vertical video

Run `python -m jumperkit --help` from the repository root.
"""

__version__ = "1.1.0"
