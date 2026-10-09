"""`python -m jumperkit <command>`: every step, for any character.

    list                               the character specs in characters/
    check                              validate every spec (and the template)
    look      -c ID                    4-view studio sheet  -> build/ID/look.png
    stills    -c ID                    3 views in the plaza -> build/ID/plaza.png
    skin      -c ID                    build + verify the .skin -> build/ID/skin/
    skin-preview -c ID                 render the packaged skin (its thumbnail)
    map       -c ID                    build + verify the .map (needs the skin)
    map-preview  -c ID                 render the packaged map (its thumbnail)
    dist      -c ID                    copy the .skin and .map into dist/
    verify    -c ID                    re-run verify-package on dist/ files
    smoke     [-c ID]                  enter every mode; nobody may fall
    record    CHOREO OUT [-c ID|--stock]   simulate a choreography -> .npz
    video     -c ID [--take --shots --out --scale --step --stills]
    lineup    [-c ID,ID,...] [--take --at --out]   everyone at one frame
    energy    TAKE...                  per-second motion energy of takes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .paths import CHOREO, DIST, TAKES, build_dir, media_dir

DEFAULT_TAKE = TAKES / "take1.npz"
DEFAULT_SHOTS = CHOREO / "take1.shots.json"


def _character(args):
    from .character import load
    return load(args.character)


def cmd_list(args) -> int:
    from .character import available, load
    for cid in available():
        c = load(cid)
        print(f"{cid:16s} {c.title:28s} shell: {', '.join(c.shell_slots)}")
    return 0


def cmd_check(args) -> int:
    import importlib

    from .character import Character, available, load
    failed = 0
    for cid in available() + ["_template"]:
        try:
            c = load(cid) if cid != "_template" else \
                importlib.import_module("characters._template").CHARACTER
            assert isinstance(c, Character)
            meshes = c.decorations()
            faces = {k: len(v.faces) for k, v in meshes.items()}
            print(f"ok    {cid:16s} slots={list(c.shell_slots)} decoration faces={faces}")
        except Exception as e:  # report every spec, then fail
            failed += 1
            print(f"FAIL  {cid:16s} {type(e).__name__}: {e}")
    return 1 if failed else 0


def cmd_look(args) -> int:
    from .previews import look_sheet
    out = Path(args.out) if args.out else build_dir(args.character) / "look.png"
    print("wrote", look_sheet(_character(args), out))
    return 0


def cmd_stills(args) -> int:
    from .previews import plaza_stills
    out = Path(args.out) if args.out else build_dir(args.character) / "plaza.png"
    print("wrote", plaza_stills(_character(args), out))
    return 0


def cmd_skin(args) -> int:
    from .skin import AUTHOR, LICENSE, build_skin
    build_skin(_character(args), version=args.version, author=args.author or AUTHOR,
               license_id=args.license or LICENSE,
               preview=Path(args.preview) if args.preview else None)
    return 0


def cmd_skin_preview(args) -> int:
    from .previews import skin_preview
    sim = build_dir(args.character) / "skin" / "simulation"
    if not sim.is_dir():
        sys.exit(f"no {sim}: run `skin -c {args.character}` first")
    out = Path(args.out) if args.out else media_dir(args.character) / "skin-preview.png"
    if out.exists() and not args.force:
        print(f"kept {out} (it is embedded in the package; --force to re-render)")
        return 0
    print("wrote", skin_preview(sim, out))
    return 0


def cmd_map(args) -> int:
    from .maps import build_map
    from .skin import skin_path
    c = _character(args)
    skin = Path(args.skin) if args.skin else skin_path(c)
    if not skin.is_file():
        sys.exit(f"no {skin}: run `skin -c {c.id}` first")
    build_map(c, skin, preview=Path(args.preview) if args.preview else None)
    return 0


def cmd_map_preview(args) -> int:
    from .maps import map_path
    from .previews import map_preview
    c = _character(args)
    src = map_path(c)
    if not src.is_file():
        sys.exit(f"no {src}: run `map -c {c.id}` first")
    out = Path(args.out) if args.out else media_dir(c.id) / "map-preview.png"
    if out.exists() and not args.force:
        print(f"kept {out} (it is embedded in the package; --force to re-render)")
        return 0
    print("wrote", map_preview(src, out))
    return 0


def _dist_files(c) -> list[Path]:
    from .maps import map_path
    from .skin import skin_path
    return [skin_path(c), map_path(c)]


def cmd_dist(args) -> int:
    c = _character(args)
    DIST.mkdir(exist_ok=True)
    for src in _dist_files(c):
        if not src.is_file():
            sys.exit(f"no {src}: build it first (skin, then map)")
        shutil.copy2(src, DIST / src.name)
        print("copied", DIST / src.name)
    return 0


def cmd_verify(args) -> int:
    from .maps import verify_map
    from .skin import verify_skin
    c = _character(args)
    skin, plaza = (DIST / p.name for p in _dist_files(c))
    ok = verify_skin(skin).get("ok") and verify_map(plaza).get("ok")
    return 0 if ok else 1


def cmd_smoke(args) -> int:
    from .sim import smoke
    results = smoke(None if args.stock else args.character,
                    args.modes.split(",") if args.modes else None)
    for r in results:
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['mode']:18s} entered={r['entered']} "
              f"worst tilt {r['worst_tilt_deg']:5.1f} deg")
    return 0 if all(r["ok"] for r in results) else 1


def cmd_record(args) -> int:
    from .sim import save_take, simulate
    t0 = time.time()
    res = simulate(json.loads(Path(args.choreography).read_text()),
                   None if args.stock else args.character)
    save_take(res, Path(args.out))
    print(f"{args.out}: {len(res['qpos'])} frames, worst tilt {res['tilt'].max():.1f} deg, "
          f"{time.time() - t0:.0f}s wall")
    for e in res["log"]:
        print("  fsm:", e)
    return 0


def cmd_video(args) -> int:
    from .render import render_take
    c = _character(args)
    out = Path(args.out) if args.out else build_dir(c.id) / "video.mp4"
    stills = [float(x) for x in args.stills.split(",")] if args.stills else None
    render_take(Path(args.take), Path(args.shots), out, c, scale=args.scale, step=args.step,
                stills=stills)
    return 0


def cmd_lineup(args) -> int:
    from .character import available, load
    from .previews import lineup
    ids = args.character.split(",") if args.character else available()
    kwargs = {"at": args.at} if args.at is not None else {}
    out = lineup([load(i) for i in ids], Path(args.take), Path(args.out), **kwargs)
    print("wrote", out)
    return 0


def cmd_energy(args) -> int:
    import numpy as np
    for path in args.takes:
        z = np.load(path)
        q, fps = z["qpos"], int(z["fps"])
        js = np.abs(np.diff(q[:, 7:29], axis=0)).mean(axis=1) * fps  # rad/s, mean over joints
        n = len(js) // fps
        per_s = js[: n * fps].reshape(n, fps).mean(axis=1)
        zs = q[: n * fps, 2].reshape(n, fps)
        bob = (zs.max(axis=1) - zs.min(axis=1)) * 1000
        print(Path(path).name, "| tilt max %.1f" % z["tilt"].max())
        print("  speed rad/s:", " ".join(f"{v:3.1f}" for v in per_s))
        print("  bob mm:     ", " ".join(f"{v:3.0f}" for v in bob))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m jumperkit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def command(name, fn, character=True, default="alebrije"):
        sp = sub.add_parser(name)
        if character:
            sp.add_argument("-c", "--character", default=default,
                            help=f"character id in characters/ (default: {default})")
        sp.set_defaults(fn=fn)
        return sp

    command("list", cmd_list, character=False)
    command("check", cmd_check, character=False)
    command("look", cmd_look).add_argument("--out")
    command("stills", cmd_stills).add_argument("--out")
    sp = command("skin", cmd_skin)
    sp.add_argument("--version")
    sp.add_argument("--author")
    sp.add_argument("--license")
    sp.add_argument("--preview", help="PNG embedded in the package (default media/ID/skin-preview.png)")
    sp = command("skin-preview", cmd_skin_preview)
    sp.add_argument("--out")
    sp.add_argument("--force", action="store_true", help="re-render an existing thumbnail")
    sp = command("map", cmd_map)
    sp.add_argument("--skin", help="default robot (default: the character's built skin)")
    sp.add_argument("--preview", help="PNG embedded in the package (default media/ID/map-preview.png)")
    sp = command("map-preview", cmd_map_preview)
    sp.add_argument("--out")
    sp.add_argument("--force", action="store_true", help="re-render an existing thumbnail")
    command("dist", cmd_dist)
    command("verify", cmd_verify)
    sp = command("smoke", cmd_smoke)
    sp.add_argument("--stock", action="store_true", help="the stock robot instead")
    sp.add_argument("--modes", help="comma list (default: all)")
    sp = command("record", cmd_record)
    sp.add_argument("choreography")
    sp.add_argument("out")
    sp.add_argument("--stock", action="store_true", help="the stock robot instead")
    sp = command("video", cmd_video)
    sp.add_argument("--take", default=str(DEFAULT_TAKE))
    sp.add_argument("--shots", default=str(DEFAULT_SHOTS))
    sp.add_argument("--out")
    sp.add_argument("--scale", type=float, default=1.0, help="0.5 renders 540x960")
    sp.add_argument("--step", type=int, default=1, help="render every Nth frame")
    sp.add_argument("--stills", help="comma list of seconds: PNGs instead of a video")
    sp = command("lineup", cmd_lineup, default="")
    sp.add_argument("--take", default=str(DEFAULT_TAKE))
    sp.add_argument("--at", type=float, default=None, help="seconds into the take "
                    "(default: standing after the bow in take1)")
    sp.add_argument("--out", default=str(build_dir("lineup") / "lineup.png"))
    sp = command("energy", cmd_energy, character=False)
    sp.add_argument("takes", nargs="+")

    args = p.parse_args(argv)
    return args.fn(args)
