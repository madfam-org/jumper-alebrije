"""Headless check: every mode in the bundle is entered and the robot stays upright.

    python sim/smoke_test.py

Each case presses the bundle's own key binding from standing, runs a few
seconds of physics with the alebrije look, and fails if the controller never
enters the mode or the body tilts past 30 degrees.
"""

from __future__ import annotations

import sys

from record import simulate

CTRL = "key_left_ctrl"
CASES = {
    # mode: (seconds, events)
    "dance_crab": (12, [[0.4, "down", CTRL], [0.5, "down", "key_1"], [0.7, "up", "key_1"]]),
    "dance_brazilian": (12, [[0.4, "down", CTRL], [0.5, "down", "key_2"], [0.7, "up", "key_2"]]),
    "dance_maze": (12, [[0.4, "down", CTRL], [0.5, "down", "key_3"], [0.7, "up", "key_3"]]),
    "dance_dream_wings": (12, [[0.4, "down", CTRL], [0.5, "down", "key_4"], [0.7, "up", "key_4"]]),
    "gesture_hello": (9, [[0.5, "down", "key_1"], [0.7, "up", "key_1"]]),
    "gesture_bow": (10, [[0.5, "down", "key_2"], [0.7, "up", "key_2"]]),
    "gesture_paw": (9, [[0.5, "down", "key_3"], [0.7, "up", "key_3"]]),
    "gesture_salute": (9, [[0.5, "down", "key_4"], [0.7, "up", "key_4"]]),
    "jump": (5, [[0.5, "down", "key_space"], [0.7, "up", "key_space"]]),
    "claw_left": (5, [[0.5, "down", "key_v"], [0.7, "up", "key_v"]]),
    "claw_right": (5, [[0.5, "down", "key_b"], [0.7, "up", "key_b"]]),
}

if __name__ == "__main__":
    failed = []
    for mode, (seconds, events) in CASES.items():
        r = simulate({"seconds": seconds, "fps": 30, "events": events})
        entered = any(e[2] == mode for e in r["log"])
        worst = float(r["tilt"][30:].max())
        ok = entered and worst < 30.0
        print(f"{'ok  ' if ok else 'FAIL'} {mode:18s} entered={entered} worst tilt {worst:5.1f} deg")
        if not ok:
            failed.append(mode)
    sys.exit(1 if failed else 0)
