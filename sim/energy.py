"""Per-second motion energy of recorded takes: mean joint speed and base bob."""
import sys
import numpy as np
for path in sys.argv[1:]:
    z = np.load(path); q = z["qpos"]; fps = int(z["fps"])
    js = np.abs(np.diff(q[:, 7:29], axis=0)).mean(axis=1) * fps          # rad/s, mean over joints
    n = len(js) // fps
    per_s = js[: n * fps].reshape(n, fps).mean(axis=1)
    bob = (q[: n * fps, 2].reshape(n, fps).max(axis=1) - q[: n * fps, 2].reshape(n, fps).min(axis=1)) * 1000
    print(path.split("/")[-1], "| tilt max %.1f" % z["tilt"].max())
    print("  speed:", " ".join(f"{v:3.1f}" for v in per_s))
    print("  bob mm:", " ".join(f"{v:3.0f}" for v in bob))
