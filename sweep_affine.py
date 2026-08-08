#!/usr/bin/env python3
"""Is arm 2 a discovery or a tautology?

The objection: the affine remap `A = m*B + c` used in `experiment_timesformer.py`
was fitted on the same renders the experiment then evaluates. If the low delta in
arm 2 only appears at coefficients tuned on that data, the result restates the fit
instead of testing anything.

This sweeps the coefficients instead of trusting one pair, and asks three questions
the single run cannot answer:

  1. Does the collapse survive coefficients measured on data disjoint from the
     evaluated ROI? The issue's aggregate fit `A = 0.6154*B + 104.32` came from
     five volume chunks sampled across the scan, not from the rendered window.
  2. Is the minimum a knife edge or a basin? A fitted degree of freedom gives a
     sharp optimum; a physical property of the encoding gives a wide flat floor.
  3. Which half of the affine does the work? Slope-only and intercept-only are
     run as controls, together with identity (no remap at all).

Only `remap(B)` depends on the coefficients, so `inkA` and `inkB` are reused from
the committed run rather than recomputed. That keeps the comparison exact: every
delta below is measured against the identical `inkA`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from calibrate import iou_at_q, spearman, window  # noqa: E402
from infer import InkEngine  # noqa: E402

R = ROOT / "renders"
OUT = ROOT / "out"

WIN_START, WIN_COUNT, REVERSE = 1, 30, False
STRIDE = 16
BLEND = "hann"

# (label, slope, intercept, provenance)
CANDIDATES = [
    ("identity (no remap)",       1.0000,   0.00, "control: arm 1 restated"),
    ("intercept only",            1.0000, 104.02, "control: shift without scale"),
    ("slope only",                0.6165,   0.00, "control: scale without shift"),
    ("render fit (original)",     0.6165, 104.02, "in-sample: fitted on these renders"),
    ("volume aggregate",          0.6154, 104.32, "OUT-OF-SAMPLE: 5 chunks, 10.42 Mvoxel"),
    ("volume chunk 60/26/35",     0.6154, 104.32, "out-of-sample: single chunk"),
    ("volume chunk 90/26/34",     0.6151, 104.34, "out-of-sample: single chunk"),
    ("volume chunk 100/27/36",    0.6155, 104.31, "out-of-sample: single chunk"),
    # Deliberately detuned, to map the basin around the optimum.
    ("slope -5%",                 0.5856, 104.32, "detuned"),
    ("slope +5%",                 0.6462, 104.32, "detuned"),
    ("intercept -10",             0.6154,  94.32, "detuned"),
    ("intercept +10",             0.6154, 114.32, "detuned"),
    ("slope -20%",                0.4923, 104.32, "detuned hard"),
    ("intercept -40",             0.6154,  64.32, "detuned hard"),
]


def remap(b: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    x = b.astype(np.float64) * slope + intercept
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def delta_at(a: np.ndarray, b: np.ndarray, q: float) -> float:
    return 1.0 - iou_at_q(a, b, q)


def main() -> None:
    inkA = np.load(OUT / f"ts_{BLEND}_A.npy")
    inkB = np.load(OUT / f"ts_{BLEND}_B.npy")
    B33 = np.load(R / "render_131839.npy")
    B = window(B33, WIN_START, WIN_COUNT, REVERSE)

    d_arm1 = delta_at(inkA, inkB, 0.05)
    print(f"reference points, blend={BLEND}, stride={STRIDE}")
    print(f"  ARM 1  A vs B (no remap)      delta@5% = {d_arm1:.4f}")
    print(f"  the committed arm 2 reported  delta@5% = 0.054\n")

    engine = InkEngine(
        model_kind="timesformer",
        num_frames=WIN_COUNT,
        stride=STRIDE,
        batch_size=1,
        blend=BLEND,
    )

    print(f"  {'coefficients':<26} {'slope':>7} {'inter':>7} {'D@5%':>8} {'rho':>7}   provenance")
    print("  " + "-" * 92)
    rows = []
    for label, slope, intercept, prov in CANDIDATES:
        Bm = remap(B, slope, intercept)
        inkBm = engine.predict(Bm)
        d = delta_at(inkA, inkBm, 0.05)
        rho = spearman(inkA, inkBm)
        red = (d_arm1 - d) / d_arm1 if d_arm1 else float("nan")
        rows.append({
            "label": label, "slope": slope, "intercept": intercept,
            "provenance": prov, "delta5": d, "rho": rho, "reduction": red,
        })
        print(f"  {label:<26} {slope:7.4f} {intercept:7.2f} {d:8.4f} {rho:7.3f}   {prov}")

    OUT.mkdir(exist_ok=True)
    (OUT / "sweep_affine.json").write_text(json.dumps({
        "blend": BLEND, "stride": STRIDE,
        "window": [WIN_START, WIN_START + WIN_COUNT], "reverse": REVERSE,
        "d_arm1": d_arm1, "rows": rows,
    }, indent=2))
    print(f"\nwritten: {OUT / 'sweep_affine.json'}")


if __name__ == "__main__":
    main()
