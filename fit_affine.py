#!/usr/bin/env python3
"""Where the affine coefficients come from.

`experiment_timesformer.py` applies `A = m*B + c` with m, c written as constants.
Constants in a file are not evidence, and a reviewer cannot tell from the artifact
alone whether they were fitted on the data the experiment then evaluates. They were:
the README says "measured voxel-by-voxel on the renders". This script makes that
procedure explicit and testable, and reports two things the single fit cannot show.

  1. A held-out fit. The rendered window is split in half, the relation is fitted on
     one half and scored on the other. If the relation is a property of the encoding
     it transfers; if it is an artefact of fitting, it does not.

  2. The agreement with an estimate taken entirely off these renders. The issue's
     aggregate `A = 0.6154*B + 104.32` was fitted on five volume chunks sampled
     across the scan, disjoint from the rendered ROI. Coefficients agreeing to the
     third decimal from disjoint data are a measurement, not a tuned parameter.

`sweep_affine.py` then runs both through the checkpoint, which is the part that
answers whether arm 2 restates the fit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from calibrate import window  # noqa: E402

R = ROOT / "renders"
OUT = ROOT / "out"

WIN_START, WIN_COUNT, REVERSE = 1, 30, False

# Fitted off these renders, on five volume chunks sampled across the scan (10.42 Mvoxel).
VOLUME_AGGREGATE = (0.6154, 104.32)


def fit(b: np.ndarray, a: np.ndarray) -> tuple[float, float, float]:
    """Least squares a = m*b + c, plus Pearson r, on paired voxels."""
    x = b.astype(np.float64).ravel()
    y = a.astype(np.float64).ravel()
    m, c = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(m), float(c), r


def score(b: np.ndarray, a: np.ndarray, m: float, c: float) -> dict:
    """How well m, c carry a to b on this data, in the units that reach the model."""
    pred = np.clip(np.rint(b.astype(np.float64) * m + c), 0, 255).astype(np.uint8)
    err = pred.astype(np.int16) - a.astype(np.int16)
    return {
        "within_1_level": float((np.abs(err) <= 1).mean()),
        "within_2_levels": float((np.abs(err) <= 2).mean()),
        "mae_levels": float(np.abs(err).mean()),
    }


def main() -> None:
    A = window(np.load(R / "render_131838.npy"), WIN_START, WIN_COUNT, REVERSE)
    B = window(np.load(R / "render_131839.npy"), WIN_START, WIN_COUNT, REVERSE)
    print(f"rendered window {A.shape}, {A.size} paired voxels\n")

    report: dict = {}

    m_all, c_all, r_all = fit(B, A)
    print(f"full-window fit        A = {m_all:.4f}*B + {c_all:.2f}   r = {r_all:.5f}")
    report["full"] = {"slope": m_all, "intercept": c_all, "r": r_all,
                      **score(B, A, m_all, c_all)}

    # --- held out: fit on the left half of the window, score on the right ----
    w = A.shape[-1]
    left = slice(None, w // 2)
    right = slice(w // 2, None)
    m_l, c_l, r_l = fit(B[..., left], A[..., left])
    held = score(B[..., right], A[..., right], m_l, c_l)
    print(f"fitted on left half    A = {m_l:.4f}*B + {c_l:.2f}   r = {r_l:.5f}")
    print(f"  scored on right half : {100*held['within_1_level']:.2f}% of voxels within one 8-bit level"
          f"   MAE {held['mae_levels']:.3f} levels")
    report["held_out_spatial"] = {"slope": m_l, "intercept": c_l, "r": r_l, **held}

    # --- the estimate that never saw these renders --------------------------
    m_v, c_v = VOLUME_AGGREGATE
    vol = score(B, A, m_v, c_v)
    print(f"\nvolume-chunk estimate  A = {m_v:.4f}*B + {c_v:.2f}   (fitted off these renders)")
    print(f"  scored on the whole rendered window: {100*vol['within_1_level']:.2f}% within one level"
          f"   MAE {vol['mae_levels']:.3f} levels")
    print(f"  distance from the in-sample fit: slope {abs(m_v-m_all):.4f}, intercept {abs(c_v-c_all):.2f}")
    report["volume_estimate"] = {"slope": m_v, "intercept": c_v, **vol,
                                 "slope_gap": abs(m_v - m_all),
                                 "intercept_gap": abs(c_v - c_all)}

    OUT.mkdir(exist_ok=True)
    (OUT / "fit_affine.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {OUT / 'fit_affine.json'}")


if __name__ == "__main__":
    main()
