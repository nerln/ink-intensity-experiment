#!/usr/bin/env python3
"""The causal experiment for villa#1372, run with an IN-DISTRIBUTION checkpoint.

`scrollprize/timesformer_scroll5_july_retreat` is the model that produced one
of the two published predictions on this segment, so unlike
`PHerc.1667-iteration-0` it is not being applied out of distribution here.

Configuration is not assumed. It was recovered by `calibrate.py`, which swept
the z-window and the layer order against the published map from this same
checkpoint on this same derivation, and passed at IoU@5% = 0.420 against a
pass mark of 0.285 (chance 0.026).

Three arms:
  control    A vs A                  must be exactly 0
  arm 1      A vs B                  the observed difference
  arm 2      A vs remap(B)           if it collapses, intensity is the cause

Everything is reported under two blending conventions, because the published
pipeline used a third one (Gaussian numerator over a uniform count, which is
not a weighted average) and the conclusion should not depend on that choice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Volumes/AppsAndFiles/dev/op6-causal")
sys.path.insert(0, str(ROOT))

from calibrate import chance_iou, iou_at_q, load_published_ink, spearman, window  # noqa: E402
from infer import InkEngine  # noqa: E402

R = ROOT / "renders"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

# Recovered by calibrate.py, not assumed.
WIN_START, WIN_COUNT, REVERSE = 1, 30, False
STRIDE = 16  # matches the published run's "tile64-stride16"

# Intensity relation measured voxel-by-voxel on the two renders: A = m*B + c
SLOPE, INTERCEPT = 0.6165, 104.02


def remap_b_to_a(b: np.ndarray) -> np.ndarray:
    """Apply the measured affine relation, saturating to uint8 as the data is."""
    x = b.astype(np.float64) * SLOPE + INTERCEPT
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def delta_at(a: np.ndarray, b: np.ndarray, q: float, valid: np.ndarray | None = None) -> float:
    """1 - IoU@q. 0 means identical ranking of the top q fraction."""
    if valid is not None:
        a, b = a.copy(), b.copy()
        # Push excluded pixels below every retained one so they cannot enter top-k.
        a[~valid] = -1.0
        b[~valid] = -1.0
    return 1.0 - iou_at_q(a, b, q)


def main() -> None:
    A33 = np.load(R / "render_131838.npy")  # (33, 512, 512) uint8
    B33 = np.load(R / "render_131839.npy")
    print(f"render A {A33.shape}  render B {B33.shape}")

    # --- the structural floor: voxels B lost that no affine map recovers -----
    lost = (B33 == 0) & (A33 != 0)
    lost_cols = lost.any(axis=0)  # (H, W): columns touched by at least one lost voxel
    print(f"\nvoxels B lost where A has data: {lost.sum()} ({100*lost.mean():.4f}%)")
    print(f"  map pixels touched: {lost_cols.sum()} ({100*lost_cols.mean():.3f}%)")
    print(f"  remap sends 0 -> {int(round(INTERCEPT))}, i.e. mid-grey, not recovery")

    A = window(A33, WIN_START, WIN_COUNT, REVERSE)
    B = window(B33, WIN_START, WIN_COUNT, REVERSE)
    Bm = remap_b_to_a(B)
    print(f"\nwindow [{WIN_START}:{WIN_START+WIN_COUNT}) reverse={REVERSE} -> {A.shape}")
    print(f"  A mean {A.mean():.1f}   B mean {B.mean():.1f}   remap(B) mean {Bm.mean():.1f}")

    pub = load_published_ink("131838")
    qs = (0.01, 0.05, 0.20)
    report: dict = {
        "window": [WIN_START, WIN_START + WIN_COUNT],
        "reverse": REVERSE,
        "stride": STRIDE,
        "slope": SLOPE,
        "intercept": INTERCEPT,
        "lost_voxels": int(lost.sum()),
        "lost_pixels": int(lost_cols.sum()),
        "blends": {},
    }

    for blend in ("hann", "uniform"):
        print(f"\n{'='*66}\n=== blending: {blend} (stride {STRIDE}) ===")
        engine = InkEngine(
            model_kind="timesformer",
            num_frames=WIN_COUNT,
            stride=STRIDE,
            batch_size=1,
            blend=blend,
        )
        inkA = engine.predict(A)
        inkA2 = engine.predict(A)
        inkB = engine.predict(B)
        inkBm = engine.predict(Bm)

        cal = iou_at_q(inkA, pub, 0.05)
        print(f"  calibration re-check, A vs published: IoU@5% = {cal:.3f} "
              f"(pass mark 0.285, chance {chance_iou(0.05):.3f})")
        print(f"  means: A {inkA.mean():.3f}  B {inkB.mean():.3f}  remap(B) {inkBm.mean():.3f}")

        print(f"\n  {'comparison':<38} " + " ".join(f"{'D@'+str(int(q*100))+'%':>8}" for q in qs) + f" {'rho':>8}")
        rows = []
        for label, x, y in [
            ("control: A vs A (must be 0)", inkA, inkA2),
            ("ARM 1: A vs B", inkA, inkB),
            ("ARM 2: A vs remap(B)", inkA, inkBm),
            ("reference: B vs remap(B)", inkB, inkBm),
        ]:
            ds = [delta_at(x, y, q) for q in qs]
            rho = spearman(x, y)
            rows.append({"label": label, "deltas": ds, "rho": rho})
            print(f"  {label:<38} " + " ".join(f"{d:8.3f}" for d in ds) + f" {rho:8.3f}")

        d_obs, d_rem = rows[1]["deltas"][1], rows[2]["deltas"][1]

        # Same two arms, excluding the pixels B structurally lost.
        keep = ~lost_cols
        d_obs_k = delta_at(inkA, inkB, 0.05, keep)
        d_rem_k = delta_at(inkA, inkBm, 0.05, keep)

        print(f"\n  --- verdict at q=5% ---")
        print(f"  observed difference A vs B      : {d_obs:.3f}")
        print(f"  after the affine remap          : {d_rem:.3f}")
        red = (d_obs - d_rem) / d_obs if d_obs else float("nan")
        print(f"  reduction                       : {red:+.1%}")
        print(f"  excluding B's lost pixels       : {d_obs_k:.3f} -> {d_rem_k:.3f} "
              f"({(d_obs_k-d_rem_k)/d_obs_k:+.1%})" if d_obs_k else "")

        report["blends"][blend] = {
            "calibration_iou5": cal,
            "rows": rows,
            "d_observed": d_obs,
            "d_remapped": d_rem,
            "reduction": red,
            "d_observed_excl_lost": d_obs_k,
            "d_remapped_excl_lost": d_rem_k,
            "control_is_zero": rows[0]["deltas"] == [0.0, 0.0, 0.0],
        }
        for name, arr in (("A", inkA), ("B", inkB), ("Bmap", inkBm)):
            np.save(OUT / f"ts_{blend}_{name}.npy", arr)

    (OUT / "experiment_timesformer.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {OUT / 'experiment_timesformer.json'}")


if __name__ == "__main__":
    main()
