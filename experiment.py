#!/usr/bin/env python3
"""The causal experiment for ScrollPrize/villa#1372.

Question: is the difference between the ink maps produced on two derivations
of the same scan caused by the affine remap of the 8 bits?

Three arms:
  1. SAME checkpoint on the two renders -> observed Delta
  2. SAME checkpoint on B remapped onto A -> if Delta collapses, it is the intensity
  3. controls: the engine against itself (must give 0), and the floor imposed by the
     voxels B lost, which no linear remap recovers

Conventions: the 63-slice render is a superset of the two candidate windows,
so [0:62] = -31..+30 and [1:63] = -30..+31. reverse is [::-1].
The four combinations are chosen by comparing against the PUBLISHED map for
131838, knowing it comes from a different model family: we are looking for which
convention beats the others, not for high agreement in absolute terms.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "inkfloor"))

from infer import InkEngine  # noqa: E402
from inkfloor import metrics  # noqa: E402

R = ROOT / "renders"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

# The ROI in the full canvas, from the renderer's README.
CROP_X, CROP_Y, CROP_W, CROP_H = 7168, 6400, 512, 512

# Relation measured on the renders, voxel by voxel: A = slope*B + intercept
SLOPE, INTERCEPT = 0.6165, 104.02


def window(d63: np.ndarray, start: int, reverse: bool) -> np.ndarray:
    """Take 62 slices out of the 63-slice render, as (H, W, D)."""
    w = d63[:, :, start : start + 62]
    return w[:, :, ::-1].copy() if reverse else w.copy()


def remap_b_to_a(b: np.ndarray) -> np.ndarray:
    """Apply the measured affine relation, saturating to uint8 as the data is."""
    x = b.astype(np.float64) * SLOPE + INTERCEPT
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def delta(a: np.ndarray, b: np.ndarray, valid: np.ndarray, q: float) -> float:
    return 1.0 - metrics.delta_at_q(a, b, valid, q).iou


def main() -> None:
    A63 = np.load(R / "render_131838_hwd63.npy")
    B63 = np.load(R / "render_131839_hwd63.npy")
    print(f"render A {A63.shape} {A63.dtype}   render B {B63.shape} {B63.dtype}")

    pub_full = np.load(R / "published_131838_roi.npy")  # (33,512,512), already the ROI
    print(f"published (33 slices) {pub_full.shape}")

    engine = InkEngine()
    print(f"engine ready\n")

    # --- choosing the convention, against the PUBLISHED ink map ---
    # The published prediction for 131838 is in the whole .tif: crop the ROI out of it.
    import tifffile

    pred_key = next(
        p for p in (ROOT.parent / "vesuvius-op6" / "data").glob("*.tif")
        if "131838" in p.name and "july" in p.name
    )
    pred_full = tifffile.imread(pred_key)
    pub_ink = pred_full[CROP_Y : CROP_Y + CROP_H, CROP_X : CROP_X + CROP_W].astype(np.float32)
    print(f"published ink map on the ROI: {pub_ink.shape}, "
          f"non-zero {np.count_nonzero(pub_ink)/pub_ink.size:.3f}\n")

    print("=== choosing the convention (on A, against the published map) ===")
    print(f"  {'window':<12} {'reverse':<8} {'rho vs published':>20}")
    best, best_rho = None, -2.0
    conv_rows = []
    for start, rev in itertools.product((0, 1), (False, True)):
        ink = engine.predict(window(A63, start, rev))
        v = np.ones(ink.shape, bool)
        rho = metrics.spearman(ink.astype(np.float32), pub_ink, v)
        tag = f"[{start}:{start+62}]"
        print(f"  {tag:<12} {str(rev):<8} {rho:>20.4f}")
        conv_rows.append({"start": start, "reverse": rev, "rho": float(rho)})
        if rho > best_rho:
            best, best_rho = (start, rev), rho
    start, rev = best
    spread = best_rho - min(r["rho"] for r in conv_rows)
    print(f"\n  chosen: window [{start}:{start+62}], reverse={rev}, rho={best_rho:.4f}")
    print(f"  gap from the worst: {spread:.4f}"
          f"  -> {'the data decide the convention' if spread > 0.05 else 'NOT decisive, see the note'}\n")

    # --- the three arms ---
    A = window(A63, start, rev)
    B = window(B63, start, rev)
    Bmap = remap_b_to_a(B)

    print("=== inference ===")
    inkA = engine.predict(A)
    inkB = engine.predict(B)
    inkBm = engine.predict(Bmap)
    inkA2 = engine.predict(A)  # determinism control

    valid = np.ones(inkA.shape, bool)
    qs = (0.01, 0.05, 0.20)

    print(f"\n  {'comparison':<44} " + " ".join(f"{'D@'+str(int(q*100))+'%':>8}" for q in qs))
    rows = []
    for label, x, y in [
        ("control: A vs A (must be 0)", inkA, inkA2),
        ("ARM 1: A vs B (observed)", inkA, inkB),
        ("ARM 2: A vs remap(B)", inkA, inkBm),
        ("reference: B vs remap(B)", inkB, inkBm),
    ]:
        ds = [delta(x, y, valid, q) for q in qs]
        rows.append({"label": label, "deltas": ds,
                     "spearman": float(metrics.spearman(x, y, valid))})
        print(f"  {label:<44} " + " ".join(f"{d:8.3f}" for d in ds))

    d_obs = rows[1]["deltas"][1]
    d_rem = rows[2]["deltas"][1]
    print(f"\n  rho: " + ", ".join(f"{r['label'].split(':')[0]}={r['spearman']:.3f}" for r in rows))

    print(f"\n=== VERDICT (at q=5%) ===")
    print(f"  Delta observed between the two derivations : {d_obs:.3f}")
    print(f"  Delta after the affine remap               : {d_rem:.3f}")
    if d_obs > 0:
        print(f"  reduction                                  : {(d_obs-d_rem)/d_obs:+.1%}")

    # the floor imposed by the lost information
    lost = (B == 0) & (A > 0)
    print(f"\n  voxels lost by B (zero where A has a value): {lost.sum():,} "
          f"({lost.mean():.4%}) -> a floor no linear remap can recover, by construction")

    json.dump({"convention": {"start": start, "reverse": rev, "rho": best_rho,
                              "spread": float(spread), "all": conv_rows},
               "rows": rows, "qs": list(qs),
               "lost_voxels": int(lost.sum()), "lost_frac": float(lost.mean()),
               "remap": {"slope": SLOPE, "intercept": INTERCEPT}},
              open(OUT / "experiment.json", "w"), indent=1)
    np.savez_compressed(OUT / "ink_maps.npz", A=inkA, B=inkB, Bmap=inkBm)
    print(f"\n  saved in {OUT}")


if __name__ == "__main__":
    main()
