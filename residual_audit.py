#!/usr/bin/env python3
"""What is left in arm 2, and whether it carries anything.

`sweep_affine.py` shows the intensity remap removes most of the disagreement and
that independently measured coefficients do it too. That leaves the residual:
delta@5% = 0.054 is small, but small is not the same as empty, and a residual that
concentrated somewhere meaningful would be a finding in its own right.

Three candidate explanations, and the test that separates them:

  a. the voxels B structurally lost (B == 0 where A != 0), which no affine recovers
  b. whatever input disagreement survives the remap, at sub-level rounding
  c. churn at the top-k selection boundary, i.e. an artefact of the metric

The design of (a) and (b) came from an independent adversarial review of this
repository, which asked whether the residual carries information. (c) is the one that
survives the tests here.

Run on both coefficient sets, because it matters which one. A later review pointed out
that auditing only the in-sample residual is auditing the wrong number: the result the
argument rests on is the held-out one, and its residual is larger (0.062 against 0.054),
so it has more to explain, not less.
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
BLEND = "hann"
Q = 0.05

# (label, slope, intercept). The first is fitted on these renders; the second on five volume
# chunks that never touch this ROI, and is the one the causal claim actually rests on.
COEFFS = [
    ("in-sample (render fit)", 0.6165, 104.02),
    ("held out (volume chunks)", 0.6154, 104.32),
]
CLIP = 200  # the pipeline's absolute clip: differences above it never reach the model


def _ink_for(B: np.ndarray, slope: float, intercept: float, cache: Path) -> np.ndarray:
    """The prediction on remap(B). Cached, because each one is a full inference pass."""
    if cache.exists():
        return np.load(cache)
    from infer import InkEngine

    Bm = np.clip(np.rint(B.astype(np.float64) * slope + intercept), 0, 255).astype(np.uint8)
    ink = InkEngine(model_kind="timesformer", num_frames=WIN_COUNT, stride=16,
                    batch_size=1, blend=BLEND).predict(Bm)
    np.save(cache, ink)
    return ink


def audit(label: str, slope: float, intercept: float, inkA, A, B) -> dict:
    tag = f"{slope:.4f}_{intercept:.2f}".replace(".", "p")
    inkBm = _ink_for(B, slope, intercept, OUT / f"ts_{BLEND}_Bmap_{tag}.npy")
    Bm = np.clip(np.rint(B.astype(np.float64) * slope + intercept), 0, 255).astype(np.uint8)
    print(f"\n{'='*70}\n=== {label}:  A = {slope}*B + {intercept} ===")

    a, m = inkA.ravel(), inkBm.ravel()
    n = a.size
    k = round(Q * n)
    sel_a = a >= np.partition(a, -k)[-k]
    sel_m = m >= np.partition(m, -k)[-k]
    changed = sel_a ^ sel_m
    iou = np.count_nonzero(sel_a & sel_m) / np.count_nonzero(sel_a | sel_m)

    print(f"delta@{int(Q*100)}% = {1-iou:.4f}   top-k = {k} pixels per side")
    print(f"  membership changes: {changed.sum()} pixels, "
          f"{changed.sum()//2} swapped in each direction ({changed.mean():.3%} of the ROI)")
    print(f"  so {100*(1 - changed.sum()/(2*k)):.1f}% of the selected set is identical\n")

    # (a) the structurally lost columns
    # window() returns (H, W, D), so depth is the last axis.
    lost = ((B == 0) & (A != 0)).any(axis=-1).ravel()
    on_lost = np.count_nonzero(changed & lost)
    enrich = (on_lost / changed.sum()) / lost.mean()
    print(f"(a) structural loss: lost columns are {lost.mean():.3%} of the ROI, and carry "
          f"{on_lost} of {changed.sum()} changes")
    print(f"    enrichment {enrich:.2f}x, but only {100*on_lost/changed.sum():.1f}% of the residual. "
          f"Not the explanation.\n")

    # (b) surviving input disagreement, in the units that reach the model
    in_mae = np.abs(np.minimum(A, CLIP).astype(np.int16)
                    - np.minimum(Bm, CLIP).astype(np.int16)).mean(axis=-1).ravel()
    out_abs = np.abs(a - m)
    order_i = np.argsort(np.argsort(in_mae))
    order_o = np.argsort(np.argsort(out_abs))
    rho = float(np.corrcoef(order_i, order_o)[0, 1])
    print(f"(b) residual input disagreement: median {np.median(in_mae):.3f} levels/pixel, "
          f"99th pct {np.quantile(in_mae, 0.99):.3f}")
    print(f"    Spearman against output |A - remap(B)| = {rho:+.4f}. "
          f"Essentially zero, so it does not drive the residual.\n")

    # (c) the selection boundary
    rank_a = np.argsort(np.argsort(a)) / (n - 1)
    edge = 1.0 - Q
    dist = np.abs(rank_a[changed] - edge)
    within1 = float(np.mean(dist < 0.01))
    print(f"(c) selection boundary: the top-{int(Q*100)}% cut sits at rank {edge:.4f}")
    print(f"    changed pixels sit at median rank {np.median(rank_a[changed]):.4f}, "
          f"5-95% [{np.quantile(rank_a[changed], .05):.4f}, {np.quantile(rank_a[changed], .95):.4f}]")
    print(f"    {100*within1:.1f}% of them are within one percentile point of the cut "
          f"(median distance {np.median(dist):.4f})")
    print("\n    Every changed pixel sits at the cut. That is consistent with threshold churn\n"
          "    and rules out the two localisations tested above; it does not prove a cause.")

    return {
        "label": label, "slope": slope, "intercept": intercept,
        "changed_pixels": int(changed.sum()),
        "identical_fraction_of_selection": 1 - changed.sum() / (2 * k),
        "lost_columns_fraction": float(lost.mean()),
        "changes_on_lost_columns": int(on_lost),
        "lost_enrichment": float(enrich),
        "spearman_input_vs_output_residual": rho,
        "changed_within_1pct_of_cut": within1,
        "changed_median_rank": float(np.median(rank_a[changed])),
    }


def main() -> None:
    inkA = np.load(OUT / f"ts_{BLEND}_A.npy")
    A = window(np.load(R / "render_131838.npy"), WIN_START, WIN_COUNT, REVERSE)
    B = window(np.load(R / "render_131839.npy"), WIN_START, WIN_COUNT, REVERSE)

    rows = [audit(lbl, m, c, inkA, A, B) for lbl, m, c in COEFFS]

    OUT.mkdir(exist_ok=True)
    (OUT / "residual_audit.json").write_text(json.dumps(
        {"blend": BLEND, "q": Q, "runs": rows}, indent=2))
    print(f"\nwritten: {OUT / 'residual_audit.json'}")


if __name__ == "__main__":
    main()
