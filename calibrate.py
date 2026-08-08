#!/usr/bin/env python3
"""Calibration gate for the causal experiment (villa#1372).

Before any causal number is worth reporting, the pipeline has to be shown to
reproduce a *published* prediction from the *same* checkpoint on the *same*
derivation. That is the only end-to-end check that covers the render, the
z-window convention, the preprocessing and the blending all at once.

Target: `scrollprize/timesformer_scroll5_july_retreat` produced the published
map `...volume-20241024131838-20250713185324-timesformer_scroll5_july_retreat
-tile64-stride16.tif`. We run the same checkpoint on our render of 131838 and
compare.

The z-window is swept rather than assumed. The checkpoint's rotary position
embeddings do not constrain the frame count, and `optimized_inference` takes
START_LAYER/END_LAYER from the environment with no default recorded anywhere,
so the convention has to be recovered from the data.

Pass mark: IoU@5% at or above ~0.285, which is the agreement between the two
*published* maps (july vs november) on this same volume. Chance is 0.026.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Volumes/AppsAndFiles/dev/op6-causal")
sys.path.insert(0, str(ROOT))

from infer import InkEngine  # noqa: E402

R = ROOT / "renders"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
DATA = ROOT.parent / "vesuvius-op6" / "data"

# ROI position in the full segment canvas (from the renderer's README).
CROP_X, CROP_Y, CROP_W, CROP_H = 7168, 6400, 512, 512


# --------------------------------------------------------------------------
# Metric: top-q-fraction IoU, matching the coordinator's chance values
#   chance(q) = q / (2 - q)   ->  0.005 @1%, 0.026 @5%, 0.111 @20%
# --------------------------------------------------------------------------
def topk_mask(values: np.ndarray, q: float) -> np.ndarray:
    """The top q fraction by value. Partitions on the values, never on their negation.

    `argpartition(-flat, ...)` looks equivalent and is not: on an unsigned dtype the unary
    minus wraps, so 0 negates to 0 and stays the smallest, while every non-zero value
    negates to something large. On uint8 maps that ranks the zeros first and inverts the
    selection. Predictions arrive here as float32 so the published numbers never hit it, but
    a helper that silently inverts on a whole dtype family is a trap for the next caller.
    """
    flat = values.reshape(-1)
    k = max(1, int(round(q * flat.size)))
    k = min(k, flat.size)
    idx = np.argpartition(flat, flat.size - k)[flat.size - k:]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(values.shape)


def iou_at_q(a: np.ndarray, b: np.ndarray, q: float) -> float:
    ma, mb = topk_mask(a, q), topk_mask(b, q)
    union = np.count_nonzero(ma | mb)
    return float(np.count_nonzero(ma & mb) / union) if union else 0.0


def chance_iou(q: float) -> float:
    return q / (2.0 - q)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    ra, rb = rankdata(a.reshape(-1)), rankdata(b.reshape(-1))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


# --------------------------------------------------------------------------
def load_published_ink(volume: str) -> np.ndarray:
    import tifffile

    matches = [p for p in DATA.glob("*.tif") if volume in p.name and "july" in p.name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one july map for {volume}, got {matches}")
    full = tifffile.imread(str(matches[0]))
    roi = full[CROP_Y : CROP_Y + CROP_H, CROP_X : CROP_X + CROP_W]
    return roi.astype(np.float32)


def window(dhw: np.ndarray, start: int, count: int, reverse: bool) -> np.ndarray:
    """(D, H, W) -> (H, W, count), optionally reversed along depth."""
    w = dhw[start : start + count]
    if reverse:
        w = w[::-1]
    return np.ascontiguousarray(w.transpose(1, 2, 0))


def main() -> None:
    render = np.load(R / "render_131838.npy")  # (33, 512, 512)
    print(f"render 131838: {render.shape} {render.dtype}")
    pub = load_published_ink("131838")
    nz = np.count_nonzero(pub) / pub.size
    print(f"published ink ROI: {pub.shape}, non-zero {nz:.3f}, "
          f"min {pub.min():.0f} max {pub.max():.0f} mean {pub.mean():.1f}")
    print(f"chance IoU: @1% {chance_iou(.01):.3f}  @5% {chance_iou(.05):.3f}  "
          f"@20% {chance_iou(.20):.3f}")
    print(f"pass mark @5%: 0.285 (agreement between the two published maps)\n")

    depth = render.shape[0]
    # Every start for every count, not only the centred one.
    #
    # The first version of this swept `start = (depth - count) // 2` alone, which is a line
    # through a two-dimensional space, and the reference window was not on it: for count 26
    # it only ever tried [3:29). The window the published pipeline actually uses is [1:27),
    # which scores 0.449 against the published map where the centred [3:29) scores 0.395 and
    # the [1:31) this file used to choose scores 0.420. An adversarial review found it by
    # reading the source instead of trusting the sweep.
    #
    # `inference_timesformer.py` defaults to start_idx=17 with CFG.in_chans=26 and builds
    # `range(start_idx, start_idx + in_chans)`, so the reference stack is legacy layers 17
    # through 42. A 33-slice render centred on the surface spans offsets -16..+16, i.e. legacy
    # 16..48, so 17..42 lands on local indices 1..26. That is [1:27), and it is now a named
    # candidate rather than something the sweep has to stumble on.
    candidates = [
        (start, count)
        for count in (33, 30, 26)
        for start in range(0, depth - count + 1)
    ]
    candidates = sorted(set(candidates))

    print("=== sweep: z-window x reverse, scored against the published map ===")
    header = f"  {'window':<12} {'frames':>6} {'rev':<6} {'IoU@1%':>8} {'IoU@5%':>8} {'IoU@20%':>8} {'rho':>7}  {'mean':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    for (start, count), rev in itertools.product(candidates, (False, True)):
        engine = InkEngine(
            model_kind="timesformer",
            num_frames=count,
            stride=16,
            batch_size=1,
        )
        ink = engine.predict(window(render, start, count, rev))
        row = {
            "start": start,
            "count": count,
            "reverse": rev,
            "iou_1": iou_at_q(ink, pub, 0.01),
            "iou_5": iou_at_q(ink, pub, 0.05),
            "iou_20": iou_at_q(ink, pub, 0.20),
            "rho": spearman(ink, pub),
            "mean": float(ink.mean()),
        }
        results.append(row)
        np.save(OUT / f"cal_131838_s{start}_n{count}_r{int(rev)}.npy", ink)
        print(f"  [{start}:{start+count})".ljust(14)
              + f"{count:>6} {str(rev):<6} {row['iou_1']:>8.3f} {row['iou_5']:>8.3f} "
                f"{row['iou_20']:>8.3f} {row['rho']:>7.3f}  {row['mean']:>6.3f}")

    best = max(results, key=lambda r: r["iou_5"])
    print(f"\n  best: [{best['start']}:{best['start']+best['count']}) "
          f"reverse={best['reverse']}  IoU@5%={best['iou_5']:.3f}")

    passed = best["iou_5"] >= 0.285
    margin_over_chance = best["iou_5"] / chance_iou(0.05)
    print(f"  vs chance (0.026): {margin_over_chance:.1f}x")
    print(f"\n=== CALIBRATION {'PASSED' if passed else 'FAILED'} ===")
    if not passed:
        print("  The pipeline does not reproduce the published map from the same")
        print("  checkpoint on the same derivation. Causal numbers from this")
        print("  configuration are not trustworthy. Do not report them.")

    (OUT / "calibration.json").write_text(
        json.dumps(
            {
                "pass_mark_iou5": 0.285,
                "chance_iou5": chance_iou(0.05),
                "passed": bool(passed),
                "best": best,
                "sweep": results,
            },
            indent=2,
        )
    )
    print(f"\n  written: {OUT / 'calibration.json'}")

    # Exit non-zero on failure, so that "do not report them" is enforced by the
    # process rather than left to whoever is reading the output. Printing a
    # warning and exiting 0 is not a gate; an adversarial review pointed out that
    # this was described as one before it behaved like one.
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
