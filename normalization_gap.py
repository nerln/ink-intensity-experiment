"""How much of the derivation difference survives normalisation, measured on real renders.

Two published derivations of `PHerc0172` differ by an 8-bit intensity window, chosen
deliberately and explained in villa#1211. The question this answers is narrow: after the
preprocessing step, how much of that difference is still in front of the network?

The old ink-detection path clipped to an absolute 200 and divided by a constant. An absolute
clip cannot absorb an affine change of intensity, so the difference passes straight through.
The current `vesuvius.ink_detection` default normalises each crop by its own median and MAD,
and an affine change is exactly what a median-and-scale normalisation removes. This script is
an input-level comparison only; it does not run or validate the current model end to end.

Run it on the two renders in `renders/` and it prints the gap under each. The figure in
`figures/normalization_gap.png` is the same measurement with pictures.

    python normalization_gap.py
    python normalization_gap.py --villa ../villa   # cross-check against villa's own function

`--villa` points at a villa checkout. When given, `normalize_robust` is imported from it and
the local implementation below is asserted to agree, so the number cannot drift away from what
the project actually runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

# Where the two renders live, and the affine relation between the derivations, fitted on volume
# chunks outside this ROI (r = 0.99987). See docs/rendering.md for how the renders were made.
RENDER_A = "renders/render_131838.npy"
RENDER_B = "renders/render_131839.npy"

CROP = 64  # the training crop; normalisation is per crop, so that is the unit that matters
STRIDE = 128


def robust_mad(image: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Clip to the percentile span, then centre on the median and scale by the MAD.

    This mirrors `vesuvius.image_proc.intensity.normalization.normalize_robust` at its
    defaults. `--villa` checks the mirror rather than trusting it.
    """
    arr = np.asarray(image).astype(np.float32, copy=True)
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, lower), np.percentile(arr, upper)
    np.clip(arr, lo, hi, out=arr)
    median = float(np.median(arr))
    scaled_mad = 1.4826 * float(np.median(np.abs(arr - median)))
    if not np.isfinite(scaled_mad) or scaled_mad < 1e-6:
        scaled_mad = max(abs(float(np.std(arr))), abs(hi - lo) / 2.0, 1.0)
    arr -= median
    arr /= scaled_mad
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_villa_normalize(checkout: Path):
    """Import `normalize_robust` from a villa checkout, so the comparison uses their code."""
    target = checkout / "vesuvius/src/vesuvius/image_proc/intensity/normalization.py"
    if not target.exists():
        raise SystemExit(f"no normalization.py under {checkout}; expected {target}")
    spec = importlib.util.spec_from_file_location("_villa_norm", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_robust


def crops(volume: np.ndarray, size: int = CROP, stride: int = STRIDE):
    _, height, width = volume.shape
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            yield volume[:, y : y + size, x : x + size]


def relative_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between the two normalised crops, in units of the signal."""
    return float(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--villa", metavar="DIR", type=Path,
                        help="villa checkout; cross-checks robust_mad against their function")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    try:
        volume_a = np.load(root / RENDER_A).astype(np.float32)
        volume_b = np.load(root / RENDER_B).astype(np.float32)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nThe renders are large and gitignored; see docs/rendering.md.")

    normalise = robust_mad
    if args.villa:
        villa_normalize = load_villa_normalize(args.villa)
        probe = volume_a[:, :CROP, :CROP]
        mine, theirs = robust_mad(probe), villa_normalize(probe.copy())
        drift = float(np.abs(mine - theirs).max())
        if drift > 1e-5:
            raise SystemExit(f"local robust_mad disagrees with villa's by {drift:.2e}")
        print(f"  robust_mad matches villa's normalize_robust to {drift:.1e}\n")
        normalise = villa_normalize

    paths = [
        ("legacy: clip 200, /200  (inference_timesformer)", lambda c: np.clip(c, 0, 200) / 200.0),
        ("legacy: clip 200, /255  (optimized_inference)", lambda c: np.clip(c, 0, 200) / 255.0),
        ("current: robust_mad p1-p99  (ink_detection default)", lambda c: normalise(c.copy())),
    ]

    pairs = list(zip(crops(volume_a), crops(volume_b)))
    print(f"  {'preprocessing':<50}{'median':>9}{'worst':>9}")
    for label, fn in paths:
        gaps = [relative_gap(fn(a), fn(b)) for a, b in pairs]
        print(f"  {label:<50}{np.median(gaps):>8.1%}{np.max(gaps):>9.1%}")
    print(f"\n  {len(pairs)} crops of {CROP}x{CROP}x{volume_a.shape[0]}, same voxels, "
          "two derivations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
