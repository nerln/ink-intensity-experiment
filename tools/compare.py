#!/usr/bin/env python3
"""Compare a render against the published surface volume, voxel by voxel.

This is the script behind the renderer validation quoted in docs/rendering.md and in
ScrollPrize/villa#1372: 8,629,175 of 8,650,752 voxels identical, maximum difference one unit
in 255, none differing by two. It also decides the slice order, by printing the agreement
both ways round: direct z order matched 4.27% of voxels and reversed matched 99.75%, which is
how `--flip-normals` was established.

It needs `renders/`, which is gitignored because the arrays are large. Regenerate them with
the command lines in docs/rendering.md, then:

    python tools/compare.py [render.zarr] [published_roi.npy]

Everything used to run at import time, so `import tools.compare` on a clean checkout raised
FileNotFoundError before doing anything. It is now behind a main guard and takes its paths as
arguments.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

DEFAULT_RENDER = "renders/render_131838.zarr"
DEFAULT_PUBLISHED = "renders/published_131838_roi.npy"
SHAPE = (33, 512, 512)
CHUNKS = (33, 128, 128)


def load_zarr_u8(root: str, shape: tuple[int, int, int], chunks: tuple[int, int, int]):
    """Read an uncompressed uint8 zarr v2 array written with `/` as the dimension separator."""
    root = pathlib.Path(root)
    Z, Y, X = shape
    cz, cy, cx = chunks
    out = np.zeros(shape, np.uint8)
    for iy in range((Y + cy - 1) // cy):
        for ix in range((X + cx - 1) // cx):
            p = root / "0" / "0" / f"{iy}" / f"{ix}"
            if not p.exists():
                continue
            c = np.frombuffer(p.read_bytes(), np.uint8).reshape(chunks)
            out[:, iy * cy : min(Y, (iy + 1) * cy), ix * cx : min(X, (ix + 1) * cx)] = c[
                :, : min(cy, Y - iy * cy), : min(cx, X - ix * cx)
            ]
    return out


def main(argv: list[str]) -> int:
    render = argv[1] if len(argv) > 1 else DEFAULT_RENDER
    published = argv[2] if len(argv) > 2 else DEFAULT_PUBLISHED
    for p in (render, published):
        if not pathlib.Path(p).exists():
            print(f"missing: {p}\nSee docs/rendering.md for how to produce it.", file=sys.stderr)
            return 2

    r = load_zarr_u8(render, SHAPE, CHUNKS)
    ref = np.load(published)
    print(f"mine min/max/mean {r.min()} {r.max()} {r.mean():.3f}  zeros={(r == 0).sum()}")
    print(f"ref  min/max/mean {ref.min()} {ref.max()} {ref.mean():.3f}  zeros={(ref == 0).sum()}")

    # The committed renders were produced with --flip-normals, so "as rendered" is the one
    # that should match. Both are printed because comparing them is how the flag was determined
    # in the first place: without it the two rows swap.
    for name, a in (("as rendered", r), ("z reversed", r[::-1])):
        d = a.astype(np.int16) - ref.astype(np.int16)
        eq = int((d == 0).sum())
        print(f"{name:<18} equal={eq}/{d.size} ({100 * eq / d.size:.4f}%)  "
              f"maxabs={int(np.abs(d).max())}  mean={float(d.mean()):.5f}  "
              f"meanabs={float(np.abs(d).mean()):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
