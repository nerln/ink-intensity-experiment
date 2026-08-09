"""2D mask (671x747) for the experiment's segment, at the mesh's native resolution.

Fields saved in segment_mask.npz:
  valid        bool   tifxyz point with valid coordinates (x,y,z >= 0)
  ok_both      bool   the whole render column (33 samples, +-16 voxels along the normal)
                      falls in chunks present in BOTH derivations
  hit_missing  bool   the column touches at least one chunk present in 131838, absent in 131839
  margin_chunk int16  Chebyshev distance, in 128-voxel chunks, from the point's central chunk
                      to the nearest chunk present in A and absent in B (-1 if not valid)
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_HERE = str(Path(__file__).resolve().parent)
_REPO = str(Path(__file__).resolve().parent.parent)
_INK = str(Path(__file__).resolve().parent.parent.parent / "inkfloor")

sys.path.insert(0, f"{_REPO}/coverage")
from mesh_chunks import CH, NX, NY, NZ, chunk_lin, load_mesh, normals  # noqa: E402



SEG = (f"{_INK}/cache/PHerc0172/segments/"
       "20251107110950-w064_20251107110950052_flatboi/mesh/"
       "20251107110950-on-20241024131838-7.91um.tifxyz")
HALF = 16


def main():
    cov = np.load(f"{_HERE}/coverage.npz")
    ga3, gb3 = cov["grid_a"], cov["grid_b"]
    ga, gb = ga3.reshape(-1), gb3.reshape(-1)
    onlyA = np.array(np.nonzero(ga3 & ~gb3)).T.astype(np.int32)

    x, y, z, valid = load_mesh(SEG)
    nrm, _ = normals(x, y, z, valid)
    H, W = x.shape
    ts = np.arange(-HALF, HALF + 1, dtype=np.float64)

    sx = x[:, :, None] + nrm[:, :, 0:1] * ts
    sy = y[:, :, None] + nrm[:, :, 1:2] * ts
    sz = z[:, :, None] + nrm[:, :, 2:3] * ts
    lin = chunk_lin(sx, sy, sz)
    good = (lin >= 0) & valid[:, :, None]
    li = np.where(good, lin, 0)
    pres_a = ga[li] & good
    hit = (pres_a & ~gb[li]).any(axis=2)
    ok = valid & pres_a.any(axis=2) & ~hit

    # margin: distance in chunks from the central chunk to the nearest chunk missing in B
    cz = np.floor(z / CH).astype(np.int32).clip(0, NZ - 1)
    cy = np.floor(y / CH).astype(np.int32).clip(0, NY - 1)
    cx = np.floor(x / CH).astype(np.int32).clip(0, NX - 1)
    cen = np.stack([cz, cy, cx], -1).reshape(-1, 3)
    best = np.full(cen.shape[0], 32767, dtype=np.int32)
    for i in range(0, onlyA.shape[0], 1500):
        blk = onlyA[i:i + 1500]
        d = np.abs(cen[:, None, :] - blk[None, :, :]).max(axis=2).min(axis=1)
        np.minimum(best, d, out=best)
    margin = best.reshape(H, W).astype(np.int16)
    margin[~valid] = -1

    np.savez_compressed(
        f"{_HERE}/segment_mask.npz",
        valid=valid, ok_both=ok, hit_missing=hit, margin_chunk=margin,
        segment="20251107110950-w064_20251107110950052_flatboi",
        half_voxels=HALF, chunk=CH,
    )
    m = margin[valid]
    print(f"shape {H}x{W}  valid={int(valid.sum())}  ok_both={int(ok.sum())}  hit_missing={int(hit.sum())}")
    print(f"margin (chunks) over the valid points: min={m.min()} p1={np.percentile(m,1):.0f} "
          f"median={np.median(m):.0f} max={m.max()}")
    print("points with margin <=1 chunk:", int((m <= 1).sum()),
          " <=2:", int((m <= 2).sum()), " <=3:", int((m <= 3).sum()))


if __name__ == "__main__":
    main()
