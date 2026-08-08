from __future__ import annotations
from pathlib import Path
"""Maschera 2D (671x747) per il segmento dell'esperimento, alla risoluzione nativa della mesh.

Campi salvati in segment_mask.npz:
  valid        bool   punto della tifxyz con coordinate valide (x,y,z >= 0)
  ok_both      bool   tutta la colonna di render (33 campioni, +-16 voxel lungo la normale)
                      cade in chunk presenti in ENTRAMBE le derivazioni
  hit_missing  bool   la colonna tocca almeno un chunk presente in 131838 e assente in 131839
  margin_chunk int16  distanza di Chebyshev, in chunk da 128 voxel, dal chunk centrale del
                      punto al piu' vicino chunk presente in A e assente in B (-1 se non valido)
"""

import sys

import numpy as np

sys.path.insert(0, f"{_REPO}/coverage")
from mesh_chunks import CH, NX, NY, NZ, chunk_lin, load_mesh, normals  # noqa: E402

_HERE = str(Path(__file__).resolve().parent)
_REPO = str(Path(__file__).resolve().parent.parent)
_INK = str(Path(__file__).resolve().parent.parent.parent / "inkfloor")


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

    # margine: distanza in chunk dal chunk centrale al piu' vicino chunk mancante in B
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
    print(f"margine (chunk) sui punti validi: min={m.min()} p1={np.percentile(m,1):.0f} "
          f"mediana={np.median(m):.0f} max={m.max()}")
    print("punti con margine <=1 chunk:", int((m <= 1).sum()),
          " <=2:", int((m <= 2).sum()), " <=3:", int((m <= 3).sum()))


if __name__ == "__main__":
    main()
