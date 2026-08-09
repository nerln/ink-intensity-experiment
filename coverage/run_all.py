"""Every published PHerc0172 segment against the coverage of the two derivations.

For each segment: how many chunks the thick surface touches (33 slices along the normal),
how many of those are missing in B, and how deep inside the shell stored by A the surface
pushes.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

_HERE = str(Path(__file__).resolve().parent)
_REPO = str(Path(__file__).resolve().parent.parent)
_INK = str(Path(__file__).resolve().parent.parent.parent / "inkfloor")

sys.path.insert(0, f"{_REPO}/coverage")
from mesh_chunks import NX, NY, NZ, chunk_lin, densify, load_mesh, normals  # noqa: E402



HALF = 16
DENS = 2


def analyse(d: str, ga, gb, depth, half=HALF, dens=DENS):
    x, y, z, valid = load_mesh(d)
    H0, W0 = x.shape
    xd, vd = densify(x, valid, dens)
    yd, _ = densify(y, valid, dens)
    zd, _ = densify(z, valid, dens)
    nrm, _ = normals(xd, yd, zd, vd)
    H, W = xd.shape
    ts = np.arange(-half, half + 1, dtype=np.float64)
    touched = np.zeros(NZ * NY * NX, dtype=bool)
    p_missB = np.zeros((H, W), dtype=bool)
    p_anyA = np.zeros((H, W), dtype=bool)
    block = max(1, 1_500_000 // max(1, W * ts.size))
    for r0 in range(0, H, block):
        r1 = min(H, r0 + block)
        sx = xd[r0:r1, :, None] + nrm[r0:r1, :, 0:1] * ts
        sy = yd[r0:r1, :, None] + nrm[r0:r1, :, 1:2] * ts
        sz = zd[r0:r1, :, None] + nrm[r0:r1, :, 2:3] * ts
        lin = chunk_lin(sx, sy, sz)
        good = (lin >= 0) & vd[r0:r1, :, None]
        li = np.where(good, lin, 0)
        pres_a = ga[li] & good
        p_anyA[r0:r1] = pres_a.any(axis=2)
        p_missB[r0:r1] = (pres_a & ~gb[li]).any(axis=2)
        u = np.unique(li[good])
        touched[u] = True
    idx = np.flatnonzero(touched)
    ta, tb = ga[idx], gb[idx]
    dep = depth[idx]
    return dict(
        seg=d.split("/segments/")[1].split("/")[0],
        grid=[int(H0), int(W0)], valid=int(valid.sum()),
        touched=int(idx.size), in_A=int(ta.sum()), in_B=int(tb.sum()),
        only_A=int((ta & ~tb).sum()), only_B=int((~ta & tb).sum()),
        neither=int((~ta & ~tb).sum()),
        min_depth=int(dep.min()) if idx.size else -1,
        n_depth_le3=int((dep <= 3).sum()),
        points_hit_missing=int(p_missB.sum()),
        points_in_A=int(p_anyA.sum()),
    )


def main():
    cov = np.load(f"{_HERE}/coverage.npz")
    ga = cov["grid_a"].reshape(-1)
    gb = cov["grid_b"].reshape(-1)
    depth = np.load(f"{_HERE}/depth.npz")["depth"].reshape(-1)
    dirs = json.load(open(f"{_HERE}/mesh_dirs.json"))
    root = f"{_INK}/cache/"
    out = []
    for i, d in enumerate(dirs, 1):
        try:
            r = analyse(root + d, ga, gb, depth)
        except Exception as e:  # noqa: BLE001
            r = dict(seg=d.split("/segments/")[1].split("/")[0], error=str(e))
        out.append(r)
        print(f"[{i}/{len(dirs)}] {json.dumps(r)}", flush=True)
    json.dump(out, open(f"{_HERE}/all_segments.json", "w"), indent=1)
    tot_t = sum(r.get("touched", 0) for r in out)
    tot_m = sum(r.get("only_A", 0) for r in out)
    tot_n = sum(r.get("neither", 0) for r in out)
    print(f"\nTOTAL: {len(out)} segments, {tot_t} chunks touched (with repetitions), "
          f"{tot_m} present in A and absent in B, {tot_n} absent in both")


if __name__ == "__main__":
    main()
