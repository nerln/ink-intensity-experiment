"""Quali chunk attraversa la superficie renderizzata di un segmento, e quali mancano in B.

La superficie ha spessore: il render campiona lungo la normale per ~261 um, cioe' ~33
slice a 7.91 um/voxel. Qui si campiona t = -16..+16 voxel lungo la normale stimata dal
campo di posizioni della tifxyz.

Uso: mesh_chunks.py <dir tifxyz> [--dens N] [--out prefix]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import tifffile

CH = 128
SHAPE_A = (21000, 6700, 9100)
SHAPE_B = (20820, 6700, 9100)
NZ, NY, NX = 165, 53, 72
HALF = 16          # t = -16..+16  ->  33 campioni, 261 um a 7.91 um


def load_mesh(d: str):
    x = tifffile.imread(f"{d}/x.tif").astype(np.float64)
    y = tifffile.imread(f"{d}/y.tif").astype(np.float64)
    z = tifffile.imread(f"{d}/z.tif").astype(np.float64)
    valid = (x >= 0) & (y >= 0) & (z >= 0)
    return x, y, z, valid


def densify(a: np.ndarray, valid: np.ndarray, k: int):
    """Interpolazione bilineare del campo di posizioni, fattore k per asse.

    I punti invalidi restano invalidi e contaminano i loro 4 vicini: meglio perdere una
    frangia di un pixel che inventare coordinate sul bordo del segmento.
    """
    if k == 1:
        return a, valid
    h, w = a.shape
    gi = np.linspace(0, h - 1, (h - 1) * k + 1)
    gj = np.linspace(0, w - 1, (w - 1) * k + 1)
    i0 = np.floor(gi).astype(int).clip(0, h - 2)
    j0 = np.floor(gj).astype(int).clip(0, w - 2)
    fi = (gi - i0)[:, None]
    fj = (gj - j0)[None, :]
    A = a[np.ix_(i0, j0)]
    B = a[np.ix_(i0 + 1, j0)]
    C = a[np.ix_(i0, j0 + 1)]
    D = a[np.ix_(i0 + 1, j0 + 1)]
    out = (A * (1 - fi) * (1 - fj) + B * fi * (1 - fj)
           + C * (1 - fi) * fj + D * fi * fj)
    V = (valid[np.ix_(i0, j0)] & valid[np.ix_(i0 + 1, j0)]
         & valid[np.ix_(i0, j0 + 1)] & valid[np.ix_(i0 + 1, j0 + 1)])
    return out, V


def normals(x, y, z, valid):
    """Normale unitaria per punto, dal prodotto vettoriale delle derivate di griglia.

    Dove una derivata non e' calcolabile (bordo del segmento, buchi) la normale e' NaN e
    il chiamante campiona solo il punto centrale.
    """
    xf = np.where(valid, x, np.nan)
    yf = np.where(valid, y, np.nan)
    zf = np.where(valid, z, np.nan)
    du = [np.gradient(c, axis=0) for c in (xf, yf, zf)]
    dv = [np.gradient(c, axis=1) for c in (xf, yf, zf)]
    n = [du[1] * dv[2] - du[2] * dv[1],
         du[2] * dv[0] - du[0] * dv[2],
         du[0] * dv[1] - du[1] * dv[0]]
    norm = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    bad = ~np.isfinite(norm) | (norm < 1e-9)
    norm = np.where(bad, 1.0, norm)
    out = np.stack([np.where(bad, 0.0, c / norm) for c in n], axis=-1)
    return out, ~bad


def chunk_lin(x, y, z):
    """Indice lineare del chunk (z,y,x) sulla griglia comune; -1 fuori dall'array A."""
    xi = np.floor(x / CH).astype(np.int64)
    yi = np.floor(y / CH).astype(np.int64)
    zi = np.floor(z / CH).astype(np.int64)
    ok = (xi >= 0) & (xi < NX) & (yi >= 0) & (yi < NY) & (zi >= 0) & (zi < NZ)
    lin = (zi * NY + yi) * NX + xi
    return np.where(ok, lin, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tifxyz")
    ap.add_argument("--dens", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cov", default="/Volumes/AppsAndFiles/dev/op6-causal/coverage/coverage.npz")
    a = ap.parse_args()

    cov = np.load(a.cov)
    ga = cov["grid_a"].reshape(-1)
    gb = cov["grid_b"].reshape(-1)

    x, y, z, valid = load_mesh(a.tifxyz)
    H0, W0 = x.shape
    xd, vd = densify(x, valid, a.dens)
    yd, _ = densify(y, valid, a.dens)
    zd, _ = densify(z, valid, a.dens)
    nrm, has_n = normals(xd, yd, zd, vd)
    H, W = xd.shape

    ts = np.arange(-HALF, HALF + 1, dtype=np.float64)
    # per punto: unione dei chunk toccati dai 33 campioni lungo la normale
    touched = np.zeros(NZ * NY * NX, dtype=bool)
    p_any_a = np.zeros((H, W), dtype=bool)        # tocca almeno un chunk presente in A
    p_missB = np.zeros((H, W), dtype=bool)        # tocca un chunk presente in A e assente in B
    p_nsamp_missB = np.zeros((H, W), dtype=np.int16)

    block = max(1, 2_000_000 // (W * ts.size))
    for r0 in range(0, H, block):
        r1 = min(H, r0 + block)
        sx = xd[r0:r1, :, None] + nrm[r0:r1, :, 0:1] * ts
        sy = yd[r0:r1, :, None] + nrm[r0:r1, :, 1:2] * ts
        sz = zd[r0:r1, :, None] + nrm[r0:r1, :, 2:3] * ts
        lin = chunk_lin(sx, sy, sz)
        good = (lin >= 0) & vd[r0:r1, :, None]
        li = np.where(good, lin, 0)
        pres_a = ga[li] & good
        miss_b = pres_a & ~gb[li]
        touched[np.unique(li[good])] = True
        p_any_a[r0:r1] = pres_a.any(axis=2)
        p_missB[r0:r1] = miss_b.any(axis=2)
        p_nsamp_missB[r0:r1] = miss_b.sum(axis=2)

    idx = np.flatnonzero(touched)
    t_in_a = ga[idx]
    t_in_b = gb[idx]
    n_touch = idx.size
    n_a = int(t_in_a.sum())
    n_b = int(t_in_b.sum())
    n_only_a = int((t_in_a & ~t_in_b).sum())
    n_only_b = int((~t_in_a & t_in_b).sum())
    n_neither = int((~t_in_a & ~t_in_b).sum())

    res = dict(
        tifxyz=a.tifxyz, dens=a.dens, grid=[int(H0), int(W0)], grid_dens=[int(H), int(W)],
        valid_points=int(valid.sum()),
        chunks_touched=int(n_touch),
        touched_present_A=n_a, touched_present_B=n_b,
        touched_only_A=n_only_a, touched_only_B=n_only_b, touched_neither=n_neither,
        frac_missing_of_A=(n_only_a / n_a) if n_a else 0.0,
        points_valid_dens=int(vd.sum()),
        points_touching_A=int(p_any_a.sum()),
        points_hit_missing=int(p_missB.sum()),
        frac_points_hit_missing=(float(p_missB.sum()) / float(p_any_a.sum())) if p_any_a.sum() else 0.0,
        normals_ok=int(has_n.sum()),
    )
    print(json.dumps(res, indent=2))
    if a.out:
        np.savez_compressed(
            a.out + ".npz",
            mask_ok=(p_any_a & ~p_missB), mask_hit=p_missB, nsamp_missing=p_nsamp_missB,
            touched=np.flatnonzero(touched).astype(np.int32),
            valid=vd,
        )
        with open(a.out + ".json", "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
