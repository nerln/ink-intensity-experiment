from __future__ import annotations
from pathlib import Path
"""Elenca via LIST S3 tutti i chunk memorizzati al livello 0 delle due derivazioni.

Non sonda: la LIST restituisce la copertura ESATTA e completa, non un campione.
Output: coverage.npz con due bitmap booleane sulla griglia di chunk (nz, ny, nx)
e le somme delle dimensioni per chunk (utile per distinguere chunk pieni da tappi).
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, _INK)
from inkfloor import cache  # noqa: E402

_HERE = str(Path(__file__).resolve().parent)
_REPO = str(Path(__file__).resolve().parent.parent)
_INK = str(Path(__file__).resolve().parent.parent.parent / "inkfloor")


VOL_A = "PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr/0"
VOL_B = "PHerc0172/volumes/20241024131839-7.910um-53keV-masked.zarr/0"
SHAPE_A = (21000, 6700, 9100)
SHAPE_B = (20820, 6700, 9100)
CH = 128

NZ = max(-(-SHAPE_A[0] // CH), -(-SHAPE_B[0] // CH))
NY = -(-SHAPE_A[1] // CH)
NX = -(-SHAPE_A[2] // CH)


def slab(args):
    prefix, zi = args
    for attempt in range(5):
        try:
            return zi, cache.list_keys(f"{prefix}/{zi}/")
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    return zi, []


def coverage(prefix: str, nz: int):
    grid = np.zeros((NZ, NY, NX), dtype=bool)
    size = np.zeros((NZ, NY, NX), dtype=np.int64)
    tasks = [(prefix, zi) for zi in range(nz)]
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for zi, keys in pool.map(slab, tasks):
            for key, sz in keys:
                parts = key.rsplit("/", 3)
                z, y, x = int(parts[1]), int(parts[2]), int(parts[3])
                grid[z, y, x] = True
                size[z, y, x] = sz
            done += 1
            if done % 20 == 0:
                print(f"  {prefix.rsplit('/',2)[-2][:14]} {done}/{nz} slab", flush=True)
    return grid, size


def main():
    t0 = time.time()
    print("listing A ...", flush=True)
    ga, sa = coverage(VOL_A, -(-SHAPE_A[0] // CH))
    print(f"A: {ga.sum()} chunks, {time.time()-t0:.0f}s", flush=True)
    t1 = time.time()
    print("listing B ...", flush=True)
    gb, sb = coverage(VOL_B, -(-SHAPE_B[0] // CH))
    print(f"B: {gb.sum()} chunks, {time.time()-t1:.0f}s", flush=True)
    np.savez_compressed(
        f"{_HERE}/coverage.npz",
        grid_a=ga, grid_b=gb, size_a=sa, size_b=sb,
        shape_a=np.array(SHAPE_A), shape_b=np.array(SHAPE_B), chunk=CH,
    )
    print("saved coverage.npz")


if __name__ == "__main__":
    main()
