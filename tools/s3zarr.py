"""Lettura minimale di zarr v2 non compressi dal bucket pubblico, via HTTP."""
from __future__ import annotations
import json, urllib.request, urllib.error
from pathlib import Path
import numpy as np

HOST = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
CACHE = Path("/Volumes/AppsAndFiles/dev/op6-causal/cache")

def _get(key: str) -> bytes:
    dst = CACHE / key
    if dst.exists() and dst.stat().st_size > 0:
        return dst.read_bytes()
    with urllib.request.urlopen(f"{HOST}/{key}", timeout=300) as r:
        b = r.read()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b)
    return b

class Arr:
    def __init__(self, prefix: str, level: int):
        self.prefix = prefix.rstrip("/")
        self.level = level
        self.meta = json.loads(_get(f"{self.prefix}/{level}/.zarray"))
        assert self.meta["compressor"] is None, "solo zarr non compressi"
        self.shape = tuple(self.meta["shape"])
        self.chunks = tuple(self.meta["chunks"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.sep = self.meta.get("dimension_separator", ".")

    def chunk(self, iz, iy, ix):
        key = f"{self.prefix}/{self.level}/" + self.sep.join(map(str, (iz, iy, ix)))
        try:
            b = _get(key)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return np.zeros(self.chunks, self.dtype)   # fill_value
            raise
        return np.frombuffer(b, self.dtype).reshape(self.chunks)

    def read(self, z0, z1, y0, y1, x0, x1):
        cz, cy, cx = self.chunks
        out = np.zeros((z1-z0, y1-y0, x1-x0), self.dtype)
        for iz in range(z0//cz, (z1-1)//cz + 1):
            for iy in range(y0//cy, (y1-1)//cy + 1):
                for ix in range(x0//cx, (x1-1)//cx + 1):
                    c = self.chunk(iz, iy, ix)
                    gz, gy, gx = iz*cz, iy*cy, ix*cx
                    sz0, sz1 = max(z0, gz), min(z1, gz+cz)
                    sy0, sy1 = max(y0, gy), min(y1, gy+cy)
                    sx0, sx1 = max(x0, gx), min(x1, gx+cx)
                    if sz0>=sz1 or sy0>=sy1 or sx0>=sx1: continue
                    out[sz0-z0:sz1-z0, sy0-y0:sy1-y0, sx0-x0:sx1-x0] = \
                        c[sz0-gz:sz1-gz, sy0-gy:sy1-gy, sx0-gx:sx1-gx]
        return out
