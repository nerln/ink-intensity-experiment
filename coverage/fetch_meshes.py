"""Scarica le tifxyz di tutti i segmenti di PHerc0172 (derivazione 131838) nella cache."""
from __future__ import annotations

from pathlib import Path
import json
import sys
from concurrent.futures import ThreadPoolExecutor

_HERE = str(Path(__file__).resolve().parent)
_REPO = str(Path(__file__).resolve().parent.parent)
_INK = str(Path(__file__).resolve().parent.parent.parent / "inkfloor")

sys.path.insert(0, _INK)
from inkfloor import cache  # noqa: E402



VOL = "20241024131838"


def one(d: str):
    got = []
    for c in "xyz":
        try:
            got.append(str(cache.fetch(f"{d}/{c}.tif")))
        except cache.FetchError as e:
            return d, f"ERR {e}"
    return d, "ok"


def main():
    meshes = json.load(open(f"{_HERE}/segment_meshes.json"))
    dirs = []
    for seg, ms in meshes.items():
        pick = [m for m in ms if f"-on-{VOL}-" in m.rsplit("/", 1)[-1]]
        if pick:
            dirs.append(pick[0])
        elif ms:
            dirs.append(ms[0])
    print(len(dirs), "tifxyz da scaricare", flush=True)
    with ThreadPoolExecutor(8) as p:
        for i, (d, st) in enumerate(p.map(one, dirs), 1):
            if st != "ok" or i % 10 == 0:
                print(f"{i}/{len(dirs)} {d.rsplit('/',1)[-1]} {st}", flush=True)
    json.dump(dirs, open(f"{_HERE}/mesh_dirs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
