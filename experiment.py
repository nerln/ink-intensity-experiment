#!/usr/bin/env python3
"""L'esperimento causale per ScrollPrize/villa#1372.

Domanda: la differenza fra le mappe di inchiostro prodotte su due derivazioni
dello stesso scan e' causata dalla rimappatura affine degli 8 bit?

Tre bracci:
  1. STESSO checkpoint sui due render -> Delta osservato
  2. STESSO checkpoint su B rimappato su A -> se Delta collassa, e' l'intensita'
  3. controlli: motore contro se stesso (deve dare 0), e il pavimento imposto
     dai voxel che B ha perso e che nessuna rimappatura lineare recupera

Convenzioni: il render a 63 slice e' un sovrainsieme delle due finestre
candidate, quindi [0:62] = -31..+30 e [1:63] = -30..+31. reverse e' [::-1].
Le quattro combinazioni si scelgono confrontando con la mappa PUBBLICATA per
131838, sapendo che viene da un'altra famiglia di modelli: cerchiamo quale
convenzione batte le altre, non un accordo alto in assoluto.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Volumes/AppsAndFiles/dev/op6-causal")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Volumes/AppsAndFiles/dev/inkfloor")

from infer import InkEngine  # noqa: E402
from inkfloor import metrics  # noqa: E402

R = ROOT / "renders"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

# La ROI nel canvas completo, dal README del renderer.
CROP_X, CROP_Y, CROP_W, CROP_H = 7168, 6400, 512, 512

# Relazione misurata sui render, voxel per voxel: A = slope*B + intercept
SLOPE, INTERCEPT = 0.6165, 104.02


def window(d63: np.ndarray, start: int, reverse: bool) -> np.ndarray:
    """Estrae 62 slice dal render a 63, in (H, W, D)."""
    w = d63[:, :, start : start + 62]
    return w[:, :, ::-1].copy() if reverse else w.copy()


def remap_b_to_a(b: np.ndarray) -> np.ndarray:
    """Applica la relazione affine misurata, con saturazione a uint8."""
    x = b.astype(np.float64) * SLOPE + INTERCEPT
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def delta(a: np.ndarray, b: np.ndarray, valid: np.ndarray, q: float) -> float:
    return 1.0 - metrics.delta_at_q(a, b, valid, q).iou


def main() -> None:
    A63 = np.load(R / "render_131838_hwd63.npy")
    B63 = np.load(R / "render_131839_hwd63.npy")
    print(f"render A {A63.shape} {A63.dtype}   render B {B63.shape} {B63.dtype}")

    pub_full = np.load(R / "published_131838_roi.npy")  # (33,512,512), gia' ROI
    print(f"pubblicato (33 slice) {pub_full.shape}")

    engine = InkEngine()
    print(f"motore pronto\n")

    # --- scelta della convenzione, contro la mappa di inchiostro PUBBLICATA ---
    # La predizione pubblicata per 131838 sta nel .tif intero: ritaglio la ROI.
    import tifffile

    pred_key = next(
        p for p in (ROOT.parent / "vesuvius-op6" / "data").glob("*.tif")
        if "131838" in p.name and "july" in p.name
    )
    pred_full = tifffile.imread(pred_key)
    pub_ink = pred_full[CROP_Y : CROP_Y + CROP_H, CROP_X : CROP_X + CROP_W].astype(np.float32)
    print(f"mappa di inchiostro pubblicata sulla ROI: {pub_ink.shape}, "
          f"non-zero {np.count_nonzero(pub_ink)/pub_ink.size:.3f}\n")

    print("=== scelta della convenzione (su A, contro la mappa pubblicata) ===")
    print(f"  {'finestra':<12} {'reverse':<8} {'rho con pubblicata':>20}")
    best, best_rho = None, -2.0
    conv_rows = []
    for start, rev in itertools.product((0, 1), (False, True)):
        ink = engine.predict(window(A63, start, rev))
        v = np.ones(ink.shape, bool)
        rho = metrics.spearman(ink.astype(np.float32), pub_ink, v)
        tag = f"[{start}:{start+62}]"
        print(f"  {tag:<12} {str(rev):<8} {rho:>20.4f}")
        conv_rows.append({"start": start, "reverse": rev, "rho": float(rho)})
        if rho > best_rho:
            best, best_rho = (start, rev), rho
    start, rev = best
    spread = best_rho - min(r["rho"] for r in conv_rows)
    print(f"\n  scelta: finestra [{start}:{start+62}], reverse={rev}, rho={best_rho:.4f}")
    print(f"  distacco dalla peggiore: {spread:.4f}"
          f"  -> {'la convenzione e decisa dai dati' if spread > 0.05 else 'NON decisiva, vedi nota'}\n")

    # --- i tre bracci ---
    A = window(A63, start, rev)
    B = window(B63, start, rev)
    Bmap = remap_b_to_a(B)

    print("=== inferenza ===")
    inkA = engine.predict(A)
    inkB = engine.predict(B)
    inkBm = engine.predict(Bmap)
    inkA2 = engine.predict(A)  # controllo di determinismo

    valid = np.ones(inkA.shape, bool)
    qs = (0.01, 0.05, 0.20)

    print(f"\n  {'confronto':<44} " + " ".join(f"{'D@'+str(int(q*100))+'%':>8}" for q in qs))
    rows = []
    for label, x, y in [
        ("controllo: A contro A (deve essere 0)", inkA, inkA2),
        ("BRACCIO 1: A contro B (osservato)", inkA, inkB),
        ("BRACCIO 2: A contro B rimappato", inkA, inkBm),
        ("riferimento: B contro B rimappato", inkB, inkBm),
    ]:
        ds = [delta(x, y, valid, q) for q in qs]
        rows.append({"label": label, "deltas": ds,
                     "spearman": float(metrics.spearman(x, y, valid))})
        print(f"  {label:<44} " + " ".join(f"{d:8.3f}" for d in ds))

    d_obs = rows[1]["deltas"][1]
    d_rem = rows[2]["deltas"][1]
    print(f"\n  rho: " + ", ".join(f"{r['label'].split(':')[0]}={r['spearman']:.3f}" for r in rows))

    print(f"\n=== VERDETTO (a q=5%) ===")
    print(f"  Delta osservato fra le due derivazioni : {d_obs:.3f}")
    print(f"  Delta dopo la rimappatura affine       : {d_rem:.3f}")
    if d_obs > 0:
        print(f"  riduzione                              : {(d_obs-d_rem)/d_obs:+.1%}")

    # pavimento imposto dall'informazione persa
    lost = (B == 0) & (A > 0)
    print(f"\n  voxel persi da B (zero dove A ha dato): {lost.sum():,} "
          f"({lost.mean():.4%}) -> pavimento non recuperabile per costruzione")

    json.dump({"convention": {"start": start, "reverse": rev, "rho": best_rho,
                              "spread": float(spread), "all": conv_rows},
               "rows": rows, "qs": list(qs),
               "lost_voxels": int(lost.sum()), "lost_frac": float(lost.mean()),
               "remap": {"slope": SLOPE, "intercept": INTERCEPT}},
              open(OUT / "experiment.json", "w"), indent=1)
    np.savez_compressed(OUT / "ink_maps.npz", A=inkA, B=inkB, Bmap=inkBm)
    print(f"\n  salvato in {OUT}")


if __name__ == "__main__":
    main()
