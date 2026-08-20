# ink-intensity-experiment

The controlled intervention for [ScrollPrize/villa#1372](https://github.com/ScrollPrize/villa/issues/1372):
**on one deterministically selected 512×512 ROI, does the legacy TimeSformer inference path
produce different ink maps from two derivations of one CT scan because their 8-bit values were
scaled differently?**

Result in that scope: yes. Δ@5% between the regenerated ROI maps falls from **0.757 to 0.045**
when one derivation's intensity is remapped onto the other's using coefficients measured away
from the evaluated ROI — a 94.0% reduction — with rank correlation going from 0.510 to 0.999.
This is a causal result for that fixed input/model intervention, not a full-map reproduction,
an estimate across segments, or evidence that intensity explains every published map difference.

The committed-array audit runs on CPU with no network. Full regeneration also runs on a laptop
and uses Apple MPS where available, but it requires the pinned 1.7 GB checkpoint, one 43 MB
published map and roughly 700 MB of streamed render traffic. No credentials or full-volume copy
is required.

## The result

| | Δ@1% | Δ@5% | Δ@20% | ρ |
|---|---|---|---|---|
| control: A vs A | 0.000 | 0.000 | 0.000 | 1.000 |
| arm 1: A vs B | 0.865 | **0.757** | 0.650 | 0.510 |
| arm 2: A vs remap(B), coefficients fit on this ROI | 0.037 | **0.044** | 0.031 | 0.999 |
| held-out coefficients, fit on five scan chunks | — | **0.045** | — | 0.999 |
| reference: B vs remap(B) | 0.869 | 0.757 | 0.648 | 0.512 |

A is `PHerc0172/volumes/20241024131838-…`, B is `…131839-…`, both derivations of one scan.
These are model outputs regenerated on the ROI, not crops of the two published ink maps. Under
the other blending convention the reduction is 96.6%, from 0.771 to 0.026.

## It does not restate its own fit

The first arm-2 row uses `A = 0.6165*B + 104.02`, measured voxel-by-voxel on these renders, so it
would be circular if that were the only evidence. The headline 0.045 instead uses
`A = 0.6154*B + 104.32`, measured away from the ROI. `fit_affine.py` and `sweep_affine.py` make
that distinction explicit:

| coefficients | Δ@5% | |
|---|---|---|
| identity, no remap | 0.757 | arm 1 restated |
| intercept only | **0.882** | worse than doing nothing |
| slope only | 0.594 | |
| fitted on these renders | 0.044 | in-sample |
| fitted on five volume chunks that never touch the ROI | **0.045** | out-of-sample |

The in-sample fit is worth a tenth of a point out of ninety-four. Neither half of the affine
does the work alone, and the offset alone makes the disagreement worse, so this is not "any move
that makes the inputs look more alike". Those independent coefficients also carry 99.31% of the
region's voxels to within one 8-bit level.

## Checking the result without running anything

The ink maps are committed, so every figure above can be recomputed from this repository with
numpy alone: no torch, no checkpoint, no renders, no network.

```python
import numpy as np
a  = np.load("out/ts_hann_A.npy")                         # model on derivation A
b  = np.load("out/ts_hann_B.npy")                         # same model on B
bm = np.load("out/ts_hann_Bmap_0p6154_104p32.npy")         # held-out affine remap

def delta(x, y, q=0.05):
    k = round(q * x.size)
    def exact_topk(z):
        flat = z.ravel()
        idx = np.argpartition(flat, flat.size-k)[flat.size-k:]
        mask = np.zeros(flat.size, dtype=bool)
        mask[idx] = True
        return mask
    sx, sy = exact_topk(x), exact_topk(y)
    return 1 - (sx & sy).sum() / (sx | sy).sum()

print(delta(a, a), delta(a, b), delta(a, bm))   # 0.000  0.757  0.045
```

About five seconds. `python -m pytest -q tests/test_artifacts.py` checks the headline, the
calibration record and `ARTIFACTS.sha256` using only numpy and pytest. Everything below is how
the arrays were produced.

## The relevance gate

The experiment refuses to print its causal arms until its regenerated A map has substantial
agreement with the published map made by the same checkpoint. `calibrate.py` is that gate, and
it exits non-zero on failure rather than printing a warning.

| | IoU@5% |
|---|---|
| our reproduction vs published july map | **0.449** |
| published july vs published november | 0.284 |
| chance | 0.026 |

IoU 0.449 is a partial-reproduction proxy, not identity with the published output. It establishes
that this configuration is relevant to the checkpoint and resembles the July map more than the
two named published maps resemble each other. The 0.285 mark is that comparator, not a universal
validation threshold.

A first attempt with `scrollprize/PHerc.1667-iteration-0` gave the same qualitative answer (96.8%
reduction) and scored 0.081 here, far below the 0.285 pass mark though above chance. It is in the
repository and in `out/`, excluded from the headline rather than hidden. An out-of-distribution
model being sensitive to intensity says nothing about the published maps.

## The depth window is the reference one, not a guess

`ink-detection/inference_timesformer.py` defaults to `start_idx=17` with `CFG.in_chans=26` and
builds `range(start_idx, start_idx + in_chans)`, so the reference stack is legacy layers 17
through 42. A 33-slice render centred on the surface spans offsets −16…+16, that is legacy
16…48, so 17…42 lands on local indices 1…26: `[1:27)`.

An earlier version of this repository used `[1:31)` with 30 frames and said the window was
recorded nowhere. It was recorded, in the source, and the sweep that chose 30 frames only ever
tried the centred start for each frame count, so it never tested `[1:27)` at all. Scored against
the published map: `[3:29)` gives 0.395, `[1:31)` gives 0.420, `[1:27)` gives 0.449. The window
with the independent justification is also the best-scoring one. The sweep now enumerates every
start for every count.

## Reproducing

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Materialises the exact checkpoint revision under the path infer/ expects.
HF_HOME="$PWD/weights/hf" .venv/bin/hf download \
  scrollprize/timesformer_scroll5_july_retreat \
  --revision 5b714296b256b8f993ce69e8e57aea585125d782

# Put the published July TIFF in data/, or point this at another directory.
mkdir -p data
curl --fail --location \
  "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0172/segments/20251107110950-w064_20251107110950052_flatboi/ink-detection/PHerc0172-20251107110950-7.91um-53keV-volume-20241024131838-20250713185324-timesformer_scroll5_july_retreat-tile64-stride16.tif" \
  --output data/PHerc0172-20251107110950-7.91um-53keV-volume-20241024131838-20250713185324-timesformer_scroll5_july_retreat-tile64-stride16.tif
shasum -a 256 -c SOURCE_DATA.sha256
export OP6_PUBLISHED_DATA="$PWD/data"
.venv/bin/python calibrate.py                 # relevance gate; refuses to proceed if it fails
.venv/bin/python experiment_timesformer.py    # three arms, both blending conventions
```

Renders are produced by `vc_render_tifxyz` from
[ScrollPrize/villa](https://github.com/ScrollPrize/villa). [docs/rendering.md](docs/rendering.md)
has the exact command lines, where every parameter comes from (the published `.zattrs`, not
guesswork), how the ROI was chosen, and the validation: the render matches the published surface
volume on 99.75% of voxels, with a maximum difference of one unit in 255 and none of two.
Coverage analysis is in `coverage/`, and needs no payload download at all: it enumerates chunk
presence with `ListObjectsV2`, 96 seconds for the whole scroll. The coverage scripts and archived
ResNet experiment additionally require a neighbouring `inkfloor` checkout; they are not part of
the headline TimeSformer run.

## Legacy path versus the current normaliser

The causal arms above intentionally reproduce the legacy published TimeSformer preprocessing:
absolute clipping at 200 followed by constant division. They do **not** test the current
`vesuvius.ink_detection` model end to end. As a separate input-level companion,
`normalization_gap.py` applies the current `normalize_robust` median/MAD default to 16 matched
64×64×33 crops. On these renders the median relative input gap falls from 52.1% under the legacy
constant scaling to 3.0% under robust MAD (worst crop 58.3% to 6.4%). That supports the proposed
mechanism, but it is not a current-model accuracy or stability result. Pass `--villa ../villa`
to assert that the local mirror still matches villa's implementation.

## What the code will not do

It will not tell you the cause generalizes. This is one 512×512 ROI from one segment, selected
as the nearest-to-centre fully valid aligned crop. Per the census in
[inkfloor](https://github.com/nerln/inkfloor) that one segment is the entire population of
same-model, different-derivation pairs in the published corpus as of 8 August 2026.

It will not tell you which derivation reads better either. Δ compares two outputs without
scoring either of them, and there is no infrared ground truth in scrolls, so nothing here is a
statement about legibility. What is measured is that the pipeline is unstable with respect to an
input nobody controls, and that a remap measured independently removes almost all of it.

## Layout

```
calibrate.py                the partial-reproduction/relevance gate
experiment.py               archived ResNet3D run (failed the gate; needs sibling inkfloor)
experiment_timesformer.py   three arms, the checkpoint that produced the published map
infer/                      standalone loaders and tiling engine; rejects wrong input shapes
coverage/                   exact chunk-coverage enumeration and the per-segment mask
tools/                      render comparison helpers
out/                        JSON reports and the ink maps themselves, both committed
```

Weights, renders and the S3 cache are gitignored: 1.7 GB, 136 MB and 27 MB respectively, all
regenerable.

MIT.
