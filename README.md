# ink-intensity-experiment

The causal experiment for [ScrollPrize/villa#1372](https://github.com/ScrollPrize/villa/issues/1372):
**do two published derivations of one CT scan produce different ink maps because of how their
8-bit values were scaled?**

Answer: yes. Δ@5% between the two ink maps falls from **0.757 to 0.045** when one derivation's
intensity is remapped onto the other's using coefficients measured off the evaluated region — a
94.0% reduction — with rank correlation going from 0.510 to 0.999.

Everything runs on a laptop. No CUDA, no credentials, no bulk download; inference uses Apple MPS
where available.

## The result

| | Δ@1% | Δ@5% | Δ@20% | ρ |
|---|---|---|---|---|
| control: A vs A | 0.000 | 0.000 | 0.000 | 1.000 |
| arm 1: A vs B | 0.865 | **0.757** | 0.650 | 0.510 |
| arm 2: A vs remap(B) | 0.037 | **0.044** | 0.031 | 0.999 |
| reference: B vs remap(B) | 0.869 | 0.757 | 0.648 | 0.512 |

A is `PHerc0172/volumes/20241024131838-…`, B is `…131839-…`, both derivations of one scan.
Under the other blending convention the reduction is 96.6%, from 0.771 to 0.026.

## It does not restate its own fit

The remap above is `A = 0.6165*B + 104.02`, measured voxel-by-voxel on these renders, so arm 2
would be circular if that were the only evidence. `fit_affine.py` and `sweep_affine.py` are the
answer:

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

## The validity gate

The experiment refuses to print its causal arms until it has reproduced a *published* prediction
with the checkpoint that produced it. `calibrate.py` is that gate, and it exits non-zero on
failure rather than printing a warning.

| | IoU@5% |
|---|---|
| our reproduction vs published july map | **0.449** |
| published july vs published november | 0.284 |
| chance | 0.026 |

The agreement is specific to the checkpoint we ran: our map resembles the july map more than the
two published maps resemble each other.

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
python -m pip install torch numpy tifffile imagecodecs numcodecs safetensors huggingface_hub
python calibrate.py                 # the gate; refuses to proceed if it fails
python experiment_timesformer.py    # the three arms, under both blending conventions
```

Renders are produced by `vc_render_tifxyz` from
[ScrollPrize/villa](https://github.com/ScrollPrize/villa). [docs/rendering.md](docs/rendering.md)
has the exact command lines, where every parameter comes from (the published `.zattrs`, not
guesswork), how the ROI was chosen, and the validation: the render matches the published surface
volume on 99.75% of voxels, with a maximum difference of one unit in 255 and none of two.
Coverage analysis is in `coverage/`, and needs no payload download at all: it enumerates chunk
presence with `ListObjectsV2`, 96 seconds for the whole scroll.

## What the code will not do

It will not tell you the cause generalizes. This is one segment, and per the census in
[inkfloor](https://github.com/nerln/inkfloor) that one segment is the entire population of
same-model, different-derivation pairs in the published corpus as of 8 August 2026.

It will not tell you which derivation reads better either. Δ compares two outputs without
scoring either of them, and there is no infrared ground truth in scrolls, so nothing here is a
statement about legibility. What is measured is that the pipeline is unstable with respect to an
input nobody controls, and that a remap measured independently removes almost all of it.

## Layout

```
calibrate.py                the validity gate
experiment.py               three arms, resnet3d checkpoint (failed the gate, kept for the record)
experiment_timesformer.py   three arms, the checkpoint that produced the published map
infer/                      standalone loaders and tiling engine; rejects wrong input shapes
coverage/                   exact chunk-coverage enumeration and the per-segment mask
tools/                      render comparison helpers
out/                        JSON reports (committed); ink maps (not)
```

Weights, renders and the S3 cache are gitignored: 1.7 GB, 136 MB and 27 MB respectively, all
regenerable.

MIT.
