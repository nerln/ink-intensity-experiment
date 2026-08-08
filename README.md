# ink-intensity-experiment

The causal experiment for [ScrollPrize/villa#1372](https://github.com/ScrollPrize/villa/issues/1372):
**do two published derivations of one CT scan produce different ink maps because of how their
8-bit values were scaled?**

Answer: yes. Δ@5% between the two ink maps falls from **0.748 to 0.054** when one derivation's
intensity is remapped onto the other's — a 92.8% reduction — with rank correlation going from
0.457 to 0.999.

Everything runs on a laptop. No GPU, no credentials, no bulk download.

## The result

| | Δ@1% | Δ@5% | Δ@20% | ρ |
|---|---|---|---|---|
| control: A vs A | 0.000 | 0.000 | 0.000 | 1.000 |
| arm 1: A vs B | 0.754 | **0.748** | 0.638 | 0.457 |
| arm 2: A vs remap(B) | 0.025 | **0.054** | 0.027 | 0.999 |
| reference: B vs remap(B) | 0.754 | 0.747 | 0.638 | 0.459 |

A is `PHerc0172/volumes/20241024131838-…`, B is `…131839-…`, both derivations of one scan.
The remap is `A = 0.6165*B + 104.02`, measured voxel-by-voxel on the renders.

## The validity gate

The experiment refuses to report causal numbers until it has reproduced a *published*
prediction with the checkpoint that produced it. `calibrate.py` is that gate.

| | IoU@5% |
|---|---|
| our reproduction vs published july map | **0.420** |
| our reproduction vs published november map | 0.279 |
| published july vs published november | 0.284 |
| chance | 0.026 |

The agreement is specific to the checkpoint we ran: our map resembles the july map more than
it resembles the november one, and more than the two published maps resemble each other.

A first attempt with `scrollprize/PHerc.1667-iteration-0` gave the same qualitative answer
(96.8% reduction) but failed this gate — its calibration sat at chance. That run is not
reported. An out-of-distribution model being sensitive to intensity says nothing about the
published maps.

## Reproducing

```bash
python -m pip install torch numpy tifffile imagecodecs numcodecs safetensors huggingface_hub
python calibrate.py                 # the gate; refuses to proceed if it fails
python experiment_timesformer.py    # the three arms, under both blending conventions
```

Renders are produced by `vc_render_tifxyz` from
[ScrollPrize/villa](https://github.com/ScrollPrize/villa); `renders/README.md` has the exact
command lines and the parameters, all derived from the published `.zattrs` rather than guessed.
Coverage analysis is in `coverage/`, and needs no payload download at all: it enumerates chunk
presence with `ListObjectsV2`, 96 seconds for the whole scroll.

## What the code will not do

It will not choose the layer window for you. The number of frames the published run used is not
recorded anywhere — `optimized_inference/entrypoint.py` reads `START_LAYER`/`END_LAYER` from the
environment with no default — so the engine **refuses to start without an explicit
`num_frames`** rather than picking one quietly. The calibration sweep in `calibrate.py` narrows
it empirically to 30 frames, which happens to match the only hardcoded value in the repo, but
all three candidate windows pass the gate and the conclusion holds across all of them
(91.4%–92.8%).

It also will not tell you the cause generalizes. This is one segment, and per the census in
[inkfloor](https://github.com/nerln/inkfloor) that one segment is the entire population of
same-model, different-derivation pairs in the published corpus.

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
