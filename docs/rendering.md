# Rendering the ROI from both derivations

This is the first step of the experiment and the one a clean checkout could not previously
reproduce: `renders/` is gitignored, because the arrays are large, and the recipe lived inside
it. The recipe is here instead.

Everything below was run on an M4 laptop, 16 GB, no CUDA. Substitute your own paths for
`$REPO` (this repository) and `$VC3D` (a `volume-cartographer` build).

## The parameters are read, not guessed

They come from the published surface volume's own `.zattrs`:

```
https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0172/segments/
20251107110950-w064_20251107110950052_flatboi/surface-volumes/
7.91um-53keV-volume-20241024131838.zarr/.zattrs
```

which gives `num_slices: 33`, `slice_step: 1.0`, `source_group: 0`,
`canvas_size: [14940, 13420]`, L0 scale `[7.91, 7.91, 7.91]`, and no `accum_*` field. Its
`0/.zarray` gives `shape [33, 13420, 14940]`, so the published render covers the whole canvas
with no crop.

`writeZarrAttrs` in `core/src/Zarr.cpp` writes exactly those fields, and adds
`accum_step`/`accum_type`/`accum_samples` only when accumulation is on. Their absence therefore
means no `--accum`; the rest reads off directly as `--group-idx 0`, `--slice-step 1`,
`--num-slices 33`. The L0 scale of 7.91 with `sZ = baseVoxelSize * sliceStep` and
`sYX = baseVoxelSize / pixelsPerVoxel` gives `--scale 1 --voxel-size 7.91 --voxel-unit micrometer`.

An independent confirmation is the canvas. The tifxyz mesh is 747 x 671 at `scale = 0.05`, and
`round(747/0.05) x round(671/0.05) = 14940 x 13420`, which is `canvas_size` exactly.

The one parameter `.zattrs` does not record is `--flip-normals`, determined below.

## Choosing the ROI

`--crop-x 7168 --crop-y 6400 --crop-width 512 --crop-height 512`.

Level 5 of the published surface volume (420 x 467, 8.6 MB) was downloaded, the mask of
non-zero pixels computed across all 33 slices, and every 24 x 24 L5 block that is entirely
valid — a 128 px L0 margin of real data on each side — enumerated. The block nearest the canvas
centre was taken. Checked at L0: the `(33, 512, 512)` block has zero voxels at zero, and so does
the 128 px frame around it.

The crop is aligned to 128 on both axes, so it coincides with chunk boundaries in the output and
in the published surface volume. In `vc_render_tifxyz.cpp` the canvas origin comes from
`full_size` via `computeCanvasOrigin`, and the crop enters only as an additive offset on
`u0`/`v0`, so a cropped render is bit-identical to the corresponding sub-rectangle of the full
render.

## The commands

The volume itself is never downloaded. `--remote-url` streams over HTTP from the public bucket
and `--cache-gb` is an in-memory cache, not a disk one: `vc_render_tifxyz` with `--remote-url`
stages nothing on the filesystem. This ROI needed **169 chunks per derivation**, roughly 350 MB
of traffic, and about 88 seconds per render.

`--volume` is still required in remote mode, but only to read `voxelsize` and to end up in
`source_zarr` in the output `.zattrs`. Two stub directories, `vol38/` and `vol39/`, each
containing only a `meta.json`, are enough.

Both renders use the **same mesh**, the one shipped with `20241024131838`. The two published
meshes are bit-identical in x, y and z and differ only in `scale` in their `meta.json`
(`0.05` against `0.05000000074505806`); rendering the same volume with the other mesh gives
bit-identical output, 0 voxels different.

```bash
# derivation 20241024131838
$VC3D/build/bin/vc_render_tifxyz \
  --volume $REPO/vol38 \
  --remote-url https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr \
  --segmentation $REPO/mesh/on-131838.tifxyz \
  --scale 1 --group-idx 0 --num-slices 33 --slice-step 1.0 --flip-normals \
  --crop-x 7168 --crop-y 6400 --crop-width 512 --crop-height 512 \
  --zarr-output $REPO/renders/render_131838.zarr \
  --zarr-compressor none --zarr-separator / --pyramid false \
  --voxel-size 7.91 --voxel-unit micrometer \
  --cache-gb 3 --prefetch-remote

# derivation 20241024131839: identical but for the URL and the output name
```

## Validating the renderer against the published surface volume

Voxel by voxel, `render_131838` against `published[:, 6400:6912, 7168:7680]`, 8,650,752 voxels:

```
identical voxels       8,629,175 / 8,650,752   (99.7506%)
maximum difference     1
mean difference        -0.000025
mean absolute          0.002494

histogram
  -1   10,896    (0.1260%)
   0   8,629,175 (99.7506%)
  +1   10,681    (0.1235%)
```

The render matches the published volume everywhere except a quarter of a percent of voxels,
where it differs by exactly one unit in 255. No voxel differs by two or more. The residual is
8-bit truncation of float32 coordinates that differ in the last bit between the build that
produced the published file and this one. There is no difference of parameters, geometry, mesh,
source volume or interpolation, and neither render is more accurate than the other.

### Slice order

With the direct z order only 4.27% of voxels matched; reversed, 99.75%. So the published render
used `--flip-normals`. That flag is literally a z flip and nothing else: rendering with and
without it gives output bit-identical to `render[::-1]`. The code agrees, the offsets being
`(zi - 16) * 1.0`, symmetric about zero, so negating the normal swaps slice `zi` with `32 - zi`.

## Why the two renders are comparable pixel by pixel

Same shape `(33, 512, 512)`, same dtype, same mesh, same crop, same parameters. Pixel `(z, y, x)`
is the same point of the surface in both files by construction: the sampling coordinates depend
only on the mesh and the render parameters, and those are identical.

## Building the renderer

At the time of writing, `volume-cartographer` needed one unmerged change to build on macOS with
Homebrew LLVM: `"CMAKE_FIND_FRAMEWORK": "LAST"` in the `macos-homebrew-llvm` preset in
`CMakePresets.json`, submitted as ScrollPrize/villa#1337. Without it `find_package(TIFF)` takes
its headers from `/Library/Frameworks/Mono.framework/Headers` instead of Homebrew. Then
`./scripts/build_macos.sh --ccache --jobs 6`.
