`get_img_splits`' overlap-add weights the numerator with a gaussian kernel and the denominator with ones, then divides one by the other:

```python
mask_count_kernel = np.ones((CFG.size, CFG.size))
kernel = gkern(CFG.size, 1)
kernel = kernel / kernel.max()
...
y_preds_multiplied = y_preds_resized * kernel_tensor
...
mask_pred[y1:y2, x1:x2]  += y_preds_multiplied_cpu[i]
mask_count[y1:y2, x1:x2] += mask_count_kernel     # <- ones, not the kernel
...
mask_pred /= np.clip(mask_count, a_min=1, a_max=None)
```

That is not a weighted average. The numerator carries the kernel, the denominator does not.

## What it costs

With `CFG.size = 64` and `gkern(64, 1)` normalized to max 1, the kernel's mean is **0.732**, so the output is attenuated by about 27%.

The attenuation is not uniform, which is the part that matters. The kernel is 1.0 at each tile centre and 0.380 at its edge, so the error carries a pattern at the tile-grid frequency — a periodic ripple rather than a constant a downstream threshold could absorb.

`optimized_inference/inference.py` does this correctly: both `hann2d` and `gkern` there are normalized to sum 1 and the same kernel is accumulated on both sides. Comparing the two paths is what made this visible.

## The change

Accumulate `kernel` instead of `np.ones` and drop the now-unused constant. One line.

## The consequence you need to weigh

**This changes inference output for anyone running this path**: predictions get brighter and lose the grid-frequency ripple. Whether any published map came from here rather than from `optimized_inference` is not something I can tell from outside, so whether anything needs re-running is your call.

---

Found while reproducing a published prediction with `scrollprize/timesformer_scroll5_july_retreat` for the causal experiment in #1372. The result there was run under both this convention and a coherent Hann window and holds either way (91.2% against 92.8%), so nothing in that thread depends on this fix.

Related: #1371 is the same class of defect — a preprocessing constant that differs between two paths with no test comparing them.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
