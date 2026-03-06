# Video Latent Mask Node

## Overview

The `VideoLatentMask` node is a utility designed for ComfyUI video workflows. it generates a 5D latent mask tensor `[B, C, F, H, W]` where the initial frames of a segment are "masked" (set to 0.0) and the remaining frames are "unmasked" (set to 1.0).

This node is particularly useful in **temporal video segment transitions** or **video-to-video** workflows where you want to preserve the beginning of a segment (e.g., to maintain continuity with a previous segment's end) while allowing the model to freely generate or modify the subsequent frames.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | INT | 1 | Number of latent batches. |
| `channels` | INT | 16 | Number of latent channels (typically 16 for models like CogVideoX or Wan). |
| `frames` | INT | 16 | Total number of frames in the latent segment. |
| `height` | INT | 64 | Height of the latent (in latent units, e.g., pixel_height / 8). |
| `width` | INT | 64 | Width of the latent (in latent units, e.g., pixel_width / 8). |
| `black_frames` | INT | 4 | Number of initial frames to set to 0.0 (masked). |

## Logic

The node creates a tensor of ones with the specified dimensions and then slices the frame dimension (index 2) to set the first `black_frames` to zero:

```python
mask = torch.ones((batch_size, channels, frames, height, width))
mask[:, :, :black_frames, :, :] = 0.0
```

## Workflow Integration

1. **Temporal Continuity**: Use this node in conjunction with `VideoSegmentPrepare` and `LatentPrependCache`.
2. **Masked Sampling**: Pass the generated mask to a sampler that supports latent masks (like `InpaintModelConditioning` or custom samplers that accept a `mask` in the latent dictionary).
3. **Preservation**: By masking the first few frames, you ensure the model focuses its denoising energy on the later frames, keeping the starting frames closer to the original or cached reference.

## Example Usage

If you have a 16-frame segment and set `black_frames` to 4:
- Frames 0, 1, 2, 3 will be 0.0 (preserved/masked).
- Frames 4 through 15 will be 1.0 (modified/unmasked).

---
**Status**: ✅ Implementation Complete
**Date**: March 6, 2026
