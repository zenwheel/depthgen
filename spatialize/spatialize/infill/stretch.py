"""Background stretch: the depth-displaced-mesh look (generate_usdz.py / fake3d.js).

Rendering the image as a height-field mesh means the triangle that straddles a depth edge gets
stretched across the disocclusion, texturing it with a linear ramp between the two edge pixels.
Filling every hole run by interpolating its two flanks along the row is exactly that.
"""
from __future__ import annotations

import numpy as np

from . import Backend, FillContext


def stretch_fill(rgb: np.ndarray, hole: np.ndarray, left_idx: np.ndarray, right_idx: np.ndarray) -> np.ndarray:
    h, w = hole.shape
    if not hole.any():
        return rgb
    out = rgb.copy()
    ys, xs = np.nonzero(hole)
    li = left_idx[ys, xs]
    ri = right_idx[ys, xs]
    has_l = li >= 0
    has_r = ri < w
    lc = rgb[ys, np.clip(li, 0, w - 1)].astype(np.float32)
    rc = rgb[ys, np.clip(ri, 0, w - 1)].astype(np.float32)
    t = np.where(has_l & has_r, (xs - li) / np.maximum(ri - li, 1), np.where(has_r, 1.0, 0.0)).astype(np.float32)
    out[ys, xs] = np.clip(lc * (1 - t[:, None]) + rc * t[:, None] + 0.5, 0, 255).astype(np.uint8)
    return out


class Stretch(Backend):
    name = "stretch"

    def fill(self, warped, hole, depth, original, ctx: FillContext):
        return stretch_fill(warped, hole, ctx.geom.left_idx, ctx.geom.right_idx)
