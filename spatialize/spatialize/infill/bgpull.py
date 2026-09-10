"""Background pull: copy hole pixels from the far side of the hole along the epipolar (row)
direction, mirrored so the seam is continuous, then feathered. The standard DIBR fallback."""
from __future__ import annotations

import cv2
import numpy as np

from . import Backend, FillContext


def bgpull_fill(rgb: np.ndarray, hole: np.ndarray, geom, feather: float = 1.5) -> np.ndarray:
    h, w = hole.shape
    if not hole.any():
        return rgb
    out = rgb.copy()
    ys, xs = np.nonzero(hole)
    bg = geom.bg_idx[ys, xs]
    dist = np.abs(xs - bg)                      # 1 = adjacent to the flank
    step = np.where(geom.bg_is_right[ys, xs], 1, -1)
    src = bg + step * (dist - 1)                # mirror about the flank: x_bg, x_bg+1, ...
    src = np.clip(src, 0, w - 1)
    # if the mirrored source itself runs into another hole or off the frame, fall back to the flank
    bad = hole[ys, src]
    src = np.where(bad, bg, src)
    out[ys, xs] = rgb[ys, src]
    if feather > 0:
        # soften the copied texture inside the hole (mirror symmetry is very visible otherwise)
        blurred = cv2.GaussianBlur(out, (0, 0), feather)
        # feather weight: 1 deep inside the hole, 0 at the boundary
        dt = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
        a = np.clip(dt / (feather * 2.0), 0, 1)[..., None]
        mixed = out.astype(np.float32) * (1 - a) + blurred.astype(np.float32) * a
        out = np.where(hole[..., None], np.clip(mixed + 0.5, 0, 255).astype(np.uint8), out)
    return out


class BgPull(Backend):
    name = "bgpull"

    def fill(self, warped, hole, depth, original, ctx: FillContext):
        return bgpull_fill(warped, hole, ctx.geom)
