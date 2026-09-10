"""cv2.inpaint (Telea / Navier-Stokes) on crops around each hole cluster."""
from __future__ import annotations

import cv2
import numpy as np

from . import Backend, FillContext
from ..holes import clusters


def cv_inpaint(rgb: np.ndarray, hole: np.ndarray, flag: int, radius: int = 3) -> np.ndarray:
    if not hole.any():
        return rgb
    out = rgb.copy()
    for (x0, y0, x1, y1), _ in clusters(hole, margin=24, join=8):
        crop = out[y0:y1, x0:x1]
        m = hole[y0:y1, x0:x1].astype(np.uint8) * 255
        out[y0:y1, x0:x1] = cv2.inpaint(crop, m, radius, flag)
    return out


class Telea(Backend):
    name = "opencv"

    def fill(self, warped, hole, depth, original, ctx: FillContext):
        return cv_inpaint(warped, hole, cv2.INPAINT_TELEA)


class NavierStokes(Backend):
    name = "opencv-ns"

    def fill(self, warped, hole, depth, original, ctx: FillContext):
        return cv_inpaint(warped, hole, cv2.INPAINT_NS)
