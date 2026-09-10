"""Hole geometry shared by the infill backends and the selector."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HoleGeometry:
    hole: np.ndarray        # HxW bool (what has to be filled)
    left_idx: np.ndarray    # nearest valid column to the left (-1 = none)
    right_idx: np.ndarray   # nearest valid column to the right (W = none)
    bg_idx: np.ndarray      # per pixel, the column of the background-side flank (valid only on holes)
    bg_is_right: np.ndarray # bool per pixel
    width: np.ndarray       # per hole pixel, the run width in px
    fg_depth: np.ndarray    # depth of the near flank (per pixel)
    bg_depth: np.ndarray    # depth of the far flank (per pixel)


def analyze(hole: np.ndarray, depth: np.ndarray, left_idx: np.ndarray, right_idx: np.ndarray) -> HoleGeometry:
    h, w = hole.shape
    li = np.clip(left_idx, 0, w - 1)
    ri = np.clip(right_idx, 0, w - 1)
    rows = np.arange(h)[:, None]
    dl = depth[rows, li]
    dr = depth[rows, ri]
    has_l = left_idx >= 0
    has_r = right_idx < w
    # background = the far flank (smaller inverse depth); with one flank only, use it
    bg_is_right = np.where(has_l & has_r, dr < dl, has_r)
    bg_idx = np.where(bg_is_right, ri, li).astype(np.int32)
    bg_depth = np.where(bg_is_right, dr, dl)
    fg_depth = np.where(bg_is_right, dl, dr)
    width = (right_idx - left_idx - 1).astype(np.int32)
    return HoleGeometry(hole=hole, left_idx=left_idx, right_idx=right_idx, bg_idx=bg_idx,
                        bg_is_right=bg_is_right, width=width, fg_depth=fg_depth, bg_depth=bg_depth)


def bg_strip(geom: HoleGeometry, n: int = 8) -> np.ndarray:
    """Mask of the `n` background pixels immediately beyond each hole's far flank."""
    h, w = geom.hole.shape
    strip = np.zeros((h, w), bool)
    ys, xs = np.nonzero(geom.hole)
    if len(ys) == 0:
        return strip
    bg = geom.bg_idx[ys, xs]
    step = np.where(geom.bg_is_right[ys, xs], 1, -1)
    for k in range(n):
        xk = np.clip(bg + k * step, 0, w - 1)
        strip[ys, xk] = True
    strip &= ~geom.hole
    return strip


def clusters(hole: np.ndarray, margin: int = 32, join: int = 16, max_clusters: int = 64):
    """Bounding boxes (x0, y0, x1, y1) of hole clusters, holes within `join` px merged, grown by `margin`.
    Returned largest-area first."""
    h, w = hole.shape
    if not hole.any():
        return []
    m = hole.astype(np.uint8)
    if join > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * join + 1, 2 * join + 1))
        m = cv2.dilate(m, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        sub = hole[y:y + bh, x:x + bw]
        hole_area = int(sub.sum())
        if hole_area == 0:
            continue
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
        boxes.append(((x0, y0, x1, y1), hole_area))
    boxes.sort(key=lambda b: -b[1])
    return boxes[:max_clusters]
