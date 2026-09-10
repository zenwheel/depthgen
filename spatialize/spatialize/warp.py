"""Forward warping with z-buffering and sub-pixel splatting; hole/border bookkeeping; reprojection.

Convention: `shift` is the signed horizontal displacement in px applied to each *source* pixel
for the eye being rendered (x_target = x_source + shift). Nearer content has larger |shift|.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class Warped:
    rgb: np.ndarray          # HxWx3 uint8, holes are 0
    hole: np.ndarray         # HxW bool: disocclusion holes (border strips excluded)
    border: np.ndarray       # HxW bool: uncovered strips at the frame edges (content slid away)
    depth: np.ndarray        # HxW float32: warped depth (0..255), -1 where nothing landed
    weight: np.ndarray       # HxW float32: splat weight sum
    shift: np.ndarray        # HxW float32: the source-space shift used
    src_visible: np.ndarray  # HxW bool: source pixels that survive z-buffering
    left_idx: np.ndarray     # HxW int32: for every target pixel, nearest non-hole column to the left (-1 none)
    right_idx: np.ndarray    # HxW int32: nearest non-hole column to the right (W = none)
    cracks_filled: int = 0


def _nearest_valid_indices(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel, the column index of the nearest valid pixel at or to the left / right."""
    h, w = valid.shape
    cols = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))
    left = np.maximum.accumulate(np.where(valid, cols, -1), axis=1)
    right = np.minimum.accumulate(np.where(valid, cols, w)[:, ::-1], axis=1)[:, ::-1]
    return left.astype(np.int32), right.astype(np.int32)


def forward_warp(rgb: np.ndarray, depth: np.ndarray, shift: np.ndarray, device: str = "cpu",
                 same_surface_tol: float | None = None, island_px: int = 2) -> Warped:
    """Splat every source pixel to x + shift with bilinear (2-tap) footprints; nearest depth wins.

    `same_surface_tol` (depth units, 0..255) decides which contributions count as the winning
    surface: anything within tol of the z-buffer max blends in; the rest is occluded.
    """
    h, w = depth.shape
    n = h * w
    dev = torch.device(device)
    if same_surface_tol is None:
        # depth difference worth ~1 px of shift, at least 3 levels
        span = float(np.abs(shift).max()) + 1e-6
        same_surface_tol = max(3.0, 255.0 / span)

    dep = torch.from_numpy(np.ascontiguousarray(depth)).to(dev).reshape(-1)
    col = torch.from_numpy(np.array(rgb, copy=True)).to(dev).reshape(-1, 3).to(torch.float32)
    xs = torch.arange(w, device=dev, dtype=torch.float32).repeat(h) + torch.from_numpy(
        np.ascontiguousarray(shift)).to(dev).reshape(-1)
    rows = torch.arange(h, device=dev, dtype=torch.int64).repeat_interleave(w)
    x0 = torch.floor(xs)
    f = xs - x0
    x0 = x0.to(torch.int64)

    taps = []
    for xt, wt in ((x0, 1.0 - f), (x0 + 1, f)):
        ok = (xt >= 0) & (xt < w) & (wt > 1e-3)
        idx = (rows * w + xt)[ok]
        taps.append((idx, wt[ok], ok))

    zbuf = torch.full((n,), -1.0, device=dev, dtype=torch.float32)
    for idx, wt, ok in taps:
        zbuf.scatter_reduce_(0, idx, dep[ok], reduce="amax", include_self=True)

    wsum = torch.zeros(n, device=dev, dtype=torch.float32)
    acc = torch.zeros(n, 3, device=dev, dtype=torch.float32)
    visible = torch.zeros(n, device=dev, dtype=torch.bool)
    for idx, wt, ok in taps:
        accept = dep[ok] >= zbuf[idx] - same_surface_tol
        idx_a = idx[accept]
        wt_a = wt[accept]
        wsum.index_add_(0, idx_a, wt_a)
        acc.index_add_(0, idx_a, col[ok][accept] * wt_a[:, None])
        src_ok = torch.nonzero(ok, as_tuple=True)[0][accept]
        visible[src_ok] = True

    covered = wsum > 1e-3
    out = torch.where(covered[:, None], acc / wsum.clamp_min(1e-3)[:, None], torch.zeros_like(acc))
    out = out.round().clamp(0, 255).to(torch.uint8).reshape(h, w, 3).cpu().numpy()
    wsum_np = wsum.reshape(h, w).cpu().numpy()
    zbuf_np = zbuf.reshape(h, w).cpu().numpy()
    hole = ~covered.reshape(h, w).cpu().numpy()
    visible_np = visible.reshape(h, w).cpu().numpy()

    # Edge debris: source pixels with in-between depth at an object boundary land alone in the middle
    # of the disocclusion as 1-2 px streaks. Absorb covered islands narrower than `island_px` that
    # sit between hole pixels on the same row (a horizontal closing of the hole mask).
    if island_px > 0:
        k = np.ones((1, 2 * island_px + 1), np.uint8)
        closed = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        debris = closed & ~hole
        # ...and the fractional tap of a foreground edge pixel that lands alone next to a hole (a 1 px
        # line of foreground colour on the far side of the disocclusion)
        near_hole = cv2.dilate(hole.astype(np.uint8), np.ones((1, 3), np.uint8)).astype(bool)
        debris |= near_hole & ~hole & (wsum_np < 0.35)
        if debris.any():
            hole |= debris
            out[debris] = 0
            zbuf_np[debris] = -1.0
            wsum_np[debris] = 0.0

    # Border strips: hole runs touching the frame edge (content slid away, nothing behind it).
    border = _border_runs(hole)
    inner = hole & ~border

    # Cracks: gaps whose two flanks belong to the same surface (a stretched slanted plane). They are
    # a sampling artifact, not a disocclusion, so close them by interpolation right here.
    left_idx, right_idx = _nearest_valid_indices(~hole)
    cracks = np.zeros_like(inner)
    if inner.any():
        li = np.clip(left_idx, 0, w - 1)
        ri = np.clip(right_idx, 0, w - 1)
        rows_np = np.arange(h)[:, None]
        dl = zbuf_np[rows_np, li]
        dr = zbuf_np[rows_np, ri]
        cracks = inner & (np.abs(dl - dr) <= same_surface_tol) & (right_idx - left_idx - 1 <= 2)
        if cracks.any():
            t = (np.arange(w)[None, :] - li) / np.maximum(ri - li, 1)
            interp = (out[rows_np, li].astype(np.float32) * (1 - t[..., None])
                      + out[rows_np, ri].astype(np.float32) * t[..., None])
            out[cracks] = np.clip(interp[cracks] + 0.5, 0, 255).astype(np.uint8)
            zbuf_np[cracks] = np.maximum(dl, dr)[cracks]
            inner = inner & ~cracks
            hole = inner | border
            left_idx, right_idx = _nearest_valid_indices(~hole)

    return Warped(rgb=out, hole=inner, border=border, depth=zbuf_np, weight=wsum_np, shift=shift,
                  src_visible=visible_np, left_idx=left_idx, right_idx=right_idx,
                  cracks_filled=int(cracks.sum()))


def _border_runs(hole: np.ndarray) -> np.ndarray:
    """Hole pixels connected to the left or right frame edge along their row."""
    h, w = hole.shape
    # from the left: run of holes starting at column 0
    left_run = np.cumprod(hole, axis=1).astype(bool)
    right_run = np.cumprod(hole[:, ::-1], axis=1)[:, ::-1].astype(bool)
    return left_run | right_run


def border_widths(border: np.ndarray) -> tuple[int, int]:
    """(left_px, right_px) widths of the uncovered border strips (max over rows)."""
    h, w = border.shape
    left = int(np.cumprod(border, axis=1).sum(axis=1).max()) if border.any() else 0
    right = int(np.cumprod(border[:, ::-1], axis=1).sum(axis=1).max()) if border.any() else 0
    return left, right


def reproject_error(original: np.ndarray, synthesized: np.ndarray, shift: np.ndarray,
                    src_visible: np.ndarray, exclude_target: np.ndarray | None = None) -> float:
    """Warp the synthesized eye back to the source viewpoint (inverse mapping with the source's own
    shift field) and measure the mean absolute error against the original over visible pixels.
    Pixels whose target sample lands on `exclude_target` (e.g. filled holes) are ignored."""
    h, w = shift.shape
    map_x = (np.arange(w, dtype=np.float32)[None, :] + shift).astype(np.float32)
    map_y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w)).astype(np.float32)
    back = cv2.remap(synthesized, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = src_visible & (map_x >= 0) & (map_x <= w - 1)
    if exclude_target is not None:
        landed = cv2.remap(exclude_target.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=1)
        valid &= landed == 0
    if not valid.any():
        return float("nan")
    diff = np.abs(back.astype(np.float32) - original.astype(np.float32)).mean(axis=2)
    return float(diff[valid].mean())
