"""Depth map acquisition and conditioning, and the depth -> disparity mapping.

depthgen writes normalized *inverse* depth (255 = nearest, 0 = farthest, stretched to the
full range per image). Inverse depth is linear in disparity, so the horizontal shift is

    shift_px = parallax_px * (value/255 - zero_plane/255)

and never anything like 1/value.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .imio import load_depth8, to_gray_f32


# ----------------------------------------------------------------------------- acquisition

def default_depth_path(image_path: str) -> str:
    stem, _ = os.path.splitext(image_path)
    return stem + "-depth.png"


def find_or_make_depth(image_path: str, explicit: str | None, log) -> tuple[str, float]:
    """Return (depth_png_path, seconds spent running depthgen)."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(explicit)
        return explicit, 0.0
    cand = default_depth_path(image_path)
    if os.path.isfile(cand):
        return cand, 0.0
    exe = shutil.which("depthgen")
    if not exe:
        raise RuntimeError(f"no depth map at {cand} and `depthgen` is not on PATH")
    t0 = time.time()
    log(f"  running depthgen on {os.path.basename(image_path)}")
    r = subprocess.run([exe, image_path], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(cand):
        raise RuntimeError(f"depthgen failed ({r.returncode}): {r.stderr.strip()[-300:]}")
    return cand, time.time() - t0


# ----------------------------------------------------------------------------- conditioning

def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter (gray guide), box-filter implementation. float32 in/out."""
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(a):
        return cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    mean_i = box(guide)
    mean_p = box(src)
    corr_ip = box(guide * src)
    corr_ii = box(guide * guide)
    var_i = corr_ii - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


@dataclass
class DepthConditioning:
    blur: float = 0.0            # sigma (px) of a bilateral-style smoothing to kill 8-bit banding
    edge_refine: bool = True     # guided filter against the color image
    guided_radius: int = 8
    guided_eps: float = 400.0    # on a 0..255 guide; larger = smoother, smaller = follows image edges harder
    fg_erode: int = 0            # erode near regions by N px (kills the halo mono depth leaves around edges)
    edge_sharpen: int = 3        # radius px: turn depth ramps at strong edges into steps (0 = off)
    edge_sharpen_min: float = 12.0  # only where the local depth range exceeds this many levels


def condition_depth(depth8: np.ndarray, rgb: np.ndarray, cfg: DepthConditioning) -> np.ndarray:
    """Return a float32 HxW map in 0..255 (same convention as the input) ready for warping."""
    d = depth8.astype(np.float32)
    if cfg.blur > 0:
        # bilateral in depth space: smooths quantization steps but keeps real depth edges
        d = cv2.bilateralFilter(d, d=0, sigmaColor=6.0, sigmaSpace=float(cfg.blur))
    if cfg.edge_refine:
        # Work at reduced resolution when huge: the filter's cost is ~10 box filters per pass.
        h, w = d.shape
        scale = min(1.0, 2048.0 / max(h, w))
        guide = to_gray_f32(rgb)
        if scale < 1.0:
            small = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            g_s = cv2.resize(guide, small, interpolation=cv2.INTER_AREA)
            d_s = cv2.resize(d, small, interpolation=cv2.INTER_AREA)
            r = max(1, int(round(cfg.guided_radius * scale)))
            out_s = guided_filter(g_s, d_s, r, cfg.guided_eps)
            # keep the low-res refinement's edge placement but add back full-res detail
            d = cv2.resize(out_s, (w, h), interpolation=cv2.INTER_LINEAR) + (
                d - cv2.resize(d_s, (w, h), interpolation=cv2.INTER_LINEAR))
        else:
            d = guided_filter(guide, d, cfg.guided_radius, cfg.guided_eps)
    if cfg.edge_sharpen > 0:
        # Depth Pro output is resampled from 1536 px, so every depth edge is a 3-6 px ramp. Warping a ramp
        # stretches the pixels across it (rubber-sheet look) instead of opening a hole the infill can
        # handle. A morphological toggle (snap to local min or max where the range is large) makes steps.
        r = cfg.edge_sharpen
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        hi = cv2.dilate(d, k)
        lo = cv2.erode(d, k)
        rng = hi - lo
        snap = np.where(d - lo > hi - d, hi, lo)
        d = np.where(rng > cfg.edge_sharpen_min, snap, d).astype(np.float32)
    if cfg.fg_erode > 0:
        # a grayscale erosion of inverse depth shrinks near regions into the background side
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.fg_erode + 1, 2 * cfg.fg_erode + 1))
        d = cv2.erode(d, k)
    return np.clip(d, 0.0, 255.0).astype(np.float32)


# ----------------------------------------------------------------------------- parallax / convergence

def parse_parallax(spec: str, width: int) -> float:
    """'2.0' (percent of width) or '40px' -> pixels."""
    s = spec.strip().lower()
    if s.endswith("px"):
        return float(s[:-2])
    if s.endswith("%"):
        s = s[:-1]
    return float(s) / 100.0 * width


def resolve_convergence(mode: str, depth: np.ndarray, parallax_px: float, width: int,
                        far_limit_pct: float = 1.2, near_frac: float = 0.25) -> tuple[float, str]:
    """Zero-parallax plane as a depth value (0..255).

    auto:   the nearest `near_frac` of pixels come forward of the screen, unless that would push
            far content beyond `far_limit_pct` of the width behind it (uncrossed disparity), in
            which case the plane moves back until it does not.
    near:   everything behind the screen (plane at the nearest value)
    far:    everything in front (plane at the farthest value)
    median: median depth
    NN%:    percentile of depth; NN: literal value
    """
    m = mode.strip().lower()
    lo, hi = float(depth.min()), float(depth.max())
    if m == "near":
        return hi, "near"
    if m == "far":
        return lo, "far"
    if m == "median":
        return float(np.median(depth)), "median"
    if m.endswith("%"):
        return float(np.percentile(depth, float(m[:-1]))), f"percentile {m}"
    if m != "auto":
        return float(m), "literal"  # may sit outside 0..255 (everything in front of / behind the screen)
    z = float(np.percentile(depth, 100.0 * (1.0 - near_frac)))
    # far divergence in px = parallax_px * (z - lo)/255 must stay under the comfort limit
    limit_px = far_limit_pct / 100.0 * width
    if parallax_px > 0:
        max_z = lo + 255.0 * limit_px / parallax_px
        if z > max_z:
            return float(max(lo, max_z)), f"auto (far-limit {far_limit_pct}% clamped from {z:.0f})"
    return z, "auto (nearest 25% forward)"


def shift_field(depth: np.ndarray, parallax_px: float, zero_plane: float) -> np.ndarray:
    """Per-pixel total disparity in px (positive = nearer than the screen = crossed)."""
    return ((depth - zero_plane) * (parallax_px / 255.0)).astype(np.float32)
