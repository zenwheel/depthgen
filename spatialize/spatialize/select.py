"""Automatic infill selection: hole statistics, cheap rules, then reference-free scoring.

Scores (all lower = better):
  photo  mean abs error (gray levels) when the synthesized eye is warped back to the source view
         with the same depth and compared to the original outside the holes. Same for every
         backend that only touches holes; catches whole-frame backends and boundary bleed.
  seam   gradient step across the background-side hole boundary after filling, minus the typical
         horizontal gradient of the adjacent background (so textured backgrounds are not penalized).
  lpips  perceptual distance between each filled hole patch and the neighbouring background
         context patch of the same size (does the fill look like the background around it?).
  total  weighted sum; weights tuned on real stereo pairs with eval/eval_pairs.py.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .holes import HoleGeometry, bg_strip, clusters
from .imio import to_gray_f32
from .infill import WEIGHTS_DIR
from .warp import reproject_error

# --- tunables -------------------------------------------------------------------------------
W_PHOTO = 1.0 / 5.0      # photo error in gray levels -> ~0..1
W_SEAM = 1.0 / 20.0      # seam excess in gray levels -> ~0..1
W_LPIPS = 1.0
CRACK_MAX_WIDTH = 2      # px: holes this narrow are always stretched
SMOOTH_BG_GRAD = 4.0     # mean |grad| (gray levels/px) below which the background counts as smooth
WIDE_HOLE = 24           # px (p95 run width): only holes this wide are worth trying the learned inpainter on
LPIPS_MAX_WINDOWS = 24
LPIPS_PATCH = 128


@dataclass
class HoleStats:
    area_px: int
    area_frac: float
    max_width: int
    p95_width: float
    mean_width: float
    n_clusters: int
    bg_grad: float           # mean gradient magnitude in the adjacent background strip
    bg_std: float            # mean local (5x5) std there
    border_px: tuple[int, int] = (0, 0)

    def as_dict(self):
        return {"area_px": self.area_px, "area_frac": round(self.area_frac, 6), "max_width": self.max_width,
                "p95_width": round(self.p95_width, 1), "mean_width": round(self.mean_width, 2),
                "n_clusters": self.n_clusters, "bg_grad": round(self.bg_grad, 2), "bg_std": round(self.bg_std, 2),
                "border_px": list(self.border_px)}


def hole_stats(geom: HoleGeometry, rgb: np.ndarray, border_px=(0, 0)) -> HoleStats:
    hole = geom.hole
    h, w = hole.shape
    area = int(hole.sum())
    if area == 0:
        return HoleStats(0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, border_px)
    widths = geom.width[hole]
    strip = bg_strip(geom, n=8)
    gray = to_gray_f32(rgb)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.sqrt(gx * gx + gy * gy)
    mean = cv2.blur(gray, (5, 5))
    sq = cv2.blur(gray * gray, (5, 5))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    if strip.any():
        bg_grad, bg_std = float(grad[strip].mean()), float(std[strip].mean())
    else:
        bg_grad, bg_std = 0.0, 0.0
    return HoleStats(area, area / (h * w), int(widths.max()), float(np.percentile(widths, 95)),
                     float(widths.mean()), len(clusters(hole, margin=0, join=16, max_clusters=10000)),
                     bg_grad, bg_std, border_px)


def plan(stats: HoleStats, candidates: list[str], available: list[str]) -> tuple[str | None, list[str], str]:
    """Cheap rules. Returns (forced_choice or None, candidates to score, reason)."""
    cands = [c for c in candidates if c in available]
    if not cands:
        cands = ["stretch"]
    if stats.area_px == 0:
        return "stretch", ["stretch"], "no holes"
    if stats.max_width <= CRACK_MAX_WIDTH:
        return "stretch", ["stretch"], f"max hole width {stats.max_width}px <= {CRACK_MAX_WIDTH}"
    if len(cands) == 1:
        return cands[0], cands, "single candidate"
    smooth = stats.bg_grad < SMOOTH_BG_GRAD
    if smooth:
        cheap = [c for c in cands if c not in ("lama", "iw3")]
        if cheap:
            return None, cheap, f"smooth background (grad {stats.bg_grad:.1f} < {SMOOTH_BG_GRAD}); cheap backends only"
    if stats.p95_width >= WIDE_HOLE:
        return None, cands, f"textured background (grad {stats.bg_grad:.1f}), wide holes (p95 {stats.p95_width:.0f}px); score all"
    cheap = [c for c in cands if c not in ("lama", "iw3")] or cands
    return None, cheap, f"narrow holes (p95 {stats.p95_width:.0f}px) on textured background (grad {stats.bg_grad:.1f}); cheap backends only"


class Scorer:
    """Holds the LPIPS network (loaded once) and scores candidates."""

    def __init__(self, device: str = "cpu", use_lpips: bool = True):
        self.device = device
        self.lpips = None
        self.use_lpips = use_lpips
        self.load_seconds = 0.0

    def _load_lpips(self):
        if self.lpips is not None or not self.use_lpips:
            return
        import warnings

        os.environ.setdefault("TORCH_HOME", os.path.join(WEIGHTS_DIR, "torch"))
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import lpips
            import torch

            self.torch = torch
            self.lpips = lpips.LPIPS(net="alex", verbose=False).eval()
            try:
                self.lpips.to(self.device)
                self.lpips_device = self.device
            except Exception:
                self.lpips_device = "cpu"
        self.load_seconds = time.time() - t0

    def seam(self, filled: np.ndarray, geom: HoleGeometry, strip: np.ndarray) -> float:
        hole = geom.hole
        h, w = hole.shape
        ys, xs = np.nonzero(hole)
        if len(ys) == 0:
            return 0.0
        bg = geom.bg_idx[ys, xs]
        adj = np.abs(xs - bg) == 1
        if not adj.any():
            return 0.0
        f = filled.astype(np.float32)
        step = np.abs(f[ys[adj], xs[adj]] - f[ys[adj], bg[adj]]).mean(axis=1)
        # context: horizontal differences inside the background strip
        dx = np.abs(np.diff(f, axis=1)).mean(axis=2)
        ctx = dx[strip[:, 1:] & strip[:, :-1]]
        ctx_mean = float(ctx.mean()) if ctx.size else 0.0
        return float(max(0.0, step.mean() - ctx_mean))

    def perceptual(self, filled: np.ndarray, geom: HoleGeometry) -> float:
        """LPIPS between small windows centred on hole pixels and the same windows shifted onto the
        adjacent background. Outside the hole the window is replaced by the context pixels, so the
        distance only measures whether the *fill* looks like the background next to it."""
        self._load_lpips()
        if self.lpips is None:
            return 0.0
        hole = geom.hole
        h, w = hole.shape
        ys, xs = np.nonzero(hole)
        if len(ys) == 0:
            return 0.0
        rng = np.random.default_rng(0)
        pick = rng.choice(len(ys), size=min(LPIPS_MAX_WINDOWS, len(ys)), replace=False)
        torch = self.torch
        pa, pb = [], []
        for i in pick:
            y, x = int(ys[i]), int(xs[i])
            side = int(np.clip(4 * geom.width[y, x], 32, 160))
            half = side // 2
            x0, y0 = x - half, y - half
            if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
                continue
            dx = side if geom.bg_is_right[y, x] else -side
            if not (0 <= x0 + dx and x0 + dx + side <= w):
                continue
            b = filled[y0:y0 + side, x0 + dx:x0 + dx + side]
            hm = hole[y0:y0 + side, x0:x0 + side][..., None]
            # only the fill may differ from the context: everything outside the hole is taken from b
            a = np.where(hm, filled[y0:y0 + side, x0:x0 + side], b)
            pa.append(cv2.resize(a, (LPIPS_PATCH, LPIPS_PATCH), interpolation=cv2.INTER_LINEAR))
            pb.append(cv2.resize(b, (LPIPS_PATCH, LPIPS_PATCH), interpolation=cv2.INTER_LINEAR))
        if not pa:
            return 0.0
        ta = torch.from_numpy(np.stack(pa)).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.lpips_device)
        tb = torch.from_numpy(np.stack(pb)).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.lpips_device)
        with torch.inference_mode():
            d = self.lpips(ta, tb).flatten().cpu().numpy()
        return float(d.mean())

    def score(self, filled: np.ndarray, original: np.ndarray, shift: np.ndarray, src_visible: np.ndarray,
              geom: HoleGeometry, strip: np.ndarray, whole_frame: bool = False) -> dict:
        t0 = time.time()
        photo = reproject_error(original, filled, shift, src_visible, None if whole_frame else geom.hole)
        seam = self.seam(filled, geom, strip)
        lp = self.perceptual(filled, geom)
        total = W_PHOTO * photo + W_SEAM * seam + W_LPIPS * lp
        return {"photo": round(photo, 3), "seam": round(seam, 3), "lpips": round(lp, 4),
                "total": round(total, 4), "score_seconds": round(time.time() - t0, 3)}
