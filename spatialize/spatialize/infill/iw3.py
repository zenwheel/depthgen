"""Optional reference backend: nunif/iw3's own stereo synthesis, fed with *our* depth map.

iw3 accepts a pre-computed depth via its export/import format: a directory holding
`iw3_export.yml`, `rgb/00000.png` and a 16-bit `depth/00000.png` (large = near). We write our
conditioned depthgen map there with the mapper disabled, so iw3 skips its depth model and only
runs its own warping + inpainting ("mlbw" / "row_flow" methods). The whole eye comes from iw3,
so this backend replaces the frame instead of filling our holes (whole_frame = True).
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from . import HERE, Backend, FillContext

NUNIF_DIR = os.environ.get("NUNIF_DIR", os.path.join(HERE, "vendor", "nunif"))


class IW3(Backend):
    name = "iw3"
    whole_frame = True

    def __init__(self, method: str = "mlbw_l2"):
        self.method = method

    def available(self) -> bool:
        return os.path.isdir(os.path.join(NUNIF_DIR, "iw3"))

    def synthesize_pair(self, original: np.ndarray, depth: np.ndarray, parallax_px: float, zero_plane: float,
                        log=None) -> tuple[np.ndarray, np.ndarray]:
        import yaml
        from PIL import Image

        h, w = depth.shape
        tmp = tempfile.mkdtemp(prefix="spatialize_iw3_")
        try:
            os.makedirs(os.path.join(tmp, "in", "rgb"))
            os.makedirs(os.path.join(tmp, "in", "depth"))
            Image.fromarray(original).save(os.path.join(tmp, "in", "rgb", "00000.png"))
            d16 = np.clip(depth / 255.0 * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
            Image.fromarray(d16).save(os.path.join(tmp, "in", "depth", "00000.png"))
            with open(os.path.join(tmp, "in", "iw3_export.yml"), "w") as f:
                yaml.safe_dump({"type": "images", "basename": "spatialize", "rgb_dir": "rgb", "depth_dir": "depth",
                                "mapper": "none", "skip_mapper": True, "skip_edge_dilation": False,
                                "user_data": {"source": "depthgen"}}, f)
            divergence = parallax_px / w * 100.0
            # iw3 bounds the convergence plane to the depth range; a zero plane outside 0..255 (normal for a
            # parallel-camera baseline) is emulated by clamping here and translating the eyes afterwards
            convergence = float(np.clip(zero_plane / 255.0, 0.0, 1.0))
            residual_px = parallax_px * (convergence * 255.0 - zero_plane) / 255.0  # our shift minus iw3's
            # full side-by-side is iw3's default output; --depth-model NULL keeps it from loading a depth
            # network since the export config supplies the depth
            cmd = [sys.executable, "-m", "iw3", "-i", os.path.join(tmp, "in", "iw3_export.yml"), "-o", os.path.join(tmp, "out"),
                   "--depth-model", "NULL", "--format", "png", "--yes",
                   "--divergence", f"{divergence:.3f}", "--convergence", f"{convergence:.3f}",
                   "--method", self.method]
            env = dict(os.environ, PYTHONPATH=NUNIF_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
            r = subprocess.run(cmd, cwd=NUNIF_DIR, capture_output=True, text=True, env=env)
            if r.returncode != 0:
                # the default device is GPU 0 (MPS on a Mac); retry on the CPU before giving up
                r = subprocess.run(cmd + ["--gpu", "-1"], cwd=NUNIF_DIR, capture_output=True, text=True, env=env)
            if r.returncode != 0:
                err = "\n".join(l for l in r.stderr.strip().splitlines() if "objc[" not in l)
                raise RuntimeError(f"iw3 failed ({r.returncode}): {err[-500:]}")
            outs = sorted(glob.glob(os.path.join(tmp, "out", "**", "*.*"), recursive=True))
            outs = [p for p in outs if p.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not outs:
                raise RuntimeError("iw3 produced no image")
            sbs = np.asarray(Image.open(outs[0]).convert("RGB"))
            if sbs.shape[1] != 2 * w or sbs.shape[0] != h:
                sbs = cv2.resize(sbs, (2 * w, h), interpolation=cv2.INTER_LINEAR)
            left, right = sbs[:, :w].copy(), sbs[:, w:].copy()
            if abs(residual_px) > 0.05:
                # per eye the shift is s/2, so iw3's right eye sits residual/2 px too far right and its left eye
                # residual/2 px too far left; translate each back
                left = self._translate(left, residual_px / 2.0)
                right = self._translate(right, -residual_px / 2.0)
            return left, right

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _translate(img: np.ndarray, dx: float) -> np.ndarray:
        m = np.float32([[1, 0, dx], [0, 1, 0]])
        return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def fill(self, warped, hole, depth, original, ctx: FillContext):
        cache = ctx.extra.setdefault("iw3_pair", {})
        # iw3 always splits the divergence over both eyes; in right-only mode our right eye carries the
        # whole shift, so ask iw3 for twice the range and use only its right eye
        parallax = ctx.parallax_px * (2.0 if ctx.extra.get("eyes") == "right" else 1.0)
        key = (id(original), round(parallax, 2), round(ctx.zero_plane, 2))
        if key not in cache:
            cache.clear()
            cache[key] = self.synthesize_pair(original, ctx.depth_src, parallax, ctx.zero_plane)
        left, right = cache[key]
        return left if ctx.extra.get("eye") == "left" else right
