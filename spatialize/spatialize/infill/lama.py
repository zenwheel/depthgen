"""LaMa (big-lama, TorchScript) inpainting on crops around each hole cluster, on MPS when available."""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from . import WEIGHTS_DIR, Backend, FillContext
from ..holes import clusters

WEIGHTS = os.path.join(WEIGHTS_DIR, "big-lama.pt")


class LaMa(Backend):
    name = "lama"

    def __init__(self, max_side: int = 768, min_side: int = 256, margin: int = 64):
        self.model = None
        self.device = None
        self.max_side = max_side
        self.min_side = min_side
        self.margin = margin
        self.load_seconds = 0.0

    def available(self) -> bool:
        return os.path.isfile(WEIGHTS)

    def _load(self, device: str):
        if self.model is not None:
            return
        import torch

        t0 = time.time()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch.jit.load warns on Python 3.14; the model loads fine
            self.model = torch.jit.load(WEIGHTS, map_location="cpu").eval()
        self.device = device
        try:
            self.model.to(device)
            # probe once: FFT ops in LaMa's FFC blocks may be unsupported on some backends
            with torch.inference_mode():
                self.model(torch.zeros(1, 3, 256, 256, device=device), torch.zeros(1, 1, 256, 256, device=device))
        except Exception:
            self.device = "cpu"
            self.model.to("cpu")
        self.load_seconds = time.time() - t0

    def _inpaint_crop(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        h, w = mask.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        img = np.pad(crop, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        m = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
        it = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255.0).to(self.device)
        mt = torch.from_numpy(m.astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            out = self.model(it, mt)[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        return out[:h, :w]

    def fill(self, warped, hole, depth, original, ctx: FillContext):
        if not hole.any():
            return warped
        self._load(ctx.device)
        H, W = hole.shape
        out = warped.copy()
        # the network also sees the boundary blend pixels as unknown
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_d = cv2.dilate(hole.astype(np.uint8), k5)
        for (x0, y0, x1, y1), _ in clusters(hole, margin=self.margin, join=24):
            for (tx0, ty0, tx1, ty1) in self._tiles(x0, y0, x1, y1, W, H):
                m = mask_d[ty0:ty1, tx0:tx1]
                if not m.any():
                    continue
                crop = out[ty0:ty1, tx0:tx1]
                res = self._inpaint_tile(crop, m)
                # paste only the hole (plus a sub-pixel feather) back; keep the warp's real pixels elsewhere
                hm = hole[ty0:ty1, tx0:tx1].astype(np.float32)
                a = np.maximum(cv2.GaussianBlur(hm, (0, 0), 0.7), hm)[..., None]
                out[ty0:ty1, tx0:tx1] = np.clip(crop * (1 - a) + res * a + 0.5, 0, 255).astype(np.uint8)
        return out

    def _tiles(self, x0, y0, x1, y1, W, H):
        """Cover a cluster box with tiles of at most max_side (native resolution), each at least
        min_side so the network gets context, overlapping by the margin."""
        step = self.max_side - 2 * self.margin
        xs = list(range(x0, max(x1 - self.margin, x0 + 1), step))
        ys = list(range(y0, max(y1 - self.margin, y0 + 1), step))
        for ty in ys:
            for tx in xs:
                tw = min(self.max_side, max(self.min_side, x1 - tx))
                th = min(self.max_side, max(self.min_side, y1 - ty))
                # keep the tile inside the frame by sliding it, not shrinking it
                tx0, ty0 = max(0, min(tx, W - tw)), max(0, min(ty, H - th))
                yield tx0, ty0, min(W, tx0 + tw), min(H, ty0 + th)

    def _inpaint_tile(self, crop: np.ndarray, m: np.ndarray) -> np.ndarray:
        ch, cw = m.shape
        scale = min(1.0, self.max_side / max(ch, cw))
        if scale < 1.0:
            sw, sh = max(8, int(round(cw * scale))), max(8, int(round(ch * scale)))
            crop_s = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
            m_s = (cv2.resize(m.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.05).astype(np.uint8)
            return cv2.resize(self._inpaint_crop(crop_s, m_s), (cw, ch), interpolation=cv2.INTER_CUBIC)
        return self._inpaint_crop(crop, m)
