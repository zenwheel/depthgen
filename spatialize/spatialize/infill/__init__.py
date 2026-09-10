"""Infill backends behind one interface.

    fill(warped_rgb, hole_mask, depth, original, ctx) -> rgb

`ctx` is a FillContext carrying the hole geometry and warp bookkeeping every backend needs
(flank indices, shift field, device). Backends must only modify pixels inside `hole_mask`
(a small feather across the boundary is allowed) so the selector's scores stay comparable.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field

import numpy as np

from ..holes import HoleGeometry

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.environ.get("SPATIALIZE_WEIGHTS", os.path.join(HERE, "weights"))


@dataclass
class FillContext:
    geom: HoleGeometry
    shift: np.ndarray            # source-space shift field for this eye
    depth_src: np.ndarray        # conditioned source depth (0..255)
    parallax_px: float
    zero_plane: float
    device: str = "cpu"
    extra: dict = field(default_factory=dict)


class Backend:
    name = "base"
    whole_frame = False          # True: synthesizes the eye itself instead of filling our warp

    def available(self) -> bool:
        return True

    def fill(self, warped: np.ndarray, hole: np.ndarray, depth: np.ndarray, original: np.ndarray,
             ctx: FillContext) -> np.ndarray:
        raise NotImplementedError


_REGISTRY: dict[str, str] = {
    "stretch": "spatialize.infill.stretch:Stretch",
    "bgpull": "spatialize.infill.bgpull:BgPull",
    "opencv": "spatialize.infill.opencv_inpaint:Telea",
    "opencv-ns": "spatialize.infill.opencv_inpaint:NavierStokes",
    "lama": "spatialize.infill.lama:LaMa",
    "iw3": "spatialize.infill.iw3:IW3",
}
_INSTANCES: dict[str, Backend] = {}

ALL_BACKENDS = list(_REGISTRY)
DEFAULT_CANDIDATES = ["stretch", "bgpull", "opencv", "lama"]


def get_backend(name: str) -> Backend:
    if name not in _INSTANCES:
        mod, cls = _REGISTRY[name].split(":")
        _INSTANCES[name] = getattr(importlib.import_module(mod), cls)()
    return _INSTANCES[name]


def available_backends() -> list[str]:
    out = []
    for n in ALL_BACKENDS:
        try:
            if get_backend(n).available():
                out.append(n)
        except Exception:
            pass
    return out
