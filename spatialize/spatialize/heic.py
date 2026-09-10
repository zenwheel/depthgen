"""Spatial HEIC output through the existing spatialPhotoTool (Swift), plus our XMP marks via exiftool."""
from __future__ import annotations

import math
import os
import shutil
import subprocess

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EXIFTOOL_CFG = os.path.join(HERE, "stereolenses.exiftool.cfg")


def hfov_from_exif(exif: bytes | None, width: int, height: int, default: float = 60.0) -> tuple[float, str]:
    """Horizontal FOV from FocalLengthIn35mmFilm if present (36 mm wide frame; 24 mm for portrait)."""
    if not exif:
        return default, "default"
    try:
        ex = Image.Exif()
        ex.load(exif)
        ifd = ex.get_ifd(0x8769)
        f35 = ifd.get(0xA405) or ex.get(0xA405)
        if f35:
            frame = 36.0 if width >= height else 24.0
            return round(2 * math.degrees(math.atan(frame / (2 * float(f35)))), 2), f"from FocalLengthIn35mmFilm={f35}"
    except Exception:
        pass
    return default, "default"


def write_spatial_heic(left_path: str, right_path: str, out_path: str, hfov: float, baseline_mm: float,
                       tags: dict[str, str] | None, log) -> bool:
    exe = shutil.which("spatialPhotoTool")
    if not exe:
        log("  spatialPhotoTool not on PATH; skipping HEIC")
        return False
    cmd = [exe, "--pairs", "--hfov", f"{hfov:g}", "-b", f"{baseline_mm:g}", left_path, right_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    produced = os.path.splitext(left_path)[0] + ".heic"
    if r.returncode != 0 or not os.path.isfile(produced):
        log(f"  spatialPhotoTool failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")
        return False
    if os.path.abspath(produced) != os.path.abspath(out_path):
        os.replace(produced, out_path)
    if tags:
        et = shutil.which("exiftool")
        if et and os.path.isfile(EXIFTOOL_CFG):
            args = [et, "-config", EXIFTOOL_CFG, "-overwrite_original", "-q"]
            args += [f"-XMP-stereolenses:{k}={v}" for k, v in tags.items()]
            r2 = subprocess.run(args + [out_path], capture_output=True, text=True)
            if r2.returncode != 0:
                log(f"  exiftool could not tag the HEIC: {r2.stderr.strip()[-200:]}")
        else:
            log("  exiftool missing; HEIC carries the _spatialized name but no XMP mark")
    return True
