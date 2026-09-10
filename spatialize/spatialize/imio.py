"""Image I/O: display-oriented loading (JPEG/HEIC/PNG), EXIF/XMP carry-over, stereolenses XMP tags."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

try:  # HEIC input
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional at import time
    pillow_heif = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}
STEREOLENSES_NS = "http://stereolenses.com/xmp/1.0/"
ORIENTATION_TAG = 0x0112


@dataclass
class Photo:
    rgb: np.ndarray                      # HxWx3 uint8, display orientation
    exif: bytes | None = None            # EXIF blob with orientation reset to 1
    xmp: bytes | None = None             # original XMP packet (orientation removed), or None
    info: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]


def _strip_xmp_orientation(xmp: bytes) -> bytes:
    """Drop tiff:Orientation from an XMP packet (attribute or element form)."""
    s = xmp.decode("utf-8", "replace")
    s = re.sub(r'\s+tiff:Orientation\s*=\s*"[^"]*"', "", s)
    s = re.sub(r"<tiff:Orientation>[^<]*</tiff:Orientation>\s*", "", s)
    return s.encode("utf-8")


def load_photo(path: str) -> Photo:
    """Load as RGB in display orientation (EXIF orientation applied), keeping EXIF/XMP."""
    im = Image.open(path)
    im.load()
    exif = im.getexif()
    xmp = im.info.get("xmp")
    if isinstance(xmp, str):
        xmp = xmp.encode("utf-8")
    im = ImageOps.exif_transpose(im)
    if exif:
        exif[ORIENTATION_TAG] = 1
        exif_bytes = exif.tobytes()
    else:
        exif_bytes = None
    if xmp:
        xmp = _strip_xmp_orientation(xmp)
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return Photo(rgb=rgb, exif=exif_bytes, xmp=xmp, info={"path": path, "mode": im.mode, "size": im.size})


def load_depth8(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load a depthgen map as HxW uint8 (255 = near). Resized (bilinear) if `size` (w, h) differs."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("L")
    if size is not None and im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def save_depth_png(depth8: np.ndarray, path: str) -> None:
    """Same format depthgen writes: 8-bit RGB PNG, bright = near."""
    Image.fromarray(np.repeat(depth8[..., None], 3, axis=2), "RGB").save(path, format="PNG", optimize=True)


def _xmp_packet(tags: dict[str, str], base: bytes | None) -> bytes:
    """Return an XMP packet carrying `tags` in the stereolenses namespace, merged into `base`."""
    attrs = " ".join(f'stereolenses:{k}="{v}"' for k, v in tags.items())
    desc = (f'<rdf:Description rdf:about="" xmlns:stereolenses="{STEREOLENSES_NS}" {attrs}/>')
    if base:
        s = base.decode("utf-8", "replace")
        s = re.sub(r'\s+stereolenses:\w+\s*=\s*"[^"]*"', "", s)          # replace earlier marks
        s = re.sub(r"<stereolenses:\w+>[^<]*</stereolenses:\w+>\s*", "", s)
        if "</rdf:RDF>" in s:
            return s.replace("</rdf:RDF>", desc + "</rdf:RDF>", 1).encode("utf-8")
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="spatialize">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f"{desc}</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    ).encode("utf-8")


def save_jpeg(rgb: np.ndarray, path: str, photo: Photo | None, tags: dict[str, str] | None = None,
              quality: int = 95) -> None:
    """Write a JPEG with the source EXIF (orientation = 1) and XMP plus our stereolenses tags."""
    im = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    kw: dict = {"quality": quality, "subsampling": 0, "optimize": True}
    if photo is not None and photo.exif:
        kw["exif"] = photo.exif
    xmp = _xmp_packet(tags, photo.xmp if photo else None) if tags else (photo.xmp if photo else None)
    if xmp:
        kw["xmp"] = xmp
    im.save(path, format="JPEG", **kw)


def save_png(rgb_or_gray: np.ndarray, path: str) -> None:
    Image.fromarray(np.ascontiguousarray(rgb_or_gray)).save(path, format="PNG", optimize=False, compress_level=3)


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMAGE_EXTS:
        return False
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith("-depth") or stem.endswith("-sdepth") or stem.endswith("-deptht") or stem.endswith("-sdeptht"):
        return False
    if re.search(r"_spatialized(_left|_right|_sbs|_xeye)?$", stem):
        return False
    return True


def to_gray_f32(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
