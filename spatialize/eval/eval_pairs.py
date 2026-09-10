#!/usr/bin/env python3
"""Evaluate infill backends against REAL stereo pairs.

    eval/eval_pairs.py OUT_DIR LEFT1.jpg [LEFT2.jpg ...]
    eval/eval_pairs.py OUT_DIR --sample N CONTENT_DIR        # N pairs spread across the library
    options: --backends stretch,bgpull,opencv,opencv-ns,lama,auto  --edge-sharpen 2  --fg-erode 0

For each <slug>_left.jpg + <slug>_right.jpg (+ <slug>_left-depth.png mono depth) the right eye is
synthesized from the left alone and compared to the real right image. The real pair has a physical
baseline, so the parallax and zero plane are fitted first: SIFT matches give real disparities, a
robust affine fit disparity = a*mono + b maps the mono map onto them (parallax_px = 255*a,
zero_plane = -b/a). Metrics: PSNR / SSIM / LPIPS over the common cropped frame and PSNR / MAE inside
the disocclusion holes (the only pixels the infill actually changes).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from spatialize.depth import DepthConditioning  # noqa: E402
from spatialize.device import pick_device  # noqa: E402
from spatialize.imio import load_depth8, load_photo, to_gray_f32  # noqa: E402
from spatialize.pipeline import Models, Options, spatialize_image  # noqa: E402


# ----------------------------------------------------------------------------- disparity fit

def fit_parallax(L: np.ndarray, R: np.ndarray, mono: np.ndarray, work_w: int = 1600):
    """Return (parallax_px, zero_plane, info) or None."""
    h, w = mono.shape
    s = min(1.0, work_w / w)
    Ls = cv2.resize(to_gray_f32(L), None, fx=s, fy=s).astype(np.uint8)
    Rs = cv2.resize(to_gray_f32(R), None, fx=s, fy=s).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = sift.detectAndCompute(Ls, None)
    k2, d2 = sift.detectAndCompute(Rs, None)
    if d1 is None or d2 is None:
        return None
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    pts = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            p1, p2 = k1[m[0].queryIdx].pt, k2[m[0].trainIdx].pt
            if abs(p1[1] - p2[1]) <= 3.0 * s + 1.0:              # epipolar: same row
                pts.append((p1[0] / s, p1[1] / s, (p1[0] - p2[0]) / s))
    if len(pts) < 40:
        return None
    pts = np.asarray(pts)
    xs, ys, disp = pts[:, 0], pts[:, 1], pts[:, 2]
    mono_v = mono[np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)].astype(np.float64)
    # robust affine fit with iterative outlier rejection
    keep = np.ones(len(disp), bool)
    a = b = 0.0
    for _ in range(6):
        if keep.sum() < 20:
            return None
        a, b = np.polyfit(mono_v[keep], disp[keep], 1)
        resid = disp - (a * mono_v + b)
        thr = max(1.5, 2.5 * np.median(np.abs(resid[keep])) * 1.4826)
        keep = np.abs(resid) < thr
    if a <= 0:
        return None
    inl = keep.sum()
    corr = float(np.corrcoef(mono_v[keep], disp[keep])[0, 1])
    return 255.0 * a, -b / a, {"matches": int(len(disp)), "inliers": int(inl), "corr": round(corr, 3),
                               "disp_range_px": [round(float(np.percentile(disp[keep], 2)), 1),
                                                 round(float(np.percentile(disp[keep], 98)), 1)],
                               "resid_px": round(float(np.median(np.abs(resid[keep]))), 2)}


# ----------------------------------------------------------------------------- metrics

def psnr(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        if not mask.any():
            return float("nan")
        d = d[mask]
    mse = float(d.mean())
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def mae(a, b, mask=None) -> float:
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if mask is not None:
        if not mask.any():
            return float("nan")
        d = d[mask]
    return float(d.mean())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    x, y = to_gray_f32(a), to_gray_f32(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())


class LPIPSMetric:
    def __init__(self, device):
        import warnings

        os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "weights", "torch"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import lpips
            import torch
        self.torch = torch
        self.net = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        self.device = device

    def __call__(self, a: np.ndarray, b: np.ndarray, max_w: int = 1024) -> float:
        s = min(1.0, max_w / a.shape[1])
        if s < 1:
            a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t = lambda z: self.torch.from_numpy(np.ascontiguousarray(z)).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(self.device)
        with self.torch.inference_mode():
            return float(self.net(t(a), t(b)).item())


# ----------------------------------------------------------------------------- main

def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    out_dir = args[0]
    os.makedirs(out_dir, exist_ok=True)
    backends = ["stretch", "bgpull", "opencv", "lama", "auto"]
    cond_kw = {}
    rest = []
    i = 1
    while i < len(args):
        if args[i] == "--backends":
            backends = args[i + 1].split(","); i += 2
        elif args[i] == "--edge-sharpen":
            cond_kw["edge_sharpen"] = int(args[i + 1]); i += 2
        elif args[i] == "--fg-erode":
            cond_kw["fg_erode"] = int(args[i + 1]); i += 2
        elif args[i] == "--no-edge-refine":
            cond_kw["edge_refine"] = False; i += 1
        else:
            rest.append(args[i]); i += 1
    if rest and rest[0] == "--sample":
        n = int(rest[1])
        allp = sorted(glob.glob(os.path.join(rest[2], "**", "*_left.jpg"), recursive=True))
        allp = [p for p in allp if os.path.isfile(p.replace("_left.jpg", "_right.jpg"))
                and os.path.isfile(p.replace("_left.jpg", "_left-depth.png"))]
        step = max(1, len(allp) // n)
        lefts = allp[::step][:n]
    else:
        lefts = rest

    device = pick_device()
    models = Models(device)
    lp = LPIPSMetric(device)
    cond = DepthConditioning(**cond_kw)
    print(f"device {device}; backends {backends}; conditioning {cond}", flush=True)

    rows = []
    for left in lefts:
        slug = os.path.basename(left).replace("_left.jpg", "")
        right = left.replace("_left.jpg", "_right.jpg")
        mono_path = left.replace("_left.jpg", "_left-depth.png")
        L = load_photo(left).rgb
        R = load_photo(right).rgb
        if R.shape != L.shape:
            print(f"== {slug}: size mismatch, skipped", flush=True)
            continue
        mono = load_depth8(mono_path, (L.shape[1], L.shape[0]))
        fit = fit_parallax(L, R, mono)
        if fit is None:
            print(f"== {slug}: could not fit disparity, skipped", flush=True)
            continue
        parallax_px, zero, info = fit
        h, w = mono.shape
        print(f"== {slug}: parallax {parallax_px:.1f}px ({100 * parallax_px / w:.2f}%), zero plane {zero:.1f}, "
              f"{info['inliers']}/{info['matches']} inliers, corr {info['corr']}", flush=True)
        row = {"slug": slug, "left": left, "width": w, "height": h, "parallax_px": round(parallax_px, 2),
               "parallax_pct": round(100 * parallax_px / w, 3), "zero_plane": round(zero, 2), "fit": info,
               "backends": {}}
        for be in backends:
            t0 = time.time()
            opt = Options(parallax_px=parallax_px, max_parallax_px=1e9, convergence=f"{zero:.3f}", eyes="right",
                          infill=be, border="crop", conditioning=cond, device=device)
            try:
                res = spatialize_image(L, mono, opt, models, log=lambda *a: None)
            except Exception as e:
                print(f"   {be}: FAILED {e}", flush=True)
                continue
            cl, cr = res.crop
            Rc = R[:, cl:w - cr]
            eye = res.eyes[0]
            hole = eye.hole[:, cl:w - cr]
            near = cv2.dilate(hole.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            m = {"psnr": round(psnr(res.right, Rc), 3), "ssim": round(ssim(res.right, Rc), 4),
                 "lpips": round(lp(res.right, Rc), 4),
                 "hole_psnr": round(psnr(res.right, Rc, near), 3), "hole_mae": round(mae(res.right, Rc, near), 3),
                 "hole_px": int(hole.sum()), "chosen": eye.chosen, "reason": eye.reason,
                 "scores": eye.scores, "seconds": round(time.time() - t0, 2)}
            row["backends"][be] = m
            print(f"   {be:9} psnr {m['psnr']:6.2f} ssim {m['ssim']:.4f} lpips {m['lpips']:.4f} | holes {m['hole_px']:7d}px "
                  f"psnr {m['hole_psnr']:6.2f} mae {m['hole_mae']:6.2f} | {m['seconds']:5.1f}s"
                  + (f"  -> {eye.chosen}" if be == "auto" else ""), flush=True)
            if be == backends[0]:
                cv2.imwrite(os.path.join(out_dir, f"{slug}_real_right.jpg"), cv2.cvtColor(Rc, cv2.COLOR_RGB2BGR), [1, 90])
            cv2.imwrite(os.path.join(out_dir, f"{slug}_{be}.jpg"), cv2.cvtColor(res.right, cv2.COLOR_RGB2BGR), [1, 90])
        rows.append(row)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump({"conditioning": cond.__dict__, "rows": rows}, f, indent=2)

    # ---- summary table
    if rows:
        lines = ["| backend | PSNR | SSIM | LPIPS | hole PSNR | hole MAE | best-hole wins | mean s |", "|---|---|---|---|---|---|---|---|"]
        for be in backends:
            vals = [r["backends"][be] for r in rows if be in r["backends"]]
            if not vals:
                continue
            wins = 0
            for r in rows:
                cands = {k: v["hole_psnr"] for k, v in r["backends"].items() if k != "auto" and not np.isnan(v["hole_psnr"])}
                if be in r["backends"] and cands and r["backends"][be]["hole_psnr"] >= max(cands.values()) - 1e-6:
                    wins += 1
            f = lambda k: np.nanmean([v[k] for v in vals])
            lines.append(f"| {be} | {f('psnr'):.2f} | {f('ssim'):.4f} | {f('lpips'):.4f} | {f('hole_psnr'):.2f} | "
                         f"{f('hole_mae'):.2f} | {wins}/{len(vals)} | {f('seconds'):.1f} |")
        if "auto" in backends:
            picks = {}
            for r in rows:
                if "auto" in r["backends"]:
                    c = r["backends"]["auto"]["chosen"]
                    picks[c] = picks.get(c, 0) + 1
            lines.append("")
            lines.append("auto picks: " + ", ".join(f"{k} x{v}" for k, v in sorted(picks.items())))
        table = "\n".join(lines)
        print("\n" + table)
        with open(os.path.join(out_dir, "results.md"), "w") as f:
            f.write(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
