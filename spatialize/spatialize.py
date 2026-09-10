#!/usr/bin/env python3
"""spatialize: synthesize a stereo pair from one photo using the depthgen depth map.

    spatialize PHOTO.jpg                       # uses PHOTO-depth.png (or runs depthgen), writes
                                               # PHOTO_spatialized_{left,right,sbs,xeye}.jpg next to it
    spatialize PHOTO.heic -o out --parallax 2.5 --convergence 40% --infill lama --heic
    spatialize DIR1 DIR2 --json --debug-dir dbg

Depth format: depthgen writes normalized inverse depth (255 = near). It is linear in disparity, so
shift_px = parallax_px * (value - zero_plane) / 255.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spatialize.imio import IMAGE_EXTS, is_image_file  # noqa: E402


def collect_inputs(args: list[str]) -> list[str]:
    out = []
    for a in args:
        if os.path.isdir(a):
            for p in sorted(glob.glob(os.path.join(a, "**", "*"), recursive=True)):
                if os.path.isfile(p) and is_image_file(p):
                    out.append(p)
        elif os.path.isfile(a):
            out.append(a)
        else:
            print(f"warning: {a} not found", file=sys.stderr)
    return out


def output_paths(image_path: str, out_dir: str | None, sbs: bool, heic: bool) -> dict[str, str]:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    d = out_dir or os.path.dirname(os.path.abspath(image_path))
    base = os.path.join(d, stem + "_spatialized")
    paths = {"left": base + "_left.jpg", "right": base + "_right.jpg"}
    if sbs:
        paths["sbs"] = base + "_sbs.jpg"
        paths["xeye"] = base + "_xeye.jpg"
    if heic:
        paths["heic"] = base + ".heic"
    return paths


def up_to_date(outputs: dict[str, str], inputs: list[str]) -> bool:
    try:
        newest_in = max(os.path.getmtime(p) for p in inputs)
        return all(os.path.isfile(p) and os.path.getmtime(p) >= newest_in for p in outputs.values())
    except OSError:
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="images and/or directories")
    p.add_argument("-o", "--out-dir", help="output directory (default: next to each input)")
    p.add_argument("--depth", help="depth PNG to use (single input only; default <stem>-depth.png or run depthgen)")
    g = p.add_argument_group("parallax")
    g.add_argument("--parallax", default="2.0", help="near-to-far disparity range, %% of width or Npx (default 2.0)")
    g.add_argument("--max-parallax", default="3.5", help="safety cap, %% of width or Npx (default 3.5)")
    g.add_argument("--convergence", default="auto", help="auto | near | far | median | 0..255 | NN%% (default auto)")
    g.add_argument("--far-limit", type=float, default=1.2,
                   help="auto convergence: max uncrossed (behind-screen) disparity, %% of width (default 1.2)")
    g.add_argument("--eyes", choices=["symmetric", "right"], default="symmetric",
                   help="synthesize both eyes with +-P/2, or keep the original as left and synthesize right")
    g.add_argument("--swap", action="store_true", help="output the pair mirrored (left<->right)")
    g = p.add_argument_group("depth conditioning")
    g.add_argument("--depth-blur", type=float, default=0.0, help="bilateral smoothing sigma in px (0 = off)")
    g.add_argument("--edge-refine", dest="edge_refine", action="store_true", default=True,
                   help="guided filter of the depth against the photo (default on)")
    g.add_argument("--no-edge-refine", dest="edge_refine", action="store_false")
    g.add_argument("--guided-radius", type=int, default=8)
    g.add_argument("--guided-eps", type=float, default=400.0)
    g.add_argument("--fg-erode", type=int, default=0, help="erode near regions by N px (halo removal)")
    g.add_argument("--edge-sharpen", type=int, default=3,
                   help="snap depth ramps at strong edges into steps within this radius (px; 0 = off, default 3)")
    g = p.add_argument_group("infill")
    g.add_argument("--infill", default="auto", help="auto | stretch | bgpull | opencv | opencv-ns | lama | iw3")
    g.add_argument("--candidates", default=",".join(["stretch", "bgpull", "opencv", "lama"]),
                   help="comma list of backends auto may try")
    g.add_argument("--border", choices=["crop", "fill"], default="crop",
                   help="frame edges: crop both eyes equally (default) or infill the uncovered strips")
    g = p.add_argument_group("output")
    g.add_argument("--no-sbs", action="store_true", help="skip the _sbs / _xeye images")
    g.add_argument("--heic", action="store_true", help="also write a spatial HEIC via spatialPhotoTool")
    g.add_argument("--hfov", type=float, help="horizontal FOV for the HEIC (default: from EXIF, else 60)")
    g.add_argument("--baseline", type=float, default=0.0, help="baseline (mm) written to the HEIC (default 0)")
    g.add_argument("--quality", type=int, default=95)
    g.add_argument("--debug-dir", help="dump conditioned depth, hole masks, every candidate, score table")
    g.add_argument("--json", action="store_true", help="print a machine-readable summary to stdout")
    g.add_argument("-f", "--force", action="store_true", help="regenerate even if outputs are up to date")
    g.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    g.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    log = (lambda *_: None) if a.quiet else (lambda *m: print(*m, file=sys.stderr, flush=True))
    files = collect_inputs(a.inputs)
    if not files:
        log("no input images")
        return 1
    if a.depth and len(files) != 1:
        log("--depth can only be used with a single input")
        return 2
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    if a.debug_dir:
        os.makedirs(a.debug_dir, exist_ok=True)

    import numpy as np

    from spatialize.depth import DepthConditioning, find_or_make_depth, parse_parallax
    from spatialize.device import pick_device
    from spatialize.heic import hfov_from_exif, write_spatial_heic
    from spatialize.imio import load_depth8, load_photo, save_jpeg, save_png
    from spatialize.pipeline import Models, Options, spatialize_image

    device = pick_device(a.device)
    models = Models(device)
    log(f"device {device}; infill backends available: {', '.join(models.available)}")
    cond = DepthConditioning(blur=a.depth_blur, edge_refine=a.edge_refine, guided_radius=a.guided_radius,
                             guided_eps=a.guided_eps, fg_erode=a.fg_erode, edge_sharpen=a.edge_sharpen)
    candidates = [c.strip() for c in a.candidates.split(",") if c.strip()]
    if a.infill != "auto" and a.infill not in models.available:
        log(f"infill backend {a.infill!r} is not available (have: {', '.join(models.available)})")
        return 2

    summaries = []
    failed = 0
    for path in files:
        t0 = time.time()
        outs = output_paths(path, a.out_dir, not a.no_sbs, a.heic)
        log(f"== {path}")
        try:
            depth_path, t_depthgen = find_or_make_depth(path, a.depth, log)
            if not a.force and up_to_date(outs, [path, depth_path]):
                log("  up to date, skipping (use --force)")
                summaries.append({"input": path, "skipped": True, "outputs": outs})
                continue
            photo = load_photo(path)
            depth8 = load_depth8(depth_path, (photo.width, photo.height))
            opt = Options(parallax_px=parse_parallax(a.parallax, photo.width),
                          max_parallax_px=parse_parallax(a.max_parallax, photo.width),
                          convergence=a.convergence, far_limit_pct=a.far_limit, eyes=a.eyes, swap=a.swap,
                          infill=a.infill, candidates=candidates, border=a.border, conditioning=cond,
                          device=device, debug=bool(a.debug_dir))
            res = spatialize_image(photo.rgb, depth8, opt, models, log)
            chosen = "+".join(sorted({e.chosen for e in res.eyes}))
            tags = {"spatialized": "true", "parallax": f"{res.stats['parallax_pct']:.3f}%",
                    "infill": chosen, "convergence": f"{res.stats['convergence']['value']:.1f}"}
            save_jpeg(res.left, outs["left"], photo, tags, a.quality)
            save_jpeg(res.right, outs["right"], photo, tags, a.quality)
            if not a.no_sbs:
                save_jpeg(np.concatenate([res.left, res.right], axis=1), outs["sbs"], photo, tags, a.quality)
                save_jpeg(np.concatenate([res.right, res.left], axis=1), outs["xeye"], photo, tags, a.quality)
            heic_ok = None
            if a.heic:
                hfov, how = (a.hfov, "option") if a.hfov else hfov_from_exif(photo.exif, photo.width, photo.height)
                log(f"  HEIC: hfov {hfov} ({how}), baseline {a.baseline} mm")
                heic_ok = write_spatial_heic(outs["left"], outs["right"], outs["heic"], hfov, a.baseline, tags, log)
                res.stats["heic"] = {"written": heic_ok, "hfov": hfov, "baseline_mm": a.baseline}
            if a.debug_dir:
                stem = os.path.splitext(os.path.basename(path))[0]
                dbg = os.path.join(a.debug_dir, stem)
                save_png(np.clip(res.depth + 0.5, 0, 255).astype(np.uint8), dbg + "_depth_conditioned.png")
                rows = []
                for e in res.eyes:
                    save_png((e.hole.astype(np.uint8) * 255), f"{dbg}_{e.name}_holes.png")
                    save_png((e.warped.border.astype(np.uint8) * 255), f"{dbg}_{e.name}_border.png")
                    save_jpeg(e.warped.rgb, f"{dbg}_{e.name}_warp_raw.jpg", None, None, 85)
                    for cname, img in e.candidates.items():
                        save_jpeg(img, f"{dbg}_{e.name}_infill_{cname}.jpg", None, None, 85)
                    for cname, sc in e.scores.items():
                        rows.append({"eye": e.name, "backend": cname, "chosen": cname == e.chosen, **sc})
                with open(dbg + "_scores.json", "w") as f:
                    json.dump({"stats": res.stats, "table": rows}, f, indent=2)
                with open(dbg + "_scores.txt", "w") as f:
                    f.write(f"{'eye':6} {'backend':10} {'photo':>8} {'seam':>8} {'lpips':>8} {'total':>8} {'fill_s':>7}\n")
                    for r in rows:
                        f.write(f"{r['eye']:6} {r['backend']:10} {r.get('photo', float('nan')):8.3f} "
                                f"{r.get('seam', float('nan')):8.3f} {r.get('lpips', float('nan')):8.4f} "
                                f"{r.get('total', float('nan')):8.4f} {r['fill_seconds']:7.2f}{'  *' if r['chosen'] else ''}\n")
            res.stats["timings"]["depthgen"] = round(t_depthgen, 3)
            res.stats["timings"]["wall"] = round(time.time() - t0, 3)
            res.stats.update({"input": path, "depth": depth_path, "outputs": outs, "skipped": False})
            summaries.append(res.stats)
            log(f"  wrote {outs['left']} (+{len(outs) - 1} more) in {time.time() - t0:.1f}s")
        except Exception as e:
            failed += 1
            log(f"  FAILED: {e}")
            summaries.append({"input": path, "error": str(e)})
            if os.environ.get("SPATIALIZE_DEBUG"):
                raise
    if a.json:
        print(json.dumps(summaries if len(summaries) != 1 else summaries[0], indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
