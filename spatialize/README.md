# spatialize

Synthesize a stereo pair from a single photo using the depth map that `depthgen` produces.
A companion to `depthgen` (Apple Depth Pro, monocular) and `stereo_depth_gen`; it lives in its
own directory with its own Python virtualenv and is called through `bin/spatialize`, like the
other tools the site build shells out to.

```
photo.jpg  +  photo-depth.png  ──►  photo_spatialized_left.jpg / _right.jpg / _sbs.jpg / _xeye.jpg  [ / .heic ]
```

## Setup

```sh
./setup.sh              # venv, pinned deps, LaMa + LPIPS weights (~250 MB into weights/, never committed)
./setup.sh --with-iw3   # optionally also install nunif/iw3 into the venv (reference backend, ~GBs)
```

`--with-iw3` clones nunif into `vendor/`, installs its requirements and downloads its depth models
(several GB, even though we feed iw3 our own depth). nunif pins PyAV 15, which has no Python 3.14
wheel and does not compile against ffmpeg 9; `setup.sh` installs current PyAV instead, built against
the system ffmpeg when `pkg-config` can find it (`brew install ffmpeg pkg-config`) and otherwise as
the binary wheel with its bundled FFmpeg.

Requires Python 3.11+ (tested on 3.14), macOS with Apple silicon for MPS (falls back to CPU),
`depthgen` on `PATH` if a depth map has to be generated, `spatialPhotoTool` on `PATH` for `--heic`,
and `exiftool` to tag the HEIC (the JPEGs are tagged without it).

## Usage

```sh
bin/spatialize IMG_1234.jpg                          # -> IMG_1234_spatialized_{left,right,sbs,xeye}.jpg next to it
bin/spatialize IMG_1234.heic -o out --heic            # + IMG_1234_spatialized.heic (spatialPhotoTool)
bin/spatialize photos/ --parallax 2.5 --json          # directories recurse; one process, models loaded once
bin/spatialize a.jpg --depth other-depth.png --infill lama --debug-dir dbg
bin/spatialize a.jpg --infill auto --candidates stretch,lama
```

Inputs: JPEG, HEIC/HEIF, PNG (TIFF/WebP work too). If `<stem>-depth.png` exists next to the input,
or `--depth PATH` is given, it is used; otherwise `depthgen <image>` is run and its output picked
up. The photo is loaded in display orientation (EXIF orientation applied) so it lines up with the
map, and the source EXIF and XMP are carried into every output with orientation reset to 1.

Outputs are skipped when they exist and are newer than the input and depth map (`--force` to redo).

### Parallax controls

| option | default | meaning |
|---|---|---|
| `--parallax P` | `2.0` | total disparity between nearest and farthest content, % of width; `40px` for absolute pixels |
| `--max-parallax` | `3.5` | safety cap (% or px); the tool warns and clamps above it |
| `--convergence` | `auto` | zero-parallax plane: `auto`, `near` (all behind the screen), `far` (all in front), `median`, a literal `0..255`, or a percentile `NN%` |
| `--far-limit` | `1.2` | `auto` only: maximum behind-screen (uncrossed) disparity, % of width |
| `--eyes` | `symmetric` | `symmetric` synthesizes both eyes with ±P/2 so artifacts split evenly; `right` keeps the original as the left eye |
| `--swap` | off | output the pair mirrored |
| `--border` | `crop` | frame edges where content slid away: crop both eyes by the same amount (pair stays aligned) or `fill` the strips |

`auto` convergence puts the plane at the 75th percentile of the depth map, so roughly the nearest
quarter of the pixels come forward of the screen, then moves it back if the farthest content
would otherwise exceed `--far-limit` behind the screen.

### Depth conditioning (before warping)

| option | default | meaning |
|---|---|---|
| `--depth-blur S` | `0` | bilateral smoothing (sigma px) to remove 8-bit banding without softening edges |
| `--edge-refine` / `--no-edge-refine` | on | guided filter of the map against the photo (`--guided-radius 8 --guided-eps 400`) so depth edges snap to image edges |
| `--edge-sharpen R` | `3` | Depth Pro output is resampled from 1536 px, so every depth edge is a ramp a few px wide. Warping a ramp *stretches* the pixels across it instead of opening a hole; snapping ramps to steps within radius R makes real disocclusions that the infill can handle. `0` keeps the rubber-sheet look |
| `--fg-erode N` | `0` | erode near regions by N px to kill the halo of background stuck to hair / foliage |

## How the depth map becomes disparity

`depthgen` writes an 8-bit grayscale PNG (stored as RGB), full input resolution, display
orientation, holding **normalized inverse depth** from Depth Pro: 255 = nearest, 0 = farthest,
stretched to the full range per image. Inverse depth is proportional to disparity, so the shift is
linear in the value and the tool never treats the map as metric depth:

```
parallax_px = P% * width                 (or the px value given)
shift(x, y) = parallax_px * (depth(x, y) - zero_plane) / 255        # + = in front of the screen
left eye  : x' = x + shift / 2          (symmetric)     right eye : x' = x - shift / 2
right only: left = original,            right eye : x' = x - shift
```

The warp is a forward splat: every source pixel is written to its two nearest target columns with
bilinear weights, a per-target z-buffer keeps the nearest surface (contributions within ~1 px worth
of depth of the winner blend), stretched slanted surfaces have their sampling cracks closed, and
edge debris (fractional taps of foreground edge pixels landing inside the disocclusion) is removed
so the holes have clean boundaries. What remains uncovered is either a **hole** (background that
was occluded in the source) or a **border strip** (content that slid off the frame). The first
`--rim` (2) background pixels past each hole are treated as hole too: in the source they are the
occluder's anti-aliased rim and would otherwise draw a ghost line along every seam.

## Infill and automatic selection

All backends implement `fill(warped_rgb, hole_mask, depth, original, ctx) -> rgb` and only touch
hole pixels (plus a sub-pixel feather):

| backend | what it does | good for |
|---|---|---|
| `stretch` | interpolates each hole run between its two flanks along the row: exactly what rendering a depth-displaced mesh does (`generate_usdz.py`, `fake3d.js`) | small parallax, narrow holes; cheapest |
| `bgpull` | copies the far-side (background) pixels mirrored into the hole, then feathers | smooth or gently textured backgrounds (sky, walls, bokeh) |
| `opencv` / `opencv-ns` | `cv2.inpaint` Telea / Navier-Stokes on crops around each hole cluster | moderate holes; tends to bleed foreground colour |
| `lama` | big-lama (TorchScript) on native-resolution 768 px tiles around each hole cluster, on MPS. The occluding layer next to the hole is masked too, otherwise LaMa extends the object into the hole instead of continuing the background | wide holes on textured backgrounds; slowest, output is smooth |
| `iw3` | optional: nunif/iw3's own warp + inpainting fed with **our** depth map through its export/import format (`iw3_export.yml` + 16-bit depth PNG, mapper disabled). Whole-frame backend, always symmetric | reference comparison |

`--infill auto` (default) picks per eye, based on measurements, and logs everything to `--json`:

1. **Hole statistics** from the warp: total area, run widths (max / p95), and the texture of the
   background immediately beyond each hole (mean gradient magnitude and local std in the 8 px
   strip past the far flank).
2. **Cheap rules**: max hole width ≤ 2 px → `stretch`, done. Smooth background (gradient < 4) →
   only cheap backends are scored (`lama` is never needed there). Textured background with p95
   run width ≥ 24 px → all candidates including `lama`. Otherwise the cheap backends are scored.
   (The 24 px gate comes from the benchmark below: LaMa never beat the cheap fills on narrower
   holes and costs several seconds per eye.)
3. **Reference-free scores** for each remaining candidate (lower is better):
   * `photo`: the synthesized eye is warped back to the source viewpoint with the same shift field
     and compared to the original outside the holes (mean abs error). Identical for backends that
     only touch holes; it catches whole-frame backends and boundary bleed.
   * `seam`: brightness step across the background-side hole boundary after filling, minus the
     typical horizontal gradient of that background (so texture is not penalized).
   * `lpips`: perceptual distance between 128 px windows on the fill and the same windows shifted
     onto the neighbouring background, with everything outside the hole replaced by the context,
     so only the fill is compared.
   * `total = photo/5 + seam/20 + lpips`.
4. The lowest `total` wins; `--candidates a,b` restricts what auto tries. Per-hole selection
   (different backends for different holes of one image) is not implemented; selection is per eye.

## Outputs and how the site consumes them

* `<stem>_spatialized_left.jpg`, `_right.jpg` (quality 95, 4:4:4), `_sbs.jpg` (parallel, left on
  the left) and `_xeye.jpg` unless `--no-sbs`; `-o DIR` sets the directory.
* `--heic` also writes `<stem>_spatialized.heic` through `spatialPhotoTool --pairs --hfov H -b B`.
  `--hfov` defaults to the value derived from EXIF `FocalLengthIn35mmFilm` (else 60°). A synthetic
  pair has no physical baseline, so `--baseline` defaults to 0 mm; the site's `image_gen.py`
  refuses a baseline of 0, so pass e.g. `--baseline 65` for HEICs that will be ingested there.
* Every output carries XMP `stereolenses:spatialized="true"`, `stereolenses:parallax`,
  `stereolenses:infill` and `stereolenses:convergence` (namespace `http://stereolenses.com/xmp/1.0/`;
  written directly into the JPEGs, via `exiftool -config spatialize/stereolenses.exiftool.cfg` into
  the HEIC). The stems carry `_spatialized`. `image_gen.py`'s `_is_marked_spatialized` recognizes
  either mark and sets `spatialized: true` for the badge; Apple's 3-image spatialize group box is
  **not** forged.
* For the site, drop the `_sbs.jpg` (or the HEIC) into `photo_library/`. JPEG sources get baseline
  and FOV from the filename (`_+65mm`, `_60deg` tokens), so name it e.g.
  `IMG_1234_spatialized_+65mm_60deg_sbs.jpg`.
* `--debug-dir DIR` dumps the conditioned depth, per-eye hole and border masks, the raw warp,
  every candidate fill, and `<stem>_scores.{json,txt}`. `--json` prints the summary (parallax used
  in px/%, convergence value and how it was chosen, per-eye hole stats, chosen backend with the
  reason and all scores, crop, timings).

## Evaluation on real stereo pairs

`eval/eval_pairs.py` takes real pairs from the site library (`<slug>_left.jpg` + `<slug>_right.jpg`
with the mono `<slug>_left-depth.png`), synthesizes the right eye from the left alone, and compares
it to the real right image. The real pair has a physical baseline, so the parallax and zero plane
are fitted first: SIFT matches on the same rows give real disparities, a robust affine fit
`disparity = a * mono + b` maps the mono map onto them (`parallax_px = 255 a`, `zero_plane = -b/a`,
usually outside 0..255 because parallel cameras put everything behind the convergence plane).

```sh
.venv/bin/python eval/eval_pairs.py eval/out --sample 12 ~/Developer/stereolenses-site/content/photos
.venv/bin/python eval/eval_pairs.py eval/out LEFT1.jpg LEFT2.jpg --backends stretch,bgpull,lama,auto --edge-sharpen 0
```

Results on 12 pairs spread across the library (2026-09-10, M4 Pro; parallax fitted per pair, 0.15-6.8 % of width; `auto` with the default candidates `stretch,bgpull,opencv,lama`; "best-hole wins" counts pairs where the backend had the highest hole-region PSNR; `mean s` is per pair including the warp and, for `auto`, the scoring):

| backend | PSNR | SSIM | LPIPS | hole PSNR | hole MAE | best-hole wins | mean s |
|---|---|---|---|---|---|---|---|
| stretch | 21.65 | 0.6766 | 0.1042 | 16.33 | 30.27 | 7/12 | 2.3 |
| bgpull | 21.59 | 0.6763 | 0.0976 | 16.25 | 29.35 | 3/12 | 2.2 |
| opencv | 21.62 | 0.6773 | 0.1000 | 16.16 | 30.38 | 2/12 | 2.0 |
| lama | 21.54 | 0.6750 | 0.1003 | 15.61 | 31.87 | 0/12 | 9.4 |
| auto | 21.63 | 0.6760 | 0.1026 | 16.32 | 29.74 | 4/12 | 5.5 |

auto picks: bgpull x11, stretch x1

The optional `iw3` backend (nunif's `mlbw_l2`, fed our depth) on 4 of those pairs, right eye only,
against `bgpull` on the same pairs:

| backend | PSNR | SSIM | LPIPS | hole PSNR | hole MAE | best-hole wins | mean s |
|---|---|---|---|---|---|---|---|
| bgpull | 21.73 | 0.6472 | 0.1183 | 16.51 | 28.35 | 4/4 | 1.9 |
| iw3 | 19.06 | 0.5746 | 0.2416 | 14.75 | 34.14 | 0/4 | 45.8 |

iw3's `--convergence` is bounded to the depth range, so the out-of-range zero planes of real pairs are
emulated by clamping and translating its eyes back by the constant residual; after that its eyes sit
within 1 px of ours. It is not a default candidate: it loses on every pair and runs 6-90 s per image.

Reading the numbers: whole-frame PSNR/SSIM/LPIPS are dominated by where the mono depth disagrees
with the real geometry (Depth Pro's relative depth is not affine in true disparity everywhere, and
the pairs have residual vertical misalignment), so they barely move between backends. The
hole-region columns (pixels within 3 px of a disocclusion) are the ones the infill actually
changes, and the spread there is small too: with the parallax matched to the real baseline the
holes are only a few pixels wide on most pairs, which is exactly the regime where `stretch` and
`bgpull` are as good as anything. The reference-free score picks the best-by-hole-PSNR backend or
one within 0.1 dB of it on most pairs; `lama` only pays off on wide holes over texture and is 3-10×
slower, which is why the cheap rules gate it.

## Performance

12 MP on an M4 Pro: depth conditioning ~0.15 s, each eye's warp ~1-1.5 s (torch on CPU, which
beats MPS for this scatter-heavy step), cheap fills 0.05-0.4 s, scoring ~0.3-0.5 s per candidate.
Measured with `--infill auto` (symmetric eyes, default candidates) on 12 MP inputs where the cheap
backends win: 6.7-8.9 s in the pipeline, 8.4-10.8 s wall including the four JPEG encodes. When the
holes are wide enough that `lama` is tried it adds 4-10 s per eye (20+ s total). `depthgen` itself,
when it has to run, is separate (and its first run loads the Core ML model).

## Known failure modes

* **Thin structures / noisy depth** (bare branches, wires, fences): the mono map is noisy around
  them, the warp shreds them, and no infill can repair a warp that is wrong. `--depth-blur 2` and
  a larger `--guided-eps` calm it down at the cost of depth detail; smaller parallax hides it.
* **Transparency and reflections** (glass, water, mirrors): a single depth per pixel cannot
  represent two surfaces; reflections stick to the glass plane instead of receding.
* **Repeated texture** next to a hole: `bgpull`'s mirror produces a visible phase flip;
  `lama` is usually the better choice there and the LPIPS term tends to pick it.
* **Halos**: Depth Pro's object masks extend a pixel or two into the background, so a rim of
  background travels with hair / foliage. `--fg-erode 1..2` shrinks the near regions.
* **Very wide holes** (> ~60 px, i.e. big parallax on a close subject): every backend invents
  content; `lama` looks plausible but smooth, the others streak or tile. Reduce `--parallax` or use
  `--eyes right` so at least one eye is the untouched original.
* `iw3` is a whole-frame backend: with our depth injected (`divergence` = our parallax in % of
  width, `convergence` = zero plane / 255) its eyes land within 0.2 px of ours, so its output is
  directly comparable, but it always synthesizes both eyes (no `--eyes right`) and runs as a
  subprocess that loads its side models each call (a few seconds per image).

## Layout

```
spatialize.py              CLI (batch, idempotent, --json / --debug-dir)
spatialize/
  imio.py                  display-oriented loading, EXIF/XMP carry-over, stereolenses XMP tags
  depth.py                 depthgen lookup/run, conditioning, parallax parsing, convergence, shift field
  warp.py                  z-buffered forward splat, crack/debris cleanup, borders, reprojection
  holes.py                 hole geometry (flanks, background side, clusters, background strip)
  infill/                  fill() interface: stretch, bgpull, opencv_inpaint, lama, iw3
  select.py                hole stats, cheap rules, photo/seam/LPIPS scoring
  pipeline.py              spatialize_image(): the whole thing for one image
  heic.py                  spatialPhotoTool + exiftool tagging
  stereolenses.exiftool.cfg
eval/eval_pairs.py         real-pair benchmark
bin/spatialize             wrapper using the venv (call this from build scripts)
setup.sh, requirements.txt weights/ (downloaded, ignored), vendor/nunif (optional, ignored)
```
