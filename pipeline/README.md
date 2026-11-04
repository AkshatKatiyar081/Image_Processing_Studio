# Image Processing Studio — pipeline

Image-processing pipeline for the IVP mini project (NMIMS MPSTME Shirpur, 702EX0E020).
It implements the four course modules, a diagnostic "quality gate" that runs the operator
directly and after a textbook prerequisite fix (Direct vs Guided), and a one-click
auto-enhance. Everything is pure Python (NumPy + OpenCV). There is no web code: the
dashboard/API layer calls the functions below.

```
pip install -r requirements.txt
python -m pytest -q                      # 294 tests, ~1 min (-m "not slow" skips the runtime guard)
python experiments/calibrate_gate.py     # threshold calibration  -> experiments/results/calibration.md
python experiments/run_benchmark.py      # Direct vs Guided grid  -> results/benchmark.md, direct_vs_guided.png, runtime.md
python experiments/make_examples.py      # visual examples        -> results/examples.png
```

## Public API (for the dashboard / backend)

```python
import cv2, numpy as np
import ivp_pipeline as ivp

bgr = cv2.imdecode(np.frombuffer(upload_bytes, np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # the pipeline speaks RGB

ivp.list_modules()                                   # param schemas -> build the UI controls from this
ivp.diagnose(rgb)                                    # {"metrics", "findings", "meta"}
ivp.run_module(rgb, "denoise", {"filter_type": "median", "kernel_size": 5})
ivp.run_module(rgb, "segment", {"method": "region_growing", "seed": [x, y]})
ivp.run_guided(rgb, "segment")                       # intent: segment | edges | sharpen | equalize
ivp.auto_enhance(rgb)
ivp.run_module(rgb, "compress", {"method": "dct", "mode": "keep_percent", "keep_percent": 10})
ivp.compare_compression(rgb, keep_percent=10)        # DCT vs Walsh–Hadamard, same budget
ivp.degradation_sweep(rgb, "edges", {"operator": "sobel"})   # accuracy vs added noise, Direct vs Guided
```

| Function | Returns |
|---|---|
| `run_module(img, module, params=None, engine="opencv")` | `input` (the working image; use as the "before" side), `image` (result), `extras` (module-specific arrays: `lut`, `background`, `gx`/`gy`, `edge_map`, `mask`, `raw_mask`, `cutout`), `histograms.before/after` (256 ints), `metrics`, `warnings`, `meta` (resolved params, engine, `time_ms`, `scale`) |
| `run_guided(img, intent, params=None, engine="opencv")` | `gate_passed`, `diagnosis`, `branch_a`, `branch_b` (with `path` and `stages[]` holding each intermediate image), `verdict` (`recommended` / `mixed` / `no_benefit` / `not_needed` plus the `checks` that decided it) |
| `auto_enhance(img, engine="opencv")` | `image`, `steps[]` (module, params, reason), `diagnosis_before/after`, `histograms` |
| `diagnose(img)` | `metrics` (mean, clipping, std, range, noise σ, impulse density, half-scale noise σ and speck density, illumination U, Otsu η, sharpness) and `findings[]` (`code`, `severity`, `value`, `threshold`, `message`; noise findings also `scale` = `pixel` / `enlarged`, specks `median_ksize`) |
| `compare_compression(img, keep_percent=10, engine="opencv")` | `dct`, `hadamard` (run_module-style results with `coef_before`, `coef_after`, `kept_mask`, `error_map`), `rd_curve` (PSNR/SSIM at 0.5–50 %), `energy_curve` (40 points), `curve_scale`, `summary` (`winner`, `psnr_margin_db`) |
| `degradation_sweep(img, intent, params=None, engine="opencv")` | `axis` (illumination / impulse / gaussian, chosen from the diagnosis), `metric` (IoU or Edge F1), `points[{level, x, direct, guided, gate_fired}]`, `user_point_index`, `reference_note`, `sweep_scale` |

Every entry point also puts `meta.resize_interp` (`none` / `area` / `nearest`) and
`meta.impulse_density_full_res` in its result (see *Impulse-aware downscaling* below).
`run_guided` adds `difference_map`, `difference_legend` and `comparison` (counts for bar charts) when Branch B ran
(segment and edges intents).

Errors raise `ivp.PipelineError` with a stable `.code` (`BAD_IMAGE`, `BAD_PARAM`, `UNKNOWN_MODULE`,
`UNKNOWN_INTENT`, `BAD_ENGINE`) and a readable `.message`. Map them to HTTP 422.

### Integration notes for the teammate

* **Images:** input is uint8 RGB, RGBA (alpha dropped) or grayscale. Images larger than 1024 px on
  the long side are downscaled (`meta.scale`). Always show `result["input"]` next to
  `result["image"]` so both sides have the same size.
* **Seeds:** send the region-growing `seed` as `[x, y]` = (column, row) **in original image pixels**.
  The pipeline applies `meta.scale` itself.
* **Encoding:** masks, edge maps and `extras.cutout` (RGBA) must be PNG. Photos can be JPEG.
  Some outputs are single-channel (edges, masks); encode them as grayscale PNG.
* **Form data:** string values such as `"5"` and `"true"` are coerced, so multipart form fields work
  as they are.
* **Engines:** `engine="opencv"` (default, fast) or `engine="numpy"` (the from-scratch
  implementations). A "show from-scratch implementation" toggle in the UI is a nice touch for the viva.
* A minimal FastAPI wrapper looks like this (your code, not part of this package):

```python
@app.post("/api/process-direct")
async def process_direct(file: UploadFile, module: str = Form(...), config: str = Form("{}")):
    rgb = decode_rgb(await file.read())
    try:
        res = ivp.run_module(rgb, module, json.loads(config))
    except ivp.PipelineError as e:
        raise HTTPException(422, e.to_dict())
    return {"input": to_b64(res["input"]), "image": to_b64(res["image"], png=True),
            "histograms": res["histograms"], "metrics": res["metrics"],
            "warnings": res["warnings"], "meta": res["meta"]}
```

## Layout

```
ivp_pipeline/
  __init__.py        public API
  config.py          every threshold/default (calibrated values documented inline)
  registry.py        module parameter schemas + validation
  core/              from-scratch NumPy engine
    filtering.py     correlation, box, separable Gaussian, median
    histogram.py     histogram equalization, Otsu (+ separability η)
    morphology.py    disk SE, erosion/dilation/opening/closing, hole filling
    region.py        BFS region growing (fixed and adaptive predicates)
    background.py    morphological background estimate, auto polarity
    transforms.py    orthonormal DCT-II and sequency-ordered Walsh–Hadamard (matrix + fast butterfly),
                     power-of-two padding, coefficient selection, energy compaction
  modules/           the course modules (each supports engine="opencv"|"numpy"):
                     contrast, denoise, sharpen_edges, segment, compress
  diagnostics.py     quality profile + findings
  guided.py          quality gate, prerequisite chain, verify-after-fix, difference maps
  sweep.py           degradation sweep (accuracy vs defect strength, Direct vs Guided)
  compare_compression.py  DCT vs Hadamard at the same coefficient budget + RD / energy curves
  auto.py            one-click enhance
  metrics.py         noise σ (Immerkær), impulse density, U_bg, IoU/Dice, edge F1, PSNR, SSIM (OpenCV box filter)
  synthetic.py       test scenes with exact masks + controlled degradations
tests/               294 tests; every NumPy algorithm is checked against OpenCV / scikit-image / SciPy
experiments/         calibration, benchmark, example figures (outputs in experiments/results/),
                     generate_synthetic_dataset.py and run_e2e_tests.py (use <repo>/dataset, git-ignored)
```

## Where the implementation refines spec v2

These came out of testing and the benchmark. The spec's §10 table should be updated to match.

1. **Noise gates are per intent, set from the benchmark:** Gaussian pre-smoothing is prescribed at
   σₙ > 8 for segment, sharpen and equalize. For edges it is σₙ > 15 for Sobel/Prewitt and σₙ > 12 for
   Roberts. Below those levels, smoothing cost more detail than the noise did (e.g. sharpening at
   σ = 5 lost 2.6 dB with the old σₙ > 5 gate). Sobel's own 3×3 smoothing is why it tolerates more
   noise than Roberts, which is a nice point for the report.
2. **Impulse threshold is 0.75 %** (calibrated; F1 0.971 → 0.993).
3. **Auto polarity removes a fitted plane before Otsu.** Without this, 2 of 10 scenes with a ±70 ramp
   picked the wrong polarity (IoU ≈ 0); with it, 100/100 are correct.
4. **Region-growing seed value is the 3×3 median** (not the mean), so a salt/pepper spike at the
   click can't poison the reference.
5. **Segmentation verify-after-fix uses fragmentation** (islands + holes), which also works for region
   growing. `UNEVEN_ILLUMINATION` is reported as `info`, because natural scenes trigger it too.
   Guided mode settles those cases with η.
6. **Segment `image` is the overlay** (RGB), which is better for display. The RGBA cutout is in
   `extras.cutout`.
7. **Edge F1 uses a 1-pixel match tolerance** (BSDS-style). Pixel-exact F1 unfairly counts 1-px shifts
   as misses.

8. **Impulse-aware downscaling.** Images above 1024 px used to be shrunk with `INTER_AREA`, which averages
   salt-and-pepper specks away before diagnostics see them (a 2048 px image with 5 % specks was reported
   as "grainy noise", and guided segmentation prescribed a Gaussian instead of a median). `prepare()` now
   estimates the impulse density on a nearest-neighbour copy (≤ 2048 px) first, and shrinks with
   `INTER_NEAREST` when specks are present, `INTER_AREA` otherwise. This is used by every entry point.
9. **Hole filling uses the complementary connectivity.** An 8-connected object needs a 4-connected
   background flood (and vice versa); otherwise the background leaks through diagonal gaps and the hole
   is never filled. Otsu masks are treated as 8-connected.
10. **Adaptive region growing re-tests rejected pixels.** The region mean keeps moving, so a pixel rejected
   early may qualify later; a pixel is marked visited only when it is accepted.
11. **Enlarged noise (screenshots) is measured at half scale.** A screenshot of a noisy web image is usually
   enlarged by the browser or display scaling, which smears each grain / speck over 2–3 px: the pixel-scale
   tests then saw *nothing* (0 of 68 noisy screenshots with grain, 0 of 102 with specks). The diagnosis now
   also measures a half-scale copy (every 2nd pixel, no averaging): **specks** = pixels that differ from their
   3×3 median by > 50 and form isolated groups of ≤ 2 px (no pure black/white needed; the size test keeps text
   strokes out), found when > 2.3 % (and pixel σ < 10, so heavy grain is not mistaken for specks); **grain** =
   half-scale σ > 5 and > 3.5 × the pixel σ (fine detail raises both together, enlarged noise only the
   half-scale one). Enlarged specks get a 5×5 median (7×7 above 5 %), enlarged grain a Gaussian σ 1.5, in Auto
   Fix, guided mode and the Health card's fix buttons. Calibration (`calibrate_gate.py`, 391 images incl. 204
   screenshot-like ones): recall 100 % for both, the only new false alarms on grass/gravel textures, which
   already fooled the pixel tests. `experiments/noise_scales_study.py` has the per-scale distributions.

## Compression (transform coding, lab DCT / Walsh–Hadamard)

`modules/compress.py` reproduces the lab scripts (`DCT 1.m`, `single_channel_dct.m`, `hadamard_transform.m`),
in colour and without their problems.

* **Per channel:** float64 on the **0–255 scale** (so the lab threshold 500 means the same as MATLAB's `dct2` on
  uint8) → forward transform → coefficient selection → inverse → `clip(0, 255)` → round → uint8. R, G and B are
  processed independently with the same settings.
* **DCT:** orthonormal DCT-II, `F = C · f · Cᵀ` with `C[k, x] = α(k) cos(π (2x+1) k / 2n)`, α(0) = √(1/n),
  α(k) = √(2/n). Equal to MATLAB `dct2` and `scipy.fft.dctn(norm="ortho")` (tested). The OpenCV engine uses
  `cv2.dct`; OpenCV 4.x only accepts even sizes, so odd sides are then mirror-padded by 1 and cropped back
  (OpenCV 5 handles odd sizes itself).
* **Walsh–Hadamard:** Sylvester matrix `H₂ₘ = [[Hₘ, Hₘ], [Hₘ, −Hₘ]]`, rows reordered into **sequency order**
  (row k = natural row `bitreverse(gray(k))`, so row k has exactly k sign changes), scaled by 1/√n so it is
  orthonormal. MATLAB's `fwht` uses the same order but scales the forward transform by 1/N, i.e.
  `fwht(x) = ours(x) / √N` (tested against the MATLAB documentation example). A fast O(N log N) butterfly is
  used from 64 px up (tested against the matrix). OpenCV has no Walsh–Hadamard transform, so **both engines use
  our NumPy code** for it. Non-power-of-two images are mirror-padded per dimension to the next power of two and
  cropped back; nothing is resized (the lab script's `imresize` to 2048 × 2048 distorted the aspect ratio).
* **Coefficient selection** (`mode`): `keep_percent` keeps the largest |F| (DC always kept, ties broken by
  position); `threshold` zeroes |F| < t (the lab rule, t = 500); `zonal` keeps low frequencies by position,
  `square` (rows < k and cols < k) or `lab` (zero only rows ≥ k **and** cols ≥ k: the L-shaped region that
  `Y(256:r, 256:c) = 0` keeps).
* **The colour bug in `single_channel_dct.m`:** it called `rescale()` on each channel after `idct2`. That
  stretches every channel min → max on its own, so the channel means move (on the astronaut image by up to
  17 gray levels) and the colours shift. We only clip and round: channel means stay within 0.3 levels.
  A test documents both behaviours.
* **Outputs:** `coef_before` / `coef_after` (log(1 + |F|) of the luminance coefficients, one shared scale),
  `kept_mask`, `error_map` (|original − result| averaged over channels, amplified so the 99th percentile is
  bright, inferno colour map), and metrics: kept %, coefficient ratio (a count ratio, not a file-size ratio),
  PSNR, SSIM, energy kept, and PNG / JPEG-q90 byte sizes of the result (indirect: a PNG of the result can even
  be larger than the original, because ringing adds new values).
* **DCT vs Hadamard** (`compare_compression`): both keep the same % of coefficients. The curves are computed on
  a ≤ 512 px copy, transforming each channel once and reusing the coefficients for every level. On natural
  images the DCT wins by about 2–3 dB at 5–10 % (tested on camera, astronaut and coffee): its smooth cosines
  match natural images better than ±1 square waves, so it packs more energy into fewer coefficients.

## Sweeps and difference maps (Compare tab)

* **Degradation sweep** (`sweep.py`): the axis follows the user's defect. Segment: illumination ramp
  [0, 20, 40, 70] if `UNEVEN_ILLUMINATION`, else salt & pepper [0, 1, 5, 10, 20] % if `IMPULSE_NOISE`, else
  Gaussian σ [0, 5, 10, 20, 30]. Edges: Gaussian σ, or salt & pepper if `IMPULSE_NOISE`. Each level is degraded
  with a fixed seed and the guided comparison runs again. Both branches are scored against a **reference**,
  the best available result on the unmodified image (Branch B if the gate fired, else Branch A): IoU for
  segmentation, edge F1 (1-px tolerance) for edges. The x value is the **measured** defect of each degraded
  image, so the user's image (level 0) sits at its true position. Runs on a ≤ 512 px copy shrunk with
  nearest-neighbour, which keeps each pixel's own noise, so x matches the health card. About 0.5 s.
* **Difference map** (`guided.difference`): edges: black = neither, white = both, orange-red = only Direct
  (mostly false edges from noise), green = only Guided. Segment: the input dimmed to 40 %, orange-red = object
  only in Direct, green = object only in Guided. `comparison` holds the counts used by the UI bars
  (edge pixels, false-edge density in flat regions, fragmentation, η, components, only-A / only-B pixels).

## Known limitations (put these in the report)

* Single global thresholds. The texture image *grass* reads as noisy, and photos with large
  sky/ground regions read as unevenly lit. Verify-after-fix turns the second case into a
  `no_benefit` verdict instead of a wrong recommendation.
* The edge check (false edges in flat regions) does not measure lost true edges. The gate threshold
  was set from a reference-based benchmark to compensate.
* The per-channel median can create slight false colours at edges. HE runs on Y only.
* The OpenCV engine's `equalizeHist` uses the (cdf − cdf_min) normalisation, so it differs slightly from
  the textbook formula used by the NumPy engine (tested, and documented in the API).
