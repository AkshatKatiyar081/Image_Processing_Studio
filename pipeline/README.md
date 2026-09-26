# IVP Enhancement & Diagnostic Studio — pipeline

Image-processing pipeline for the IVP mini project (NMIMS MPSTME Shirpur, 702EX0E020).
It implements the four course modules, a diagnostic "quality gate" that runs the operator
directly and after a textbook prerequisite fix (Direct vs Guided), and a one-click
auto-enhance. Everything is pure Python (NumPy + OpenCV). There is no web code: the
dashboard/API layer calls the functions below.

```
pip install -r requirements.txt
python -m pytest -q                      # 131 tests, ~5 s
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
```

| Function | Returns |
|---|---|
| `run_module(img, module, params=None, engine="opencv")` | `input` (the working image; use as the "before" side), `image` (result), `extras` (module-specific arrays: `lut`, `background`, `gx`/`gy`, `edge_map`, `mask`, `raw_mask`, `cutout`), `histograms.before/after` (256 ints), `metrics`, `warnings`, `meta` (resolved params, engine, `time_ms`, `scale`) |
| `run_guided(img, intent, params=None, engine="opencv")` | `gate_passed`, `diagnosis`, `branch_a`, `branch_b` (with `path` and `stages[]` holding each intermediate image), `verdict` (`recommended` / `mixed` / `no_benefit` / `not_needed` plus the `checks` that decided it) |
| `auto_enhance(img, engine="opencv")` | `image`, `steps[]` (module, params, reason), `diagnosis_before/after`, `histograms` |
| `diagnose(img)` | `metrics` (mean, clipping, std, range, noise σ, impulse density, illumination U, Otsu η, sharpness) and `findings[]` (`code`, `severity`, `value`, `threshold`, `message`) |

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
  modules/           the four course modules (each supports engine="opencv"|"numpy")
  diagnostics.py     quality profile + findings
  guided.py          quality gate, prerequisite chain, verify-after-fix
  auto.py            one-click enhance
  metrics.py         noise σ (Immerkær), impulse density, U_bg, IoU/Dice, edge F1, PSNR/SSIM
  synthetic.py       test scenes with exact masks + controlled degradations
tests/               131 tests; every NumPy algorithm is checked against OpenCV / scikit-image
experiments/         calibration, benchmark, example figures (outputs in experiments/results/)
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

## Known limitations (put these in the report)

* Single global thresholds. The texture image *grass* reads as noisy, and photos with large
  sky/ground regions read as unevenly lit. Verify-after-fix turns the second case into a
  `no_benefit` verdict instead of a wrong recommendation.
* The edge check (false edges in flat regions) does not measure lost true edges. The gate threshold
  was set from a reference-based benchmark to compensate.
* The per-channel median can create slight false colours at edges. HE runs on Y only.
* The OpenCV engine's `equalizeHist` uses the (cdf − cdf_min) normalisation, so it differs slightly from
  the textbook formula used by the NumPy engine (tested, and documented in the API).
