# CHANGES — IVP Studio v3 (2026-10-04)

Work done in phases 0 → G. Nothing has been committed; see
[Suggested commits](#suggested-commits) at the end.

## Summary

| | Before | After |
|---|---|---|
| Pipeline tests | 131 | **230** (all pass) |
| Backend tests | 19 | **40** (all pass) |
| Frontend build | ok | ok, no warnings |
| Python | 3.14 venv | **3.11.9** venv (same as Render) |

---

## 0. Set-up

* Python 3.11 was not installed. Installed **3.11.9** with `py install 3.11` and recreated `.venv`
  (`pip install -e ./pipeline -r backend/requirements-dev.txt scipy pillow`). `npm install` in `frontend/`.
* Baseline before any change: **131 pipeline + 19 backend tests passed**.
* **Windows SciPy / Smart App Control:** with the old 3.14 venv, `test_core_vs_opencv.py` could not import
  (a SciPy DLL was blocked). With the fresh 3.11 venv the block did **not** occur: every test file, including
  `test_core_vs_opencv.py` and the new SciPy-based compression tests, ran and passed on Windows. WSL has no
  Linux distribution installed, so no WSL run was made (not needed).

---

## Features added and removed

**Tools tab**
* **Added: Compression** (4th slot, where "Cut out object" was): DCT and Walsh–Hadamard, RGB in → RGB out with
  correct colours. Presets: Light (DCT 25 %), Medium (10 %), Strong (2 %), *Lab: DCT threshold 500*,
  *Lab: Hadamard zonal* (256/2048, L-shape). Result cards: coefficient maps (all / kept / kept positions), error
  map with its amplification, and PSNR, SSIM, kept %, coefficient ratio, energy kept, PNG / JPEG sizes with an
  honest note (the PNG of a result can even be larger than the original).
* **Removed:** "Cut out object" (segmentation) with click-to-select, seed marker, overlay/mask/cut-out views and
  cut-out download; the seed picker in `ParamControl` is gone too. Segmentation stays in the pipeline and in
  Compare. The backend still accepts and scales seeds in `run_module`.
* **Removed presets:** "Fix uneven lighting", "Brighten shadows" (γ 0.6), "Tone down highlights" (γ 1.6). The
  manual gamma slider stays. `contrast` mode `flatten` is hidden from the Tools Advanced controls
  (`HIDDEN_OPTIONS` in `labels.js`) but stays in the pipeline for guided segmentation.

**Image health card**
* `UNEVEN_ILLUMINATION` and `LOW_SEPARABILITY` are hidden (`SEGMENTATION_ONLY_FINDINGS`); the "Flatten lighting"
  fix button is removed. They still appear in Compare → Segment → "Why?". The healthy message now says
  "no noise, exposure or contrast problems" (it used to claim "no lighting problems", which would be misleading
  now that lighting findings are hidden).

**Compare tab:** exactly three comparisons.
* **Segment** and **Find edges** (operator selector kept): verdict → **sweep chart** (accuracy vs degradation,
  Direct orange / Guided blue, "Your image" ringed, hover tooltip) → **difference map** (legend chips, click to
  enlarge with 1× / 2× / 4× zoom) → **paired metric bars** → images (slider / side by side) → guided-route strip
  → "Why?". The sweep loads in the background with its own spinner.
* **Compression** (new): summary banner ("At 10 % of coefficients, DCT is 2.3 dB better than Hadamard" + reason),
  **rate–distortion chart** (PSNR | SSIM toggle, log x, marker at the setting), **energy-compaction chart**,
  DCT-vs-Hadamard slider / side by side, **error heat maps with a linked 3× zoom lens** over both error maps and
  both reconstructions, **coefficient maps** for both methods, **paired bars** (PSNR, SSIM, energy, PNG, JPEG).
  Keep-% slider (0.5–50) with chips 2 / 5 / 10 / 25.
* Sharpen and Equalize were removed from the UI (the backend intents remain).

**New components:** `charts/LineChart.jsx` (measured-width SVG so text stays readable on phones, linear/log x,
legend, end labels with collision avoidance, crosshair tooltip, visually hidden data table), `charts/PairedBars.jsx`,
`ZoomLens.jsx`. Series colours are CSS variables in `index.css` (`--series-direct`, `--series-guided`,
`--series-dct`, `--series-hadamard`). No new dependencies.

**Samples:** added *Coffee* and *Chelsea the cat* (colourful, for compression).

---

## Phase P — fixes, and what happened to each earlier session change

| Session change (`test-1_session_changes.md`) | Decision | What was done |
|---|---|---|
| 1. Layout `w-full` | Kept | — |
| 2a. Browser sends full resolution | **Replaced** (P.2) | Browser shrinks to ≤ 2048 px with **nearest-neighbour** (`imageSmoothingEnabled = false`) and uploads PNG; ≤ 2048 px PNG/JPEG/WebP/BMP go unchanged. The image card says "shrunk for upload from W × H". |
| 2b/2c. Impulse density before `prepare()`, Auto Fix only | **Replaced** (P.3) | One impulse-aware `prepare()` for every entry point; `pre_scale_impulse_density` removed from `auto.py` and `__init__.py`. |
| 3. Light Gaussian (σ 0.6) for moderate noise | Kept | Verified in the browser: Auto Fix lists "Gaussian blur σ = 0.6 — because: moderate sensor noise σ ≈ 7.9 (light smoothing)". |
| 4. Equalize gate without the dark condition | Kept + fixed text (P.5) | "in a dark image" is only added when the image really is dark. |
| 5. Warning for A = 0 on colour | Kept | — |
| 6. Warning for an all-zero edge response | Kept | Now covered by a test. |
| 7. `fill_holes` uses the user's connectivity | **Replaced** (P.4) | The background flood now uses the **complementary** connectivity (object 8 → background 4, and vice versa), in both engines. Otsu masks are treated as 8-connected. |
| 8. Adaptive region growing marks rejected pixels visited | **Reverted** (P.6) | A pixel is marked visited only when accepted, so it is re-tested as the region mean moves. Docstring explains why. |
| 9. 16-bit images `>> 8` | Kept | Now covered by a test. |
| 10. Editable install | Kept | — |

**P.2 server side (`backend/app/codec.py`):** the header is read with Pillow (lazy, no pixel decode). Images over
`MAX_PIXELS` (env var, default 25 000 000) are refused with `TOO_LARGE` ("Image is 6000 × 5000 (30.0 MP). The
limit is 25 MP. Please resize it and try again.") **before** decoding. The file is decoded **once** with a flag
chosen from the header mode (alpha → UNCHANGED, 16-bit gray → UNCHANGED, 8-bit gray → GRAYSCALE, else COLOR,
which applies EXIF orientation); BGR→RGB is done in place. `pillow` added to `backend/requirements.txt`.

**P.3 (`pipeline/ivp_pipeline/utils.py`):** `prepare_with_info()` estimates impulse density on a
nearest-neighbour copy (≤ 2048 px) and shrinks with `INTER_NEAREST` when specks are present, otherwise
`INTER_AREA`. `meta.resize_interp` (`none` / `area` / `nearest`) and `meta.impulse_density_full_res` are in every
result. `prepare()` keeps its two-value signature. `metrics.impulse_density` now works on uint8 with
`cv2.absdiff` (no int16 copies); a test proves the result is identical to the old formula.
*Verified:* a 2048 px scene with 5 % specks → health card shows **Salt & pepper**, Compare → Segment starts with
**Median filter**, Auto Fix applies a median first.

**P.7 (`backend/app/main.py`):** unexpected exceptions are caught by an HTTP middleware registered **before**
`CORSMiddleware` (so it sits inside it). A 500 now carries `access-control-allow-origin` and the
`{"error": {"code": "INTERNAL", …}}` body. Also added: FastAPI's own validation errors (e.g. a missing form
field) now use the same error shape (`BAD_REQUEST`), and `jsonable()` turns NumPy ±inf/NaN into `null` (PSNR of a
lossless result would otherwise produce invalid JSON).

**P.8 housekeeping:**
* `Media.jpg` and `test_salt_pepper.jpg` (repo root) are deleted; you confirmed they are not needed (the test
  data lives in `dataset/`). The deletion is not committed yet.
* `generate_synthetic_dataset.py` moved to `pipeline/experiments/`; it and `run_e2e_tests.py` use paths relative
  to the script (`Path(__file__)`) plus a `--dataset PATH` option. Both were run from another folder against a
  copy of `dataset/`: the generator wrote all 8 files, and the e2e script printed 12 × PASS.
* `dataset/` is **11 MB** and only used by those two scripts (not by pytest), so it is now in `.gitignore`.

**P.9:** "Click to select picks the wrong spot on large images" — not fixed separately; the feature is removed
from Tools and nothing in Tools uses seed picking any more.

---

## New endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/sweep` (`file`, `intent` = segment / edges, `params`, `engine`) | accuracy vs degradation, Direct vs Guided (numbers only) |
| `POST /api/compare-compression` (`file`, `keep_percent`, `engine`) | DCT vs Hadamard images (PNG), maps, metrics, `rd_curve` (7 points), `energy_curve` (40 points), `summary` |
| `/api/process` | now serves `module="compress"`; new image extras `coef_before`, `coef_after`, `kept_mask`, `error_map` (PNG) |
| `/api/guided` | adds `difference_map` (PNG), `difference_legend`, `comparison` when Branch B ran |

## New pipeline code

* `core/transforms.py`: orthonormal DCT-II (cached matrices), sequency-ordered Walsh matrix (Sylvester →
  `bitreverse(gray(k))` → 1/√n), in-place O(N log N) butterfly (used from 64 px), per-dimension power-of-two
  mirror padding, `keep_percent` (O(n) via `np.partition`, deterministic ties, DC always kept), `threshold`,
  `zonal` (square / lab L-shape), `energy_compaction`.
* `modules/compress.py` (registered in `registry.py` as `compress`, and in `pipeline.RUNNERS`).
* `compare_compression.py`, `sweep.py`; `guided.difference()` for the difference maps.
* `metrics.ssim` re-implemented with an OpenCV box filter (same formula as scikit-image's default, agrees to
  ~1e-7, much faster), and `metrics.psnr` returns `inf` explicitly for identical images (no divide warning).
  A full DCT-vs-Hadamard comparison of a 1024 px colour image went from 10.6 s to 2.9 s.

## Tests

**Pipeline: 131 → 230** (+99)

| File | Tests | What |
|---|---|---|
| `test_compression.py` (new) | 61 | DCT vs SciPy (odd/even), cv2.dct vs NumPy, Walsh orthonormality + sign changes, butterfly = matrix, MATLAB `fwht` doc example, Parseval, padding, selection rules (exact count, ties, lab threshold, zonal shapes), energy curves, perfect reconstruction, **colour correctness** + a test showing the lab `rescale()` bug shifts channel means, shapes for gray/RGB/odd sizes, DCT ≥ Hadamard on camera/astronaut/coffee at 5 and 10 %, engines agree, bad params, `compare_compression` |
| `test_sweep.py` (new) | 16 | axis choice, guided ≥ direct, user point = diagnosis value (also at 1024 px), operator changes the chart, difference-map colour counts = `only_a/only_b`, no map without Branch B, runtime guard (`@pytest.mark.slow`) |
| `test_prepare.py` (new) | 12 | P.3 on a 2048 px speckled scene (diagnose / guided / auto / run_module), clean photo → `area`, small → `none`, low-memory impulse formula = old formula |
| `test_bug_report.py` (new) | 5 | Phase F rows 3, 5, 10, 11 |
| `test_core_vs_opencv.py` | 66 → 68 | complementary fill-holes rule; adaptive region growing re-tests a rejected pixel |
| `test_modules.py` | 42 → 45 | diagonal-ring fill holes in both engines; module list includes `compress` |
| `test_guided.py` | 22 → 23 | equalize reason mentions "dark" only for dark images |

**Backend: 19 → 40** (+21): pixel cap without decoding (`cv2.imdecode` mocked to fail), 24 MP accepted, RGBA,
16-bit gray, EXIF orientation 6, 500 with CORS headers, `compress` (DCT and Hadamard), lossless PSNR → `null`,
`/api/sweep` (both intents), `/api/compare-compression`, guided `difference_map`, error cases (bad intent,
bad/zero/250 `keep_percent`, missing field), new samples, and "keep editing" a gray A = 0 result.

Windows limitation: none this time (see Set-up). WSL: not run (no distribution installed; not needed).

---

## Phase F — earlier bug report, final status

| # | Location | Claim | Final status |
|---|---|---|---|
| 1 | `prepare()` / `auto.py` | Downscaling averages salt & pepper away on large images | **CONFIRMED-FIXED** (P.2 + P.3; `test_prepare.py`, browser check) |
| 2 | `auto.py` | σ in (5, 8] only skipped sharpening | **KEPT** (author's σ 0.6 Gaussian); its step and reason appear in "What was done" (checked in the browser) |
| 3 | `auto.py` | Exposure/contrast checks use pre-fix metrics | **NOT-A-BUG**: `apply()` re-diagnoses after every step. Test: dark + 5 % specks → the gamma reason reports the post-median mean |
| 4 | `guided.py` | Equalize gate needed noisy **and** dark | **KEPT** (session removed the dark condition) + reason text fixed (P.5). Equalize also left the Compare UI |
| 5 | `guided.py` | Flatten radius capped at 75 for images > 750 px | **NOT-A-BUG** for the tested case: 1024 px scene, 12 disks of radius 40–60 px, ramp ±40 → Branch B IoU **0.9997** (Direct 0.18) with the cap in effect. Not changed |
| 8 | `denoise.py` | `kernel_size` ignored for Gaussian | **NOT-A-BUG**: size is derived from σ; the registry `when` hides the control |
| 9 | `sharpen_edges.py` | A = 0 on colour returns gray | **KEPT** (warning). Verified: sending the gray PNG back ("keep editing") works (backend test) |
| 10 | `sharpen_edges.py` | Constant image → black edges, no warning | **CONFIRMED-FIXED** (session change 6); now covered by a test (gradient and A = 0) |
| 11 | `segment.py` | OpenCV region grow returns a 1-px mask if the seed is a spike | **NOT-A-BUG**: 3×3-median seed value + seed forced into the region; test passes for both engines |
| 12 | `segment.py` | `fill_holes` connectivity | **CONFIRMED-FIXED** (P.4) |
| 14 | `backend/app/codec.py` | 16-bit min-max normalisation | **CONFIRMED-FIXED** (session change 9); covered by a backend test |
| 16 | `core/region.py` | Adaptive BFS re-tests rejected neighbours | **NOT-A-BUG** (re-testing is the correct adaptive predicate); session change reverted in P.6, documented in the docstring |

---

## Memory measurement (Phase G.5)

Peak extra working-set memory of **one request** in a fresh Python 3.11 process (Windows, `GetProcessMemoryInfo`),
after a warm-up call; the test JPEG was generated in a separate process so its construction does not count.
Baseline RSS of the app ≈ 90 MB.

| Upload | Endpoint | Peak additional memory |
|---|---|---|
| 24 MP JPEG (6000 × 4000), sent directly to the API | `/api/process` denoise | **+144 MB** |
| 24 MP | `/api/process` compress | +152 MB |
| 24 MP | `/api/compare-compression` | +165 MB |
| 24 MP | `/api/guided` edges / `/api/sweep` edges | +145 MB / +145 MB |
| 3 MP (2048 × 1536, what the browser now sends) | `/api/process` denoise | +32 MB |
| 3 MP | `/api/process` compress | +99 MB |
| 3 MP | `/api/compare-compression` | +111 MB |
| 3 MP | `/api/guided` / `/api/sweep` | +45 MB / +31 MB |

All are well under the 250 MB target. (`compress` first used +179 MB at 24 MP and +133 MB at 3 MP; it now
folds each channel into running sums instead of keeping every channel's coefficient arrays.) Note that Linux on
Render will show somewhat different absolute numbers; the relative picture should hold.

---

## Browser check (Phase G.4)

Done in the app's built-in browser against `uvicorn` (:8000) and `npm run dev` (:5173). Playwright was not used.

* Tools shows exactly four tools; contrast has three presets; no cut-out, flatten, brighten or darken.
* Coffee → Compression → Medium and *Lab: Hadamard zonal*: correct colours, coefficient maps, error map, metrics,
  the padding warning ("padded from 600×400 to 1024×512"); "Keep editing this result" and Undo work.
* Health card on "Shapes, uneven light": no "Uneven lighting" / "not clearly separated" entries.
* Compare shows three options. Segment on "Shapes, uneven light": verdict (η 0.696 → 0.969), sweep chart
  (guided 1.0 above direct ≤ 0.18), difference map, bars. Find edges on "Cameraman, noisy": all elements;
  Sobel → Roberts changes the chart (direct F1 0.65 → 0.43 at the user's point). Compression on Coffee: banner
  (DCT +2.3 dB), RD chart, energy chart, slider, error maps with a working zoom lens (verified by dispatching a
  pointer event; Hadamard's blockiness is visible in the lens), coefficient maps, bars.
* Auto Fix on "Astronaut, too dark": gamma step, image brighter.
* 390 px width: no horizontal overflow (found and fixed: the hidden chart data tables made the page 666 px
  wide; a `<table>` ignores `width: 1px`, so it is now wrapped in an `sr-only` div). Charts stay readable.
* Console: no errors apart from the deliberate 422 of the over-limit test.
* Phase P checks: a 4032 × 3024 image → card says "2048 × 1536 px (shrunk for upload from 4032 × 3024)", and the
  request body is a 2048 × 1536 PNG. A 4032 × 3024 image with 5 % specks → uploaded PNG still has exactly 5.00 %
  specks, health card shows Salt & pepper, `resize_interp = nearest`, guided route starts with Median filter.
  A 30 MP PNG sent to the API → friendly "too large" message, server stays up. (Through the normal UI a large
  image can't reach the cap, because the browser shrinks it first.)
* Forced 500 with CORS: verified by the automated backend test (`test_server_error_keeps_cors_headers`), not
  by crashing the live server.

---

## Deferred / notes

* The calibration and benchmark scripts in `pipeline/experiments/` were not re-run. They now use the new
  `metrics.psnr` / `metrics.ssim` (same formulas; SSIM agrees with scikit-image to ~1e-7), so results should not move.
* The RD and energy curves are computed on a ≤ 512 px copy, while the banner and bars use the full working
  image, so the curve's value at "your setting" can differ slightly from the banner (captioned in the UI).
* In the segment difference map, "both say object" and "both say background" look the same (dimmed input), as
  specified. A third colour for "both object" would make it easier to read; not done.
* `backend` tests show one pre-existing deprecation warning (Starlette's TestClient wants `httpx2`).
* The desktop preview launcher read another project's `launch.json`, so the dev servers were started as
  background processes for the browser check.

---

## Follow-up: full resolution locally + label fix

* **No shrinking by default.** `IVP_MAX_SIDE` (backend env, read in `config.py`) defaults to `0` = full
  resolution; `VITE_MAX_UPLOAD_SIDE` (frontend build env) defaults to `0` = upload the full file. The hosted
  setup keeps its limits: `render.yaml` sets `IVP_MAX_SIDE=1024`, and the README tells Vercel to set
  `VITE_MAX_UPLOAD_SIDE=2048`. The test suites pin `IVP_MAX_SIDE=1024` in their `conftest.py` (they test the
  downscaling path); two new tests check that `max_side=0` and the library default keep full resolution.
* **Faster Hadamard for big images.** A 3072 × 2304 photo is padded to 4096 × 4096. The 2-D transform now uses
  the Kronecker factorisation H₄₀₉₆ = H₆₄ ⊗ H₆₄ (two small matrix products per row, run by BLAS) in float32, with
  row-wise reordering. Same result (≈ 1e-13 against the matrix). Hadamard compression of the flower: 25.6 s → 7.6 s;
  Compare → Compression: 33 s → 12 s. The 1-D butterfly is kept and still tested.
* **Smaller responses.** Coefficient maps and the kept mask are display images, now capped at 1024 px
  (`COEF_MAP_MAX_SIDE`); error maps and results stay full resolution (the zoom lens needs them).
  Compare → Compression on the flower: 108 MB → 78 MB of JSON.
* Timings at full resolution (flower, 3072 × 2304, this PC): diagnose 1 s, DCT compression ≈ 4 s, Hadamard ≈ 8 s,
  guided segment 3 s, guided edges 1 s, sweep 0.6 s, Auto Fix 3 s. The NumPy-engine DCT (matrix product) takes
  ≈ 20 s at this size; OpenCV's `cv2.dct` is the default.
* **Label fix:** in Tools → Compression → Advanced, `method` and `mode` were both labelled "Method". They now read
  **Transform** and **Coefficient selection** (`TOOL_PARAM_LABELS` in `labels.js`).
* Tests: pipeline 232, backend 40, all passing; build clean. Checked in the browser: a 3072 × 2304 upload stays
  3072 × 2304 through compression, and the new labels show.

## Follow-up: JPEG-style compression, real downloadable files, generic presets

* New `pipeline/ivp_pipeline/core/jpeg.py`: the JPEG compression steps written from scratch: YCbCr, chroma
  subsampling, 8×8 block DCT, quality-scaled standard quantisation tables, smooth chroma upsampling like
  libjpeg's. It matches a real libjpeg decode at 44–52 dB. `compress` gains `method="jpeg"` with
  `quality` and `chroma_subsampling`.
* Backend: `POST /api/inspect` reports format, file size, pixel size, bits per pixel, JPEG quality estimate,
  subsampling and metadata. `POST /api/compress` returns a real file: the JPEG-style method gives a JPEG; whole-image
  DCT and Hadamard give the upload's own format (a JPEG at its own quality). EXIF and colour profile are kept.
  PSNR, SSIM and the error map describe the downloaded file.
* Frontend: an image-details card is shown before a method is picked. After compressing it shows your file vs the
  compressed file and the size change. The download is the real file with the right extension. Presets are
  generic (High quality / Balanced / Small / Smallest + whole-image DCT and Walsh–Hadamard demos); the lab and
  "Exp NN" references are removed. The zonal option `lab` is renamed `l_shape`.
* Flower (2.72 MB JPEG, q≈94): quality 75 → 1.25 MB (44 %), quality 50 → 0.83 MB (29 %).
* Tests: pipeline 263, backend 46, all passing; frontend build clean. **Not yet checked in the browser.**
  Not updated yet: the README sections on compression.

## Follow-up: "Quality vs file size" in Compare → Compression

* `compare_compression` now also returns `size_curve`: real JPEG files from JPEG-style coding at qualities
  10–95, and the whole-image DCT / Hadamard results (each keep-% level, plus the user's own setting) saved as
  near-lossless JPEGs (quality 95, 4:4:4). Every point is file size in bits per pixel against PSNR / SSIM of the
  decoded file, on the ≤ 512 px curve copy. `summary.jpeg_vs_dct` gives how many times fewer bytes JPEG-style
  needs for the quality of whole-image DCT at the user's setting.
* New card in the Compare tab with a PSNR / SSIM toggle and a green JPEG-style series (`--series-jpeg`); the
  summary banner adds the bytes comparison. `LineChart` gained an `endLabels` option.
* Coffee at 10 %: JPEG-style needs **5.3× fewer bytes** than whole-image DCT for the same quality (0.88 vs 4.69
  bits per pixel); the flower copy shows 5.0×.
* Tests: pipeline 265, backend 46, all passing; build clean; checked in the browser (no console errors).

## Follow-up: JPEG-style as a full third method in Compare → Compression

* `compare_compression` also runs JPEG-style coding at the **lowest quality whose decoded file matches the
  DCT result's PSNR** (`match_jpeg_quality`, binary search), with a shared error-map scale for all three. Each
  method gets `file_bytes` (DCT / Hadamard results saved as near-lossless JPEGs; JPEG-style = its own file).
  `summary.jpeg` = {quality, psnr_db, file_bytes, dct_file_bytes, times_smaller_than_dct}.
* UI: the banner states the JPEG result; "Quality vs file size" moved directly under it; image views
  "DCT vs Hadamard" / "DCT vs JPEG-style" / "Side by side" (three images, with sizes); the zoom lens covers all three
  error maps and results; JPEG-style coefficient maps; three-way bars (file size, PSNR, SSIM, non-zero
  coefficients, energy). `PairedBars` now takes any number of series.
* Coffee, 10 %: JPEG-style q41 = 30.0 dB in **23.6 KB** vs DCT 148.4 KB (6.3× smaller), with fewer non-zero
  coefficients (9.5 % vs 10 %) and higher SSIM (0.856 vs 0.808).
* Tests: pipeline 267, backend 46 (the compare test covers the third method). Checked in the browser.
* **JPEG-style quality slider** under the keep-% slider (Compare sidebar): "Auto: match DCT's image quality"
  (default; shows the chosen quality, e.g. "auto (41)"), or a manual quality 1–100 with chips q25 / q50 / q75 / q90
  that re-run the comparison. API: `jpeg_quality` form field (`auto` or 1–100) on `/api/compare-compression`,
  `jpeg_quality=` argument on `ivp.compare_compression`; `summary.jpeg.matched_to_dct` tells the UI which banner
  wording to use. Coffee at 10 %: q90 = 35.5 dB in 70.6 KB vs DCT 30.0 dB in 148.4 KB (smaller *and* better);
  q25 = 8.7× smaller but 28.7 dB. Tests: pipeline 268, backend 47.

## Follow-up: Side-by-side view in Tools and Auto Fix

* New `BeforeAfterView.jsx` (with `ViewToggle`): "Slider | Side by side" for a before/after result. Used by the
  Tools tab (every tool, including Compression) and the Auto Fix tab; the switch appears once there is a result.
  Side by side shows both images at the same size with their labels; the busy overlay covers both.
* Checked in the browser (Tools → Auto brightness, Auto Fix on "Astronaut, too dark"); no console errors.

## Follow-up: two independently scrolling columns on desktop

* On screens ≥ 1024 px the app is exactly one screen high: the header stays put and the area below is split into
  a left column (image, health, tool controls) and a right column (results), each scrolling on its own with a
  hidden scrollbar (`.scroll-hidden` in `index.css`). The mouse wheel moves the column under the cursor; the page
  itself does not scroll. The footer moves to the bottom of the right column. "Keep editing this result" scrolls
  both columns back to the top. All three tabs use it (shared `Layout` in `App.jsx`, `useIsDesktop` hook).
* Phones / narrow windows keep the normal single scrolling page (image + health, results, controls).
* Checked in the browser at 1440 × 900 (wheel over the left column scrolls only the left, over the right only the
  right, page never scrolls) and at 390 px (normal page, no sideways overflow); no console errors.

## Follow-up: Vercel size limits, full-quality downloads, noise in screenshots (2026-10-08)

**Problem 1 — replies too big for Vercel (4.5 MB per request and per reply).** Measured on a 12 MP photo
(flower.jpg, 3072 × 2304) at full resolution, the replies were 6–24 MB (Tools, Auto Fix), 59 MB (Compare →
Segment) and 113 MB (Compare → Compression), because every image went back as a full-size base64 PNG.

* Replies now carry **previews**: ≤ 1280 px JPEG 90 for photos, PNG for 0/255 masks and edge maps, 512 px
  coefficient maps, error maps at JPEG 80; Compare → Compression at ≤ 1024 px JPEG 92. Removed the duplicate
  `input` image and the unused full-size RGBA `cutout`. Measured again on the same photo: Tools 0.7 MB,
  compression 1.1–1.2 MB, Auto Fix 0.6 MB, Compare → Segment 2.2 MB, Edges 0.2 MB, Compression 2.6 MB.
* **Full-quality results are separate files** (`/api/process/file`, `/auto/file`, `/guided/file`,
  `/compare-compression/file`, `/compress/file`), fetched only for *Download* and *Keep editing this result*,
  so chained edits never work on a lossy preview. PNG when it fits the download limit, otherwise JPEG 95 → 80
  with a note shown to the user ("Saved as JPEG (quality 95) because the PNG file would be 11.3 MB…").
  Compare's JPEG-style download is now the real JPEG file the comparison measured.
* **Limits** in `backend/app/limits.py`: 4.2 MB on Vercel (detected automatically), 15 MB elsewhere; the
  download limit defaults to the upload limit, so every returned file can be uploaded again. `/api/health`
  reports them; the upload box shows "Max 4.2 MB · 25 MP" and refuses bigger files with a clear message
  directly under the drop zone.
* The browser-side shrink (`VITE_MAX_UPLOAD_SIDE`) keeps a JPEG as JPEG 95 instead of re-saving it as a PNG
  several times bigger; the README's "set it to 2048" advice for Vercel was wrong and is corrected.

**Problem 2 — noise in screenshots was not detected.** A screenshot of a noisy Google image (salt & pepper,
enlarged by the browser) was reported as healthy: noise σ 1.7, specks 0.02 %. Enlarging smears each speck over
2–3 px and softens it to grey, so both pixel-scale tests miss it, and Auto Fix even sharpened it. New half-scale
measures (pipeline README, refinement 11) find it (specks 4.4 %), the fix uses a 5×5 median (3×3 is too
small), and Auto Fix never sharpens when noise is found at any scale. The Health card names enlarged noise,
offers the matching filter, and shows "Still looks noisy to you? Remove specks · Smooth grain" when nothing is
found. New presets: 7×7 median, Gaussian σ 1.5. No sample or dataset image gained a false alarm.

**Tests:** pipeline +26 (`test_noise_scales.py`: enlarged specks/grain found and fixed, clean images and clean
screenshots of 6 photos stay clean at 1×, 1.5×, 2×); API 47 → 60 (previews, `/file` endpoints and identical
pixels, JPEG fallback, limits, CORS headers, and a 12 MP photo through every endpoint with every reply
< 3.5 MB). Checked in the browser with the real photo and screenshot: limits message, previews, downloads,
chaining, all Compare views. Note: Windows Smart App Control now blocks one scikit-image DLL, so
`tests/test_core_vs_opencv.py` (68 tests) cannot be collected on this machine; it is unchanged.

## Suggested commits

1. `Remove unused root images (Media.jpg, test_salt_pepper.jpg); ignore dataset/; move dataset scripts to pipeline/experiments with relative paths`
2. `Fix fill_holes connectivity (complementary rule) and revert adaptive region-growing visited change`
3. `Impulse-aware prepare() for every entry point; low-memory impulse_density; fix equalize reason text`
4. `Backend: pixel cap before decoding, single decode, 500s keep CORS headers, JSON-safe infinities`
5. `Frontend: nearest-neighbour shrink to 2048 px before upload`
6. `Pipeline: DCT and Walsh-Hadamard transforms, compress module, compare_compression; fast SSIM/PSNR`
7. `Pipeline: degradation sweep and guided difference maps`
8. `Backend: /api/sweep, /api/compare-compression, guided difference map, coffee/chelsea samples`
9. `Frontend: Compression tool, Tools cleanup, health-card filter, Compare rewrite with charts and zoom lens`
10. `Docs: README, pipeline README, CHANGES.md`

(Or one commit: `IVP Studio v3: compression transforms, Compare charts, Phase P fixes`.)
