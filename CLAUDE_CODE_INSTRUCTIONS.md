# IVP Studio — Change Request for Claude Code

You are working in the **IVP Studio** repository: an Image & Video Processing mini project
(NMIMS MPSTME Shirpur, course 702EX0E020). Read this whole file before changing anything.
Then work through the phases in order. At the end of each phase, run the tests and keep them green.

**Phase order:**

| Phase | Content |
|---|---|
| 0 | Orientation and set-up |
| **P** | **Fix existing issues** (from a code review of the current repo). Do this before anything else. |
| A | Compression transforms + module |
| B | Sweeps and difference maps |
| C | Backend |
| D | Frontend |
| E | Docs |
| F | Remaining bug-report verifications |
| G | Final verification |

**Context:** an earlier working session already modified this repo. Its notes are in
`test-1_session_changes.md` at the repo root (if present). Read it. Phase P below says, change by
change, which of those edits to **keep**, which to **replace**, and which to **revert**.

---

## 0. Orientation (do this first)

### 0.1 Repository layout (expected — verify; the local copy may be nested one level deeper, e.g. `F:\ivp-studio\ivp-studio`)

```
ivp-studio/
├── pipeline/                 Python package `ivp_pipeline` (NumPy + OpenCV) + tests + experiments
│   ├── ivp_pipeline/
│   │   ├── __init__.py       public API: list_modules, diagnose, run_module, run_guided, auto_enhance
│   │   ├── config.py         every threshold/default
│   │   ├── registry.py       module parameter schemas + validation (drives the UI controls)
│   │   ├── pipeline.py       execute()/step() helpers, RUNNERS dict
│   │   ├── diagnostics.py    image quality profile + findings
│   │   ├── guided.py         quality gate, prerequisite chain, verify-after-fix
│   │   ├── auto.py           one-click Auto Fix
│   │   ├── metrics.py        noise σ (Immerkær), impulse density, U_bg, IoU, edge F1, PSNR, SSIM …
│   │   ├── synthetic.py      synthetic scenes + degradations
│   │   ├── utils.py          prepare(), colour helpers, histograms
│   │   ├── core/             from-scratch NumPy engine (filtering, histogram, morphology, region, background)
│   │   └── modules/          contrast.py, denoise.py, sharpen_edges.py, segment.py
│   ├── tests/                pytest (≈131 tests)
│   └── experiments/          calibration/benchmark scripts + results/
├── backend/                  FastAPI app (app/main.py, app/codec.py, app/samples.py) + tests (≈19)
├── frontend/                 React 19 + Vite 7 + Tailwind 4 (src/App.jsx, src/components/*, src/lib/*)
├── render.yaml               Render deploy config (backend)
└── README.md
```

### 0.2 Set-up and baseline

1. **Use Python 3.11** so local runs match Render (`PYTHON_VERSION=3.11.9` in `render.yaml`).
   The current `.venv` may be Python 3.14. If so, recreate it. If 3.11 is not installed, tell the
   user how to install it (python.org or `winget install Python.Python.3.11`) and stop there.
   ```
   # Windows
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   # macOS / Linux
   python3.11 -m venv .venv && source .venv/bin/activate

   pip install -e ./pipeline -r backend/requirements-dev.txt scipy pillow
   ```
   The editable install (`-e`) is intentional: `uvicorn --reload` then picks up pipeline edits.
2. `cd frontend && npm install`
3. Run the baseline: `cd pipeline && python -m pytest -q` and `cd backend && python -m pytest -q`.
   Record the results. **Known environment issue:** on this Windows PC, Windows Application Control
   (Smart App Control) has blocked a SciPy DLL that scikit-image loads. That stops
   `pipeline/tests/test_core_vs_opencv.py` from importing, and it will also affect the new
   compression tests, which use SciPy. This is not a code problem. If it happens:
   - continue with the other tests;
   - try running the full suite under **WSL** (`wsl` → create a 3.11 venv inside the Linux side → run pytest);
   - report clearly in `CHANGES.md` which test files could not run on Windows and whether they passed
     under WSL.

   Any **other** baseline failure: report it and stop.
4. Read these files before editing: `registry.py`, `pipeline.py`, `guided.py`, `modules/*.py`,
   `backend/app/main.py`, `backend/app/codec.py`, `frontend/src/lib/labels.js`,
   `frontend/src/components/ToolsTab.jsx`, `CompareTab.jsx`, `HealthCard.jsx`, `Histogram.jsx`.

### 0.3 Conventions you must keep

- **The public API speaks RGB uint8** (H×W×3) or grayscale (H×W). All arithmetic runs in float32/float64,
  then `clip` + `rint` back to uint8. Never let uint8 wrap around.
- **Each course algorithm has two engines:** `engine="numpy"` (from scratch) and `engine="opencv"`
  (library). Tests assert that both engines agree.
- **Module parameters live in `registry.py` schemas.** The frontend generates its "Advanced settings"
  controls from `GET /api/modules`, so new params must be declared there (type, range, default,
  description, `when`).
- **Errors are `PipelineError(code, message)`.** The backend maps them to HTTP 422
  `{"error": {"code", "message"}}`.
- **The backend returns images as data URLs.** PNG is used for anything that must be exact; JPEG for
  photographic previews inside multi-image responses.
- Keep code readable for students: docstrings that state the formula, no clever one-liners.
- Don't add heavy frontend dependencies. Draw charts with a small hand-written SVG component.

---

## P. Phase P — Fix existing issues (do this before Phase A)

A code review of the current repo confirmed the issues below. Each item says what is wrong,
why it matters, what to do, and which test proves it. Write the test **first** where possible,
see it fail, then fix.

### P.1 What to do with the earlier session's changes (`test-1_session_changes.md`)

| Session change | Decision |
|---|---|
| 1. Layout `w-full` | **Keep** |
| 2a. Browser no longer shrinks uploads (sends full resolution) | **Replace** with P.2 (shrink to ≤ 2048 px *without smoothing*) |
| 2b/2c. Impulse density measured before `prepare()`, **in Auto Fix only** | **Replace** with P.3 (one impulse-aware `prepare()` for every entry point). Remove `pre_scale_impulse_density` from `auto.py` and `__init__.py` |
| 3. Light Gaussian (σ 0.6) for moderate noise in Auto Fix | **Keep** (the author's decision). Make sure its step reason is shown in the UI |
| 4. Equalize noise gate no longer requires a dark image | **Keep**, but fix the leftover reason text (P.5) |
| 5. Warning when A = 0 gives a grayscale edge image | **Keep** |
| 6. Warning when the edge response is all zero | **Keep** |
| 7. `fill_holes` uses the user's connectivity | **Replace**: the rule is backwards (P.4) |
| 8. Adaptive region growing marks rejected neighbours visited | **Revert** (P.6) |
| 9. 16-bit images converted with `>> 8` | **Keep** |
| 10. Editable install | **Keep** |

### P.2 Upload size: shrink in the browser to ≤ 2048 px **without smoothing**, plus a server pixel cap

**Why (the Render memory problem):** Render's free instance has **512 MB RAM**, and the process is
killed when it goes over. The app uses ~150–250 MB at rest (Python + NumPy + OpenCV + scikit-image +
FastAPI). Since session change 2a, the server receives **full-resolution** photos.

A decoded 12 MP photo is ~36 MB per RGB copy, and the current code makes several copies before
shrinking: a double decode in `decode_upload`, RGB conversion, a gray copy, a median copy and int16
working arrays. That peaks at ~220 MB, so 24–48 MP phone photos (or two users at once) crash the
service. There is also **no pixel-count limit**: a ~1 MB, highly compressible PNG can decode to
20000×20000 px (1.2 GB), an instant out-of-memory.

The original reason for change 2a was valid. Canvas downscaling uses bilinear smoothing, which blends
salt-and-pepper specks into their neighbours. **Nearest-neighbour** shrinking avoids that: every output
pixel is a copy of one input pixel, so specks stay specks and their density is preserved.

**Frontend (`frontend/src/lib/image.js`, `prepareImage`):**
- `MAX_UPLOAD_SIDE = 2048`.
- If the long side is > 2048: draw onto a canvas at the scaled size with
  `ctx.imageSmoothingEnabled = false` (nearest-neighbour), encode as **PNG** (lossless), and upload that.
- If the long side is ≤ 2048 and the type is PNG/JPEG/WebP/BMP: upload the original file unchanged.
  Other types: re-encode to PNG at full size.
- Return `{ file, url, width, height, name, resized, originalSize }`, where `width/height` are the
  **uploaded** dimensions.
- `ImagePanel.jsx`: the label must describe what was actually uploaded, e.g. "2048 × 1536 px (shrunk
  for upload from 4032 × 3024)". Remove any wording that implies only the preview was shrunk.

**Backend (`backend/app/codec.py`, `decode_upload`):**
- Add `pillow` to `backend/requirements.txt`.
- **Before decoding**, read the header with Pillow (`Image.open(BytesIO(data))`, use `.size` and
  `.mode`; this is lazy and does not decode the pixels). If `width × height > MAX_PIXELS` (env var,
  default **25_000_000**), raise `PipelineError("TOO_LARGE", "Image is W × H (N MP). The limit is 25 MP. Please resize it and try again.")`.
  If Pillow cannot read the header, fall through to the OpenCV decode (which may still fail with BAD_IMAGE).
- **Decode once, not twice.** Use the header `mode` to pick the flag:
  - modes with alpha (`RGBA`, `LA`, `PA`, …) → `IMREAD_UNCHANGED` (then drop alpha)
  - `L`/`I;16`/`I` → `IMREAD_UNCHANGED` (gray, keep the 16-bit `>> 8` rule)
  - everything else → `IMREAD_COLOR`, which applies EXIF orientation
- Keep the "store gray-looking RGB as gray" optimisation.

**Server-side `prepare()` stays the safety net** for clients that bypass the browser (direct API
calls): it still handles anything up to the pixel cap (see P.3).

**Tests:**
- Backend: a PNG whose header says 6000×5000 (30 MP; create it with a flat colour so the file is small)
  → 422 `TOO_LARGE`, returned **without** decoding (assert quickly, or mock `cv2.imdecode` to fail if
  called).
- A 24 MP-header image just under the cap is accepted.
- RGBA PNG and 16-bit gray PNG still decode correctly.
- A JPEG with EXIF orientation 6 is returned rotated (portrait).

### P.3 One impulse-aware `prepare()` for every entry point (fixes bug: specks missed on large images)

**Bug (confirmed):** for images larger than 1024 px, `prepare()` shrinks with `cv2.INTER_AREA`, which
averages salt-and-pepper specks away. Session fix 2b patched only `auto_enhance`, so:
- the Image health card does **not** report salt & pepper on a 2048 px image with 5 % specks (only "grainy noise");
- **Compare → Segment applies a Gaussian blur instead of a median filter.** This is the project's main claim;
- only Auto Fix gets it right.

**Fix (in `pipeline/ivp_pipeline/utils.py`, `prepare`):**
1. If `max(H, W) <= MAX_SIDE` (1024): return the image unchanged (scale 1.0).
2. Otherwise, estimate impulse density cheaply **before** resizing:
   - Make a probe copy of the gray image shrunk to at most 2048 px with `INTER_NEAREST`
     (nearest-neighbour keeps speck density), or use the image itself if it is already ≤ 2048.
   - Run `metrics.impulse_density(probe)`.
3. Resize to `MAX_SIDE`:
   - `INTER_NEAREST` if `impulse_density > config.IMPULSE_DENSITY` (keep the specks so every later
     stage sees the true defect);
   - otherwise `INTER_AREA` (smooth, no aliasing).
4. Return `(image, scale)` as before, and also expose the decision. Either return a third value or set a
   small dataclass/dict that the public API puts into `meta` as `resize_interp` ("none" | "area" |
   "nearest") and `impulse_density_full_res`.
5. Use this `prepare()` everywhere (`diagnose`, `run_module`, `run_guided`, `auto_enhance`, and the new
   `compare_compression`/`degradation_sweep`). **Remove** the Auto-Fix-only `pre_scale_impulse_density`
   code path.

**Reduce memory in `metrics.impulse_density`:** compute on uint8 with `cv2.medianBlur` and
`cv2.absdiff(gray, med) > IMPULSE_MEDIAN_DIFF`, combined with the extreme-value mask. Avoid int16/float
copies. The result must be unchanged (add a test comparing old and new formulas on a few images).

**Tests (`pipeline/tests/test_prepare.py`):**
- A 2048×2048 version of a synthetic scene with 5 % salt & pepper:
  - `diagnose` reports `IMPULSE_NOISE`;
  - `run_guided(…, "segment")` Branch B's **first stage is `denoise:median`** (not Gaussian);
  - `auto_enhance` applies a median first;
  - `meta.resize_interp == "nearest"`.
- A clean 2048 px photo (e.g. `cv2.resize(data.astronaut(), (2048, 2048))`) → `resize_interp == "area"`,
  and no `IMPULSE_NOISE` finding.
- Images ≤ 1024 px → `resize_interp == "none"`, scale 1.0.

### P.4 `fill_holes` connectivity rule (confirmed bug from session change 7)

**Rule (digital topology):** the background flood fill must use the **complementary** connectivity
of the foreground. If the object is treated as 8-connected, the background must be flooded with
4-connectivity, and vice versa. Otherwise, background can "leak" through diagonal gaps in an
8-connected ring and the hole is never filled.

**Fix:** in `segment.py::_fill_holes` and `core/morphology.py::fill_holes`, flood the background with
`8 if connectivity == 4 else 4`. The default call (no connectivity given) treats the object as
8-connected, so the background fill uses 4 (the original behaviour). Apply it to both the OpenCV
(`cv2.floodFill` flags) and NumPy (`bfs_region`) engines.

**Test:** a 1-px-thick ring that touches itself only diagonally (e.g. a diamond drawn with diagonal
steps) with an empty interior. With `connectivity=8` the interior **is** filled (both engines). Also
keep the existing 4-connected test.

### P.5 Wrong reason text in guided equalize (confirmed)

`guided.py` (around line 85) still appends "in a dark image" after session change 4 removed the
dark-image condition. Only add that phrase if `d["mean"] < C.DARK_MEAN`; otherwise just
"noise σ ≈ x > limit". Test: a bright noisy image → the reason does not contain "dark".

### P.6 Revert the adaptive region-growing change (session change 8)

With session change 8, a pixel rejected early is **never re-checked**, even after the running mean
moves. That changes the algorithm, not just its speed. The textbook adaptive predicate compares
against the *current* region mean, so a previously rejected neighbour may qualify later.

Restore the original logic in `core/region.py`: mark `visited` only when a pixel is **accepted**.
Region growing is leaving the Tools tab (Phase D), so performance there no longer affects users.

Add a short docstring note explaining why rejected pixels are re-tested, and a test where a pixel is
rejected early and accepted later (construct a small gradient image where this happens).

### P.7 CORS headers on 500 errors (confirmed)

The catch-all `@app.exception_handler(Exception)` runs in Starlette's `ServerErrorMiddleware`, which
sits **outside** `CORSMiddleware`. A crash therefore reaches the deployed site without CORS headers,
and the browser shows a CORS error instead of the friendly message.

**Fix:** add an HTTP middleware that wraps `call_next` in `try/except Exception` and returns the same
`{"error": {"code": "INTERNAL", ...}}` JSON (and logs the traceback). Register it so it sits
**inside** CORS: in Starlette the **last added middleware is outermost**, so register the error
middleware **before** `app.add_middleware(CORSMiddleware, …)`.

**Test:** a test-only route (or monkeypatch an endpoint) that raises `RuntimeError`, requested with an
`Origin` header, gets **500 with `access-control-allow-origin`** set. Use
`TestClient(app, raise_server_exceptions=False)`.

### P.8 Repo housekeeping

- Commit the deletions of `Media.jpg` and `test_salt_pepper.jpg` if they are intentionally gone
  (ask the user if unsure).
- Move `generate_synthetic_dataset.py` and `run_e2e_tests.py` into `pipeline/experiments/`. Replace the
  hard-coded `F:\ivp-studio\...` paths with paths relative to the script
  (`Path(__file__).resolve().parent`) or command-line arguments. Make sure both still run.
- Add `dataset/` (and any generated output folders) to `.gitignore`, unless the folder is small (< 5 MB)
  and needed for tests. Report its size.
- Don't commit or push anything yourself unless the user asks. Leave a list of suggested commits in
  `CHANGES.md`.

### P.9 Note on bug: "Click to select" picks the wrong spot on large images

Confirmed by the review, but it lives entirely in the Tools-tab "Cut out object" feature, which Phase D
removes. **Don't fix it separately; just make sure the removal is complete.** Nothing in Tools may use
seed picking afterwards. The backend can keep scaling seeds in `run_module`, since the API still
accepts them.

**End of Phase P:** all tests green (except the known Windows SciPy limitation, documented),
`npm run build` succeeds.

---

## 1. Summary of the change

**Tools tab**
1. **Add** an **Image Compression** tool (DCT and Walsh–Hadamard, RGB in → RGB out with correct colours),
   in the slot where "Cut out object" was.
2. **Remove** the "Cut out object" tool (segmentation) from the Tools tab, including click-to-select,
   the overlay/mask/cut-out views and the cut-out download.
   **Segmentation stays in the pipeline and in Compare.**
3. **Remove** the "Fix uneven lighting" preset. Keep `contrast` mode `flatten` in the pipeline,
   because guided segmentation uses it.
4. **Remove** the "Brighten shadows" (γ 0.6) and "Tone down highlights" (γ 1.6) presets.
   Keep the manual gamma slider in Advanced.

**Image health card**

5. Hide the segmentation-only findings `UNEVEN_ILLUMINATION` and `LOW_SEPARABILITY` from the health
   card, and remove the "Flatten lighting" fix button. They still appear in Compare → Segment
   ("Why?" section).

**Compare tab**

6. Offer exactly three comparisons:
   - **Segment (Otsu thresholding):** direct vs fix-first (existing)
   - **Find edges:** direct vs fix-first (existing)
   - **Compression (new):** DCT vs Hadamard at the **same % of coefficients kept**

   Remove Sharpen and Equalize from the Compare UI. The backend intents can stay.
7. Segment and Find edges get:
   - a **noise/degradation sweep chart** (accuracy vs degradation, Direct vs Guided, marker on the user's image)
   - a **colour-coded difference map**
   - **paired metric bars**
8. Compression gets:
   - a **quality-vs-compression (rate–distortion) curve**
   - an **energy compaction curve**
   - **error heat maps with a zoom lens**
   - **coefficient maps** (before/after)
   - **paired metric bars**

**Unchanged:** Auto Fix tab, Remove noise tool, Sharpen & edges tool, upload/samples/undo/download,
server wake-up handling, deployment configs.

**Explicitly NOT included:** a "show the colour bug" toggle, and a gradient-profile view.

**Also in scope:**
- Phase P (fix the issues found by the code review). Do it first.
- Phase F (close out the remaining items of the earlier bug report).

---

## 2. Phase A — Pipeline: compression transforms (from scratch)

### 2.1 Background: the lab code (MATLAB) this must reproduce

The student's lab scripts are reproduced in **Appendix A**. Key facts:

- `DCT 1.m` / `single_channel_dct.m`: `dct2` on each channel of a uint8 image (values 0–255),
  zero coefficients with `abs(c) < 500`, then `idct2`.
- **The colour bug:** `single_channel_dct.m` then calls `rescale()` **on each channel separately**.
  Each channel is stretched min→max independently, so the colour balance shifts (the "hue-y" image).
  **The correct reconstruction is**: inverse transform → `clip(0, 255)` → round → uint8, with no rescaling.
- `hadamard_transform.m`: converts to gray, `imresize` to 2048×2048 (distorts the aspect ratio),
  `im2double`, `fwht` on columns then on rows, then `Y(256:r,256:c)=0`, then `ifwht` twice.
  MATLAB's `fwht` defaults to **sequency ordering** and scales the forward transform by **1/N**.
  Note: `Y(256:r,256:c)=0` zeros only the block where *both* row ≥ 256 *and* col ≥ 256, so the kept
  region is L-shaped, not a square.

### 2.2 New file `pipeline/ivp_pipeline/core/transforms.py`

Implement and document, all in NumPy:

1. **`dct_matrix(n) -> (n, n) float64`**: orthonormal DCT-II basis
   `C[k, x] = α(k) · cos(π (2x + 1) k / (2n))`, `α(0) = √(1/n)`, `α(k) = √(2/n)` for k ≥ 1.
   Cache matrices with `functools.lru_cache`.
2. **`dct2(f) / idct2(F)`** on a 2-D float array: `F = C_M · f · C_Nᵀ`, `f = C_Mᵀ · F · C_N`.
   This equals MATLAB `dct2/idct2`, `scipy.fft.dctn(..., norm="ortho")`, and `cv2.dct` (even sizes only).
3. **`walsh_matrix(n)`**: n must be a power of 2. Build the Sylvester–Hadamard matrix (natural order)
   recursively, **reorder rows into sequency order**, and scale by `1/√n` (orthonormal).
   Sequency order: row `k` of the Walsh matrix is natural-order row `bitreverse(gray(k))`, where
   `gray(k) = k ^ (k >> 1)` and bitreverse uses `log2(n)` bits. **Test:** row `k` has exactly `k` sign changes.
4. **`fwht2(f) / ifwht2(F)`**: orthonormal 2-D Walsh–Hadamard in sequency order (`W_M · f · W_Nᵀ` and
   its transpose; the matrices are orthogonal). For speed, also implement an iterative fast butterfly
   `O(N log N)` along an axis and use it when `n ≥ 64`. The test checks that the matrix version and
   the butterfly agree.
   Docstring: MATLAB's `fwht` equals this orthonormal transform divided by `√N` per dimension
   (same ordering). We use the orthonormal scale so that thresholds and energy are comparable
   with the DCT (Parseval holds for both).
5. **`next_pow2(n)`** and **`pad_to_pow2(img)`** for Hadamard: pad each dimension *separately* to its
   next power of 2 using `np.pad(mode="reflect")` (or `edge` when a dimension is too small to
   reflect), and return the pad amounts so the result can be cropped back. **Do not resize.**
   Rectangular 2ᵃ × 2ᵇ is fine.
6. **Coefficient selection helpers** (operate on one 2-D coefficient array; return the masked
   array and the kept-mask):
   - `keep_percent(F, p)`: keep the `ceil(p/100 · size)` largest-magnitude coefficients.
     Always keep the DC term `F[0, 0]`. Break ties deterministically.
   - `threshold(F, t)`: zero `|F| < t`, but always keep DC.
   - `zonal(F, k_fraction, shape)`: `k = round(k_fraction · min(M, N))`. With `shape="square"`, keep
     rows < k **and** cols < k (classic zonal coding). With `shape="lab"`, reproduce the lab script
     exactly: zero rows ≥ k **and** cols ≥ k (keeps an L-shaped region).
7. **`energy_compaction(F, fractions)`**: for orthonormal coefficients, sort `F²` descending and return
   the cumulative energy share captured at each fraction of coefficients kept. Sum over channels
   for RGB.

**OpenCV engine:** `cv2.dct`/`cv2.idct` on float32 for the DCT. OpenCV requires even sizes, so pad
odd dimensions by 1 (reflect) and crop back. OpenCV has no Walsh–Hadamard; the Hadamard path uses the
NumPy fast butterfly in both engines, and the docs must say so.

### 2.3 New module `pipeline/ivp_pipeline/modules/compress.py`

`run(img, params, engine) -> (out, extras, metrics, warnings)`, same signature as the other modules.

- **RGB input:** process R, G and B independently with the same settings. Grayscale input: one channel.
- **Pipeline per channel:** to float64 (0–255 scale, **not** 0–1, so the lab threshold 500 means the
  same thing as in MATLAB) → forward transform (Hadamard: pad first) → coefficient selection →
  inverse → crop (Hadamard) → `clip(0, 255)` → `rint` → uint8. **No per-channel rescaling.**
- **Params**, to add in `registry.py` under a new schema `"compress"` (title "Image compression",
  syllabus "Transforms: DCT & Walsh–Hadamard (lab)"):

  | name | type | options / range | default | when |
  |---|---|---|---|---|
  | `method` | enum | `dct`, `hadamard` | `dct` | — |
  | `mode` | enum | `keep_percent`, `threshold`, `zonal` | `keep_percent` | — |
  | `keep_percent` | float | 0.1 – 100 (step 0.1) | 10 | mode = keep_percent |
  | `threshold` | float | 0 – 5000 (step 10) | 500 | mode = threshold |
  | `zonal_fraction` | float | 0.01 – 1.0 (step 0.01) | 0.125 | mode = zonal |
  | `zonal_shape` | enum | `square`, `lab` | `square` | mode = zonal |

  (0.125 = 256/2048, the lab setting.)
- **`extras` (image arrays, encoded by the backend as PNG):**
  - `coef_before`, `coef_after`: log-magnitude maps `log(1 + |F|)` of the **luminance** coefficients
    (compute Y from the RGB coefficients' channel average, or transform Y once for the map), normalised
    to 0–255 uint8. For Hadamard, show the padded 2ᵃ × 2ᵇ map.
  - `kept_mask`: uint8 0/255 map of kept coefficients (luminance or channel union), same size as the
    coefficient map.
  - `error_map`: `|original − compressed|` averaged over channels, multiplied by an amplification
    factor (report it in metrics), clipped, then colour-mapped with `cv2.applyColorMap(..., COLORMAP_INFERNO)`
    and converted BGR→RGB.
- **`metrics`:**
  - `method`
  - `coefficients_total`, `coefficients_kept`, `kept_percent`
  - `compression_ratio` = total / kept (state in the UI that this is a coefficient-count ratio,
    not a file-size ratio)
  - `psnr_db`, `ssim` (use the existing `metrics.psnr`/`metrics.ssim`; SSIM with `channel_axis` for RGB)
  - `energy_kept_percent`
  - `error_amplification`
  - `size_original_png_bytes`, `size_compressed_png_bytes`, `size_compressed_jpeg_q90_bytes`, using
    `cv2.imencode` on the original and the reconstruction. The UI labels these honestly: JPEG adds its
    own compression; PNG sizes show the effect of fewer coefficients only indirectly.
- **`warnings`:** e.g. "Hadamard needs power-of-two sizes; the image was padded from 683×1024 to
  1024×1024 and cropped back."
- Register the module in `pipeline.py` (`RUNNERS["compress"]`) and in `registry.py`. `list_modules()`
  must include it.

### 2.4 Comparison helper `pipeline/ivp_pipeline/compare_compression.py`

`compare_compression(img, keep_percent=10.0, engine="opencv") -> dict`

- Runs **DCT** and **Hadamard** with `mode="keep_percent"` and the **same** `keep_percent`
  (fair: equal coefficient budget).
- Returns, for each method: `image`, `extras` (coef maps, kept mask, error map) and `metrics`.
- **`rd_curve`:** for `keep_percent ∈ [0.5, 1, 2, 5, 10, 25, 50]`, PSNR and SSIM for both methods.
  For speed, compute the curve on a copy downscaled so the max side ≤ 512 (`INTER_AREA`); report
  `curve_scale` so the UI can say so. Do each forward transform **once** per channel and reuse it
  for every level.
- **`energy_curve`:** fractions from 0.1% to 100% (≈40 log-spaced points), energy captured for each
  method (orthonormal coefficients, summed over channels), on the same downscaled copy.
- **`summary`:** which method wins on PSNR at the user's setting, and by how many dB.
- Expose it in the public API (`ivp_pipeline/__init__.py`) as `compare_compression`, using
  `prepare()` like the other entry points.

### 2.5 Pipeline tests (new file `pipeline/tests/test_compression.py`)

Include at least:
- `dct2`/`idct2` vs `scipy.fft.dctn/idctn(norm="ortho")`: max abs diff < 1e-8 on random and real
  images, odd and even sizes.
- `cv2.dct` path agrees with the NumPy path (even and odd sizes).
- `walsh_matrix(n) @ walsh_matrix(n).T ≈ I`; row k has k sign changes; the fast butterfly equals the
  matrix product.
- Perfect reconstruction: `keep_percent=100` returns the input (max diff ≤ 1).
- **Colour correctness (the lab bug must not exist):** on `skimage.data.astronaut()` at keep 10%, the
  mean of each channel stays within 2 levels of the original, and the mean hue (HSV) difference is
  small. Also include a test showing that per-channel min-max rescaling (the lab bug) **does** shift
  channel means, to document why it was removed.
- Gray input gives gray output; RGB gives RGB; output shape equals input shape (including after
  Hadamard padding/cropping on odd sizes such as 301×457).
- `keep_percent` keeps exactly the requested count (± DC tie rule); `threshold` reproduces the lab
  semantics; `zonal` "square" vs "lab" masks have the expected shapes.
- DCT beats or equals Hadamard in PSNR at 5% and 10% on natural images (camera, astronaut, coffee).
  It's a well-known property, but assert it with a small tolerance so a single odd image can't flake.
- Energy compaction: curves are monotonic, end at 100%, and DCT ≥ Hadamard at 5%.
- Both engines agree for `compress`.
- Registry: bad params raise `PipelineError("BAD_PARAM")`.

---

## 3. Phase B — Pipeline: sweeps and difference maps for Compare

### 3.1 New file `pipeline/ivp_pipeline/sweep.py`

`degradation_sweep(img, intent, params=None, engine="opencv") -> dict` for `intent ∈ {"segment", "edges"}`.

- **Pick the degradation axis from the diagnosis of the user's image:**
  - `segment`: if `UNEVEN_ILLUMINATION` fires, sweep `synthetic.illumination_ramp` strength
    [0, 20, 40, 70]; x-value = measured `illumination_u` of each generated image.
    Else if `IMPULSE_NOISE` fires, sweep `salt_pepper` density [0, 1, 5, 10, 20]%; x = measured
    `impulse_density`.
    Otherwise, sweep Gaussian σ [0, 5, 10, 20, 30]; x = measured `noise_sigma`.
  - `edges`: Gaussian σ [0, 5, 10, 20, 30] (x = measured `noise_sigma`). If `IMPULSE_NOISE` fires,
    sweep salt & pepper instead.
  - Using the **measured** value as x means the user's own image (level 0) sits at its true position.
    Return `user_point_index = 0`.
- **Reference ("ground truth") for scoring:** the best available result on the **unmodified** input:
  Branch B of `run_guided(img, intent, params)` if the gate fired, else Branch A.
  Return `reference_note` explaining this.
- At each level: degrade with a fixed seed → `run_guided(degraded, intent, params)`.
  - Score A and B (if B is None, B = A, so the lines overlap; that's the intended story).
  - `segment`: IoU of the masks vs the reference mask.
  - `edges`: `metrics.edge_f1(..., tolerance=1)` of the thresholded edge maps (use the target
    `edge_threshold`, or `config.DEFAULT_EDGE_T` if it is 0) vs the reference edge map.
- Return `{axis: {key, label, unit}, points: [{level, x, direct, guided, gate_fired}], metric: "IoU"|"Edge F1",
  user_point_index, reference_note}`.
- Performance target: ≤ ~2 s on a 1024×1024 image with the opencv engine. Downscale to max side 512 for
  the sweep if needed, and report `sweep_scale`.

### 3.2 Difference map (add to the guided result)

In `guided.py` (or a helper used by it), when Branch B exists, add `difference_map` (RGB uint8) and
`difference_legend` to the result:
- **edges:** binary edge maps A and B (same threshold rule as above). Black = neither, **red** (235,104,52)
  = only in A (false edges from noise), **green** (27,175,122) = only in B, **white** = both.
- **segment:** masks A and B. Show the input image dimmed to 40% as background; **red** = foreground in A
  only, **green** = foreground in B only; agreement stays dimmed. Legend: "B is used as the reference;
  red = what the direct route got wrong."
- Also add counts to `metrics`/`checks` for the bars: edges → `edge_pixels_a/b`,
  `false_edge_density_a/b` (flat regions), `only_a_pixels`, `only_b_pixels`; segment →
  `fragmentation_a/b`, `eta_a/b`, `n_components_a/b`.

### 3.3 Tests (new `pipeline/tests/test_sweep.py`, extend `test_guided.py`)

- On `synthetic.scene` + ramp 40: `degradation_sweep(..., "segment")` chooses the illumination axis,
  `guided ≥ direct` at every level, and guided clearly wins at the strongest level.
- On `camera` + Gaussian 25 with Sobel: edges sweep shows `guided > direct` at σ ≥ 20 and the lines
  nearly coincide at σ = 0.
- The x of the first point equals the diagnosis value of the input (within tolerance).
- `difference_map` exists only when Branch B exists, has the image's H×W, and its colour counts match
  `only_a_pixels`/`only_b_pixels`.
- Runtime guard: a 512×512 sweep finishes in < 5 s (mark `@pytest.mark.slow` if needed).

---

## 4. Phase C — Backend (FastAPI)

In `backend/app/main.py` (keep the existing style, error handling and sync `def` endpoints):

1. **`/api/process` already serves `module="compress"`.** Make sure `codec.module_result` encodes the new
   extras: add `coef_before`, `coef_after`, `kept_mask`, `error_map` to `IMAGE_EXTRAS` (PNG).
   Main image as PNG.
2. **`POST /api/sweep`**: form fields `file`, `intent` (`segment`|`edges`), `params` (JSON),
   `engine` → `jsonable(degradation_sweep(...))`. No images in the response, only numbers.
3. **`POST /api/compare-compression`**: form fields `file`, `keep_percent` (float), `engine` →
   `{dct: {...}, hadamard: {...}, rd_curve, energy_curve, summary, meta}`. Encode the method images as
   PNG (exactness matters for judging artefacts) and the maps as PNG.
4. **`/api/guided`**: include `difference_map` (PNG data URL) and `difference_legend` when present.
5. **Samples (`app/samples.py`):** add a colourful sample for compression, e.g.
   `("coffee", "Coffee", "Colourful. Try Compression", lambda: data.coffee())`. Also consider
   `data.chelsea()`. Keep the existing samples.
6. **Tests (`backend/tests/test_api.py`):**
   - `process` with `compress` (DCT and Hadamard, RGB)
   - `/api/sweep` for both intents
   - `/api/compare-compression` (shapes and keys, `rd_curve` length 7)
   - `difference_map` present in a guided response
   - bad `intent`/`keep_percent` → 422

---

## 5. Phase D — Frontend

### 5.1 Tools tab (`src/lib/labels.js`, `src/components/ToolsTab.jsx`)

- `TOOLS`: remove the `segment` tool entirely. Add a **compression tool in the same (4th) slot**:
  ```js
  { id: "compress", name: "Compression", icon: <a lucide icon such as FileArchive or Minimize2 — verify it exists in lucide-react 0.460>,
    blurb: "DCT and Walsh–Hadamard transform coding (lab experiments).",
    presets: [
      { id: "light",  label: "Light compression",  hint: "DCT, keep 25 % of coefficients", params: { method: "dct", mode: "keep_percent", keep_percent: 25 } },
      { id: "medium", label: "Medium compression", hint: "DCT, keep 10 %",                 params: { method: "dct", mode: "keep_percent", keep_percent: 10 } },
      { id: "strong", label: "Strong compression", hint: "DCT, keep 2 %",                  params: { method: "dct", mode: "keep_percent", keep_percent: 2 } },
      { id: "lab_dct", label: "Lab: DCT threshold 500", hint: "Exactly like DCT 1.m, in colour", params: { method: "dct", mode: "threshold", threshold: 500 } },
      { id: "lab_wht", label: "Lab: Hadamard zonal",    hint: "Like hadamard_transform.m (256/2048)", params: { method: "hadamard", mode: "zonal", zonal_fraction: 0.125, zonal_shape: "lab" } },
    ] }
  ```
- `contrast` presets: **remove** `flatten`, `brighten`, `darken`. Keep `auto_gamma`, `stretch`,
  `hist_eq`. Keep the Advanced controls. `mode` may still list `flatten` because it comes from the
  schema; hide that option in the UI (e.g. filter enum options via a small `HIDDEN_OPTIONS` map in
  `labels.js`), so users can't pick it from Tools.
- `ToolsTab.jsx`: remove all segmentation-specific UI (pick mode, seed marker, Overlay/Mask/Cut-out
  view switch, "Pick another point", cut-out download naming). `CompareSlider` may keep its generic
  pick support, but nothing in Tools should use it.
- **Compression result UI** (only when `result.tool === "compress"`), below the slider:
  - a card "Coefficients" with `coef_before` and `coef_after` side by side (labels "All coefficients",
    "Kept"), plus the kept-mask thumbnail
  - a card "Error map" with `error_map` and its caption ("brighter = bigger error, ×N amplified")
  - "Result details" shows: PSNR, SSIM, kept %, compression ratio (labelled "coefficient ratio"),
    energy kept, and file sizes with a one-line honest note
- Add metric labels/formatting for the new keys in `METRIC_LABELS`.
- Health-card "fix" mappings: remove the `UNEVEN_ILLUMINATION` → flatten fix.

### 5.2 Image health card (`HealthCard.jsx`)

- Filter out `UNEVEN_ILLUMINATION` and `LOW_SEPARABILITY` before rendering (constant
  `SEGMENTATION_ONLY_FINDINGS` in `labels.js`). Everything else stays.

### 5.3 Compare tab (`CompareTab.jsx` + new components)

- `INTENTS` = **Segment (Otsu thresholding)**, **Find edges** (operator selector stays), **Compression**
  (new; control: a "Keep % of coefficients" slider, default 10, with quick chips 2 / 5 / 10 / 25).
- **Segment / Find edges flow:** keep the current flow (verdict banner, guided-route strip, slider /
  side-by-side, "Why?" with relevant findings first). After the guided result arrives, call
  `/api/sweep` in the background with its own loading state. Show **in this order**:
  1. **Verdict banner** (existing)
  2. **Sweep chart** (new `LineChart` component, see 5.4)
     - Title example: "Edge accuracy vs noise" / "Segmentation accuracy vs uneven lighting".
     - Two series: **Direct** = orange `#eb6834`, **Guided** = blue `#2a78d6`.
     - Direct labels at line ends, plus a legend.
     - The user's image point marked with a ring and a "your image" label.
     - Caption: the `reference_note`, plus "noise added by us at fixed levels".
  3. **Difference map** (new card) with its legend chips. Clicking opens it larger. A simple zoom
     (CSS `transform: scale`) is enough.
  4. **Metric bars** (new `PairedBars` component): 2–4 rows such as "False edge pixels in flat areas",
     "Edge pixels only in Direct", "Pieces (fragmentation)", "Separability η". Each row has an A bar and
     a B bar on a shared scale, value labels in text colour (not series colour), and "lower is better" /
     "higher is better" hints.
  5. Images (slider / side by side), the guided-route strip, then "Why?".
- **Compression flow:** call `/api/compare-compression` with `keep_percent`. Show:
  1. A summary banner, e.g. "At 10 % of coefficients, DCT is 3.4 dB better than Hadamard", plus a
     one-line reason: "DCT packs more of the image's energy into fewer coefficients."
  2. **Rate–distortion chart:** PSNR (dB) vs % coefficients kept, log-scale x, DCT = blue `#2a78d6`,
     Hadamard = orange `#eb6834`, marker at the user's setting. Tabs or toggle: PSNR | SSIM.
  3. **Energy compaction chart:** % energy captured vs % coefficients kept (log x), same colours.
  4. Images: **DCT vs Hadamard** in the existing `CompareSlider` (left = Hadamard, right = DCT, labelled),
     plus a side-by-side option.
  5. **Error heat maps** for both methods side by side, with a **zoom lens**: hovering one map shows the
     same magnified region (e.g. 3×) on both maps **and** on both reconstructions, so ringing (DCT) vs
     blockiness (Hadamard) is visible. Write it as a small reusable `ZoomLens` component; no library.
  6. **Coefficient maps** before/after for both methods.
  7. **Metric bars:** PSNR, SSIM, energy kept, PNG size, JPEG size (DCT vs Hadamard).
  8. Caption noting that the curves are computed on a ≤512 px copy, if `curve_scale < 1`.
- Remove Sharpen and Equalize from `INTENTS`.

### 5.4 Chart components (new `src/components/charts/`)

- **`LineChart.jsx`:** responsive SVG (`viewBox` + width 100%).
  - Props: `series=[{id, label, color, points:[{x,y}]}]`, `xLabel`, `yLabel`, `xScale: "linear"|"log"`,
    `yDomain`, `marker={x, label}`, `formatX`, `formatY`.
  - 2 px lines, ≥ 8 px point markers with a 2 px white ring, recessive grid, one y-axis only.
  - Legend above the plot; direct labels at line ends.
  - **Hover crosshair + tooltip** listing every series' value at the nearest x.
  - A visually hidden `<table>` with the data for accessibility.
- **`PairedBars.jsx`:** horizontal bars, two per row (A/B colours as above), shared scale per row,
  4 px rounded ends, 2 px gap between bars, numeric labels in slate text colour.
- Colours come from CSS variables defined once (e.g. in `index.css`: `--series-direct`,
  `--series-guided`, `--series-dct`, `--series-hadamard`), not scattered hex values.

### 5.5 API client (`src/lib/api.js`)

Add `sweep(file, intent, params, engine, signal)` and
`compareCompression(file, keepPercent, engine, signal)`. Abort stale requests like the existing calls do.

---

## 6. Phase E — Docs

- `README.md` (root): update the feature table (Tools: Light & contrast, Remove noise, Sharpen & edges,
  **Compression**; Compare: Segment, Find edges, **Compression**), the demo steps (add "Coffee → Compare →
  Compression"), and the API table (`/api/sweep`, `/api/compare-compression`).
- `pipeline/README.md`: document the compression module, transforms, sequency ordering, the colour-bug
  explanation, and the sweep/difference-map design.
- If `ivp_pipeline_spec_v2.html` is in the repo, add a short "v3 changes" section; otherwise skip.

---

## 7. Phase F — Close out the earlier bug report

An earlier automated review listed the items below. Most are now settled by the session changes or
Phase P. **Only the items marked VERIFY need work:** write a small test that would fail if the bug
were real, run it, and fix only if it fails. Record the final status of **every** row in `CHANGES.md`:
CONFIRMED-FIXED, NOT-A-BUG (with the reason), KEPT (author's decision) or DEFERRED.

| # | Location | Claim | Status / action |
|---|---|---|---|
| 1 | `prepare()` / `auto.py` | Downscaling averages salt & pepper away on large images | **Handled in Phase P.2 + P.3** (browser nearest-neighbour shrink + impulse-aware `prepare()` everywhere) → CONFIRMED-FIXED |
| 2 | `auto.py` | σ in (5, 8] only skipped sharpening, no denoising | **KEPT**: the session added a light Gaussian (σ 0.6); the author chose to keep it. Ensure the step and its reason appear in the Auto Fix "What was done" list |
| 3 | `auto.py` | Exposure/contrast checks use pre-fix metrics | **VERIFY.** Probably false: `apply()` re-diagnoses and updates `d` via `nonlocal`. Add a test: dark + 5 % speck image → the gamma step's reason reports the post-median mean |
| 4 | `guided.py` | Equalize gate needed noisy **and** dark | **KEPT** (session removed the dark condition) + reason text fixed in **P.5**. Equalize also leaves the Compare UI |
| 5 | `guided.py` | Flatten radius capped at 75 for images > 750 px | **VERIFY:** run guided segmentation on a 1024 px synthetic scene with large objects (radius 40–60 px) and an illumination ramp. Fix only if IoU of Branch B < 0.95 (e.g. scale the cap with image size). |
| 8 | `denoise.py` | `kernel_size` ignored for Gaussian | NOT-A-BUG: size is derived from σ; the registry `when` hides the control |
| 9 | `sharpen_edges.py` | A = 0 on colour returns gray | **KEPT** (session added a warning). **VERIFY** only that "Keep editing this result" works on the gray PNG (pipeline accepts gray) |
| 10 | `sharpen_edges.py` | Constant image → black edges with no warning | CONFIRMED-FIXED (session change 6). Make sure a test covers it |
| 11 | `segment.py` | OpenCV region grow returns a 1-px mask if the seed is a spike | **VERIFY.** Probably false: the seed value is the 3×3 median and the seed is forced into the region. Add the test. Low priority, since region growing leaves Tools |
| 12 | `segment.py` | `fill_holes` connectivity | **Handled in P.4** (complementary-connectivity rule) → CONFIRMED-FIXED |
| 14 | `backend/app/codec.py` | 16-bit min-max normalisation | CONFIRMED-FIXED (session change 9, `>> 8`). Make sure a test covers it (P.2 tests) |
| 16 | `core/region.py` | Adaptive BFS re-tests rejected neighbours | **Reverted in P.6**: re-testing is the correct behaviour → NOT-A-BUG (documented) |

---

## 8. Phase G — Final verification (all must pass)

1. `cd pipeline && python -m pytest -q`: all green (old and new).
2. `cd backend && python -m pytest -q`: all green.
3. `cd frontend && npm run build`: no errors. Fix any warnings you introduced.
4. **End-to-end check in a real browser.** Use Playwright if available
   (`npx playwright install chromium` if needed), otherwise give the user a manual checklist.
   Start `uvicorn app.main:app --port 8000` (in `backend/`) and `npm run dev` (in `frontend/`), then verify:
   - Tools shows exactly four tools: Light & contrast, Remove noise, Sharpen & edges, **Compression**.
     No Cut out object, no Fix uneven lighting, no Brighten shadows / Tone down highlights.
   - The Coffee sample → Compression → each preset gives a correct-colour result, coefficient maps,
     error map and metrics. Then "Keep editing this result" and Undo work.
   - The health card never shows "Uneven lighting" or "Object/background not clearly separated".
   - Compare shows exactly three options.
     - **Segment** on "Shapes, uneven light": verdict, sweep chart (guided above direct), difference
       map, bars.
     - **Find edges** on "Cameraman, noisy": same elements; switching Sobel → Roberts changes the chart.
     - **Compression** on "Coffee": summary banner, RD chart, energy chart, DCT/Hadamard slider, error
       maps with a working zoom lens, coefficient maps, bars.
   - Auto Fix still works on "Astronaut, too dark".
   - At 390 px width (mobile) nothing overflows horizontally; charts stay readable.
   - No console errors (ignore a blocked Google Fonts request if offline).
   - **Phase P checks:**
     - Upload a ≥ 4000 px photo: the image card says it was shrunk for upload to ≤ 2048 px, and the
       request body is a PNG ≤ 2048 px.
     - Upload a 2048+ px image with 5 % salt & pepper (generate it with `synthetic.salt_pepper`):
       the health card shows **Salt & pepper noise**, and Compare → Segment's guided route starts with
       **Median filter**.
     - Upload a PNG over 25 MP: a friendly "too large" message appears, and the server stays up.
     - Force a server error (temporarily, or via the test route): the browser shows the friendly
       error, **not** a CORS error.
5. **Memory sanity check:** measure peak memory of one `/api/process` call on a 24 MP photo
   (e.g. `tracemalloc`, or `psutil` RSS before and after). Report it in `CHANGES.md`. Target: peak
   additional memory well under 250 MB.
6. Write `CHANGES.md` at the repo root summarising:
   - features added and removed;
   - Phase P fixes, and what happened to each session change (kept / replaced / reverted);
   - new endpoints;
   - new and updated tests (with counts), plus which tests could not run on Windows and their WSL result;
   - the Phase F table with final statuses;
   - the memory measurement;
   - anything deferred;
   - suggested commit messages.

**Done means:** all tests pass (Windows SciPy limitation documented, ideally with a WSL run), the
build succeeds, the checklist passes, and `CHANGES.md` exists.

---

## Appendix A — The student's lab scripts (MATLAB, for reference)

**`DCT 1.m`**
```matlab
rflower=imread("flower.jpg");
flower=imread("flower.jpg");
flower=rgb2gray(flower);
dctflower=dct2(flower);
dctflower(abs(dctflower)<500)=0;
idctflower=idct2(dctflower);
idctflower=rescale(idctflower);
rgbflower=cat(3, idctflower,idctflower,idctflower);   % gray "RGB"
```

**`single_channel_dct.m`** (produces the colour shift)
```matlab
rflower = imread("flower.jpg");
R = rflower(:,:,1); G = rflower(:,:,2); B = rflower(:,:,3);
dctR = dct2(R); dctG = dct2(G); dctB = dct2(B);
threshold = 500;
dctR(abs(dctR) < threshold) = 0;  dctG(abs(dctG) < threshold) = 0;  dctB(abs(dctB) < threshold) = 0;
R2 = idct2(dctR); G2 = idct2(dctG); B2 = idct2(dctB);
R2 = rescale(R2); G2 = rescale(G2); B2 = rescale(B2);   % <-- per-channel rescale = colour bug
rgbflower = cat(3, R2, G2, B2);
```

**`hadamard_transform.m`**
```matlab
flower=imread("flower.jpg");
flower=rgb2gray(flower);
flower=imresize(flower,[2048,2048]);     % distorts aspect ratio
imgg=im2double(flower);
[r,c]=size(imgg);
Yc=fwht(imgg);  Yr=fwht(Yc');  Y=Yr';    % sequency-ordered, forward scaled by 1/N
Y(256:r,256:c)=0;                        % zeros only the bottom-right block (L-shaped kept region)
Ir1=ifwht(Y);  Ir2=ifwht(Ir1');  imgr=Ir2';
```
