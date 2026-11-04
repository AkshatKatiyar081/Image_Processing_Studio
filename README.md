# Image Processing Studio

Image & Video Processing mini project (NMIMS MPSTME Shirpur, 702EX0E020): a web dashboard for
image enhancement, diagnostics, and a Direct-vs-Guided comparison. It shows why classical
operators fail on noisy or unevenly lit images, and what fixing the image first changes.

```
ivp-studio/
├── pipeline/   Python image-processing package (NumPy + OpenCV), 294 tests, experiments
├── backend/    FastAPI server exposing the pipeline over HTTP, 60 tests
├── frontend/   React + Vite + Tailwind dashboard
└── api/       Vercel serverless entrypoint for the backend
```

```
Browser ──► Frontend (Vercel) ──HTTPS/JSON──► Backend (Vercel: FastAPI + ivp_pipeline)
```

There is no database: every request is processed and returned immediately.

---

## 1. Run it on your computer

**You need:** Python 3.11 (the version Render uses, 3.11.9) and Node.js 18+ (check with `python --version`
and `node --version`). On Windows: `winget install Python.Python.3.11` or `py install 3.11`.

### Step 1: backend (terminal 1)

```bash
cd ivp-studio
py -3.11 -m venv .venv            # macOS / Linux: python3.11 -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -e ./pipeline -r backend/requirements-dev.txt
cd backend
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/api/health. You should see `{"status":"ok",...}`.
Interactive API docs are at http://localhost:8000/docs.

### Step 2: frontend (terminal 2)

```bash
cd ivp-studio/frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The dot in the top-right corner should turn green ("Online").
In development, Vite forwards `/api` to `localhost:8000`, so no configuration is needed.

### Step 3: try it

1. Click the sample **"Shapes, uneven light"**.
2. **Compare** tab → *Segment* → **Run comparison**. Direct Otsu splits the image by
   lighting; the guided route flattens the background first and finds every object (η 0.70 → 0.97).
   Below the verdict: the accuracy-vs-lighting chart, the difference map and the measured bars.
3. Sample **"Cameraman, noisy"** → Compare → *Find edges*. Switch Sobel → Roberts and run again.
4. Sample **"Coffee"** → Compare → *Compression* → **Run comparison**: DCT vs Walsh–Hadamard at the same
   % of coefficients, with quality-vs-compression and energy curves and a linked zoom lens on the errors.
5. Sample **"Astronaut, too dark"** → **Auto Fix** tab.
6. **Tools** tab → *Compression* → *Lab: DCT threshold 500* (the lab script, with correct colours).

### Run the tests

```bash
cd pipeline && python -m pytest -q     # 294 tests (algorithms vs OpenCV/SciPy, guided mode, compression, sweeps, …)
cd ../backend && python -m pytest -q   # 60 API tests (incl. a 12 MP photo through every endpoint)
```

---

## 2. Deploy both services on Vercel

Push the whole repository to GitHub first. Deploy the backend and frontend as two Vercel projects.

### Backend on Vercel

1. Vercel dashboard → **Add New → Project** → import the repository.
2. Leave **Root Directory** as the repository root. Vercel detects `api/index.py` as a Python function.
3. Deploy, then open `https://<backend-project>.vercel.app/api/health`.
4. In the backend project settings, add `ALLOWED_ORIGINS` with the exact URL of your deployed frontend.

### Frontend on Vercel

The existing frontend project should keep **Root Directory** set to `frontend`.

1. Add or update **`VITE_API_URL`** = `https://<backend-project>.vercel.app` (no trailing slash).
2. Redeploy the frontend so Vite embeds the variable in the production bundle.
3. Open the frontend and confirm the health indicator becomes online, and that the upload box shows
   **"Max 4.2 MB · 25 MP"** (the limits come from the backend, see below).

### Image size limits for a hosted backend

Vercel functions accept at most **4.5 MB per request and per reply**. The app is built around that:

* **Replies carry previews only** (≤ 1280 px JPEG; masks as PNG), so every reply stays far below the limit,
  even for a 12 MP photo processed at full resolution: Tools 0.7–1.2 MB, Auto Fix 0.6 MB,
  Compare → Segment 2.2 MB, Compare → Compression 2.6 MB (they used to be 6–113 MB).
* **Full-quality results are separate files**, fetched only for *Download* and *Keep editing this result*
  (`/file` endpoints). A PNG that would be over the limit is sent as JPEG 95 (then 90 / 85 / 80) and the
  dashboard says so.
* **Uploads over the limit are refused in the browser** with a clear message. On Vercel the backend detects
  the platform (`VERCEL` env var) and uses **4.2 MB**; elsewhere 15 MB. Override with `MAX_UPLOAD_MB`.

Optional: `IVP_MAX_SIDE` = `1024` makes processing faster on a small server, but then results and downloads
are at most 1024 px. `VITE_MAX_UPLOAD_SIDE` (e.g. `3072`) shrinks bigger photos in the browser before upload
instead of refusing them (a JPEG stays a JPEG). Long-running requests are still bound by the function's time
and memory limits on your Vercel plan.

---

## 3. What the dashboard does

| Tab | For | What happens |
|---|---|---|
| **Tools** | Applying one technique | Pick a tool (Light & contrast · Remove noise · Sharpen & edges · **Compression**) and an effect. You get a before/after slider, histograms and the numbers; Compression adds coefficient maps, an error map, PSNR/SSIM and the kept share. *Advanced settings* shows every parameter, plus a switch to the from-scratch NumPy engine. *Keep editing this result* chains operations, and *Undo* steps back. |
| **Compare** | The project's main idea | Three comparisons. **Segment** (Otsu) and **Find edges**: the image is diagnosed, the operation runs **directly** and **after the prescribed fix**, and a verdict shows the measured difference, plus an accuracy-vs-degradation chart, a colour-coded difference map and paired bars. **Compression**: DCT vs Walsh–Hadamard at the same % of coefficients, with rate–distortion and energy-compaction curves, error heat maps with a zoom lens, coefficient maps and bars. |
| **Auto Fix** | Casual users | One click: only the fixes the image needs, in the right order, each with the reason it was applied. |

The **Image health** card (left side) is the pipeline's diagnosis in plain words. Each problem has a one-click fix
button. Lighting and separability findings matter only for segmentation, so they appear in Compare → Segment
("Why?") instead.

**Image size.** Run locally, everything works at the **full resolution** of your image (no shrinking). The
limits are settings:

| Setting | Where | Local default | Vercel |
|---|---|---|---|
| `MAX_UPLOAD_MB` | backend env | 15 | 4.2 (automatic) |
| `MAX_DOWNLOAD_MB` | backend env | = `MAX_UPLOAD_MB` | = `MAX_UPLOAD_MB` |
| `MAX_PIXELS` | backend env | 25 000 000 | 25 000 000 |
| `PREVIEW_SIDE` | backend env | 1280 | 1280 |
| `IVP_MAX_SIDE` | backend env | `0` = full resolution | optional, e.g. `1024` |
| `VITE_MAX_UPLOAD_SIDE` | frontend env (build time) | `0` = upload full size | optional |

The dashboard reads the upload and pixel limits from `/api/health`, shows them in the upload box and refuses
larger files with a message. When a size limit shrinks an image, nearest-neighbour sampling is used so
salt-and-pepper specks survive. `MAX_PIXELS` refuses images over 25 MP before decoding them.
Full resolution is slower: on a 3072 × 2304 photo, DCT compression takes about 4 s, Hadamard about 8 s
(padded to 4096 × 4096), and Compare → Compression about 12–16 s. The chart curves and the degradation sweep
always use a ≤ 512 px copy for speed (the UI says so).

## 4. API reference

| Method & path | Form fields | Returns |
|---|---|---|
| `GET /api/health` | – | `{status, version}` |
| `GET /api/modules` | – | parameter schema of every module |
| `GET /api/samples`, `GET /api/samples/{id}.png` | – | demo images |
| `POST /api/diagnose` | `file` | `{metrics, findings, meta}` |
| `POST /api/process` | `file`, `module`, `params` (JSON), `engine` | result image (PNG data URL), extras (mask, cut-out, edge map, LUT), histograms, metrics, warnings, meta |
| `POST /api/guided` | `file`, `intent`, `params`, `engine` | `gate_passed`, `diagnosis`, `branch_a`, `branch_b` (with stages), `verdict` |
| `POST /api/auto` | `file`, `engine` | fixed image, `steps`, diagnosis before/after, histograms |
| `POST /api/sweep` | `file`, `intent` (`segment` or `edges`), `params`, `engine` | `axis`, `metric`, `points[{level, x, direct, guided, gate_fired}]`, `user_point_index`, `reference_note` (numbers only) |
| `POST /api/compare-compression` | `file`, `keep_percent`, `jpeg_quality`, `engine` | `dct`, `hadamard`, `jpeg` (preview, coefficient maps, error map, metrics), `rd_curve`, `energy_curve`, `size_curve`, `curve_scale`, `summary` |
| `POST /api/inspect` | `file` | format, size, pixels, bits per pixel, JPEG quality estimate and subsampling, metadata |
| `POST /api/compress` | `file`, `params`, `engine` | preview of the compressed FILE, its metrics, `file` (format, bytes, quality, `downloadable`, `note`), `upload` |
| `POST /api/process/file`, `/api/auto/file`, `/api/compress/file` | same as without `/file` | the **full-quality result file** |
| `POST /api/guided/file` | as `/api/guided` + `branch` (`a`/`b`), `layer` (`image`/`mask`) | full-quality result of one route |
| `POST /api/compare-compression/file` | as `/api/compare-compression` + `method` | full-quality result of one method (JPEG-style: the real JPEG file) |

JSON replies carry **previews** (≤ 1280 px, `PREVIEW_SIDE`): JPEG for photos, PNG for masks and edge maps.
The `/file` endpoints run the same operation again and answer with the file itself (PNG, or JPEG 95…80 when the
PNG is over `MAX_DOWNLOAD_MB`); the headers `X-IVP-Format`, `X-IVP-Quality` and `X-IVP-Note` (URL-encoded) say
which, and are exposed through CORS. `/api/health` also returns `limits` (`max_upload_bytes`,
`max_download_bytes`, `max_pixels`). `/api/guided` also returns `difference_map`, `difference_legend` and
`comparison` (counts for the bars) when the guided route ran.

Errors come back as HTTP 422 with `{"error": {"code", "message"}}` (for example `TOO_LARGE`, `RESULT_TOO_LARGE`,
`BAD_PARAM`, `BAD_REQUEST`). Unexpected crashes return HTTP 500 with code `INTERNAL`, still with CORS headers, so the browser
shows the message instead of a CORS error.

## 5. Project docs

* `pipeline/README.md`: pipeline API, layout, calibrated thresholds, compression transforms, sweeps, known limitations.
* `CHANGES.md`: what the v3 work changed (features, fixes, tests, memory measurements, suggested commits).
* `pipeline/experiments/results/`: calibration and benchmark tables, runtime, and the figures for the report.
