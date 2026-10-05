# IVP Studio

Image & Video Processing mini project (NMIMS MPSTME Shirpur, 702EX0E020): a web dashboard for
image enhancement, diagnostics, and a Direct-vs-Guided comparison. It shows why classical
operators fail on noisy or unevenly lit images, and what fixing the image first changes.

```
ivp-studio/
├── pipeline/   Python image-processing package (NumPy + OpenCV), 268 tests, experiments
├── backend/    FastAPI server exposing the pipeline over HTTP, 47 tests
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
cd pipeline && python -m pytest -q     # 268 tests (algorithms vs OpenCV/SciPy, guided mode, compression, sweeps, …)
cd ../backend && python -m pytest -q   # 47 API tests
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
2. Add **`VITE_MAX_UPLOAD_SIDE`** = `2048`, so big photos are shrunk in the browser before upload.
3. Redeploy the frontend so Vite embeds the variables in the production bundle.
4. Open the frontend and confirm the health indicator becomes online.

### Image size limits for a hosted backend

Locally, everything runs at full resolution. On a hosted backend, set **`IVP_MAX_SIDE`** = `1024` in the
backend project's environment variables (Render: already in `render.yaml`), together with
`VITE_MAX_UPLOAD_SIDE` = `2048` on the frontend (see *Image size* in section 3).

**Note:** Vercel Python functions are request-based and have execution and package-size limits, and a
request or response body may be at most about 4.5 MB. Full-resolution results (several images per response,
especially Compare → Compression) can exceed that, which is another reason to use the limits above. Large
images or long-running processing may need a dedicated Python host such as Render.

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
limits are settings, used only for the hosted version:

| Setting | Where | Local default | Render / Vercel |
|---|---|---|---|
| `IVP_MAX_SIDE` | backend env | `0` = full resolution | `1024` (in `render.yaml`) |
| `VITE_MAX_UPLOAD_SIDE` | frontend env (build time) | `0` = upload full size | `2048` |
| `MAX_PIXELS` | backend env | 25 000 000 | 25 000 000 |

When a limit is set, photos are shrunk with nearest-neighbour sampling so salt-and-pepper specks survive.
`MAX_PIXELS` refuses images over 25 MP before decoding them (raise it for larger phone photos).
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
| `POST /api/compare-compression` | `file`, `keep_percent`, `engine` | `dct`, `hadamard` (image, coefficient maps, error map, metrics), `rd_curve`, `energy_curve`, `curve_scale`, `summary` |

`/api/process` accepts `module="compress"`. `/api/guided` also returns `difference_map` (PNG), `difference_legend`
and `comparison` (counts for the bars) when the guided route ran.

Errors come back as HTTP 422 with `{"error": {"code", "message"}}` (for example `TOO_LARGE`, `BAD_PARAM`,
`BAD_REQUEST`). Unexpected crashes return HTTP 500 with code `INTERNAL`, still with CORS headers, so the browser
shows the message instead of a CORS error.

## 5. Project docs

* `pipeline/README.md`: pipeline API, layout, calibrated thresholds, compression transforms, sweeps, known limitations.
* `CHANGES.md`: what the v3 work changed (features, fixes, tests, memory measurements, suggested commits).
* `pipeline/experiments/results/`: calibration and benchmark tables, runtime, and the figures for the report.
