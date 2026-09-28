# IVP Studio

Image & Video Processing mini project (NMIMS MPSTME Shirpur, 702EX0E020): a web dashboard for
image enhancement, diagnostics, and a Direct-vs-Guided comparison. It shows why classical
operators fail on noisy or unevenly lit images, and what fixing the image first changes.

```
ivp-studio/
├── pipeline/   Python image-processing package (NumPy + OpenCV), 131 tests, experiments
├── backend/    FastAPI server exposing the pipeline over HTTP, 19 tests
├── frontend/   React + Vite + Tailwind dashboard
└── api/       Vercel serverless entrypoint for the backend
```

```
Browser ──► Frontend (Vercel) ──HTTPS/JSON──► Backend (Vercel: FastAPI + ivp_pipeline)
```

There is no database: every request is processed and returned immediately.

---

## 1. Run it on your computer

**You need:** Python 3.10–3.12 and Node.js 18+ (check with `python --version` and `node --version`).

### Step 1: backend (terminal 1)

```bash
cd ivp-studio
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install ./pipeline -r backend/requirements.txt
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
2. **Compare** tab → *Segment objects* → **Run comparison**. Direct Otsu splits the image by
   lighting; the guided route flattens the background first and finds every object (η 0.70 → 0.97).
3. Sample **"Cameraman, noisy"** → Compare → *Find edges*.
4. Sample **"Astronaut, too dark"** → **Auto Fix** tab.
5. **Tools** tab → *Cut out object* → *Click to select*, then click an object.

### Run the tests

```bash
pip install pytest httpx
cd pipeline && python -m pytest -q     # 131 tests (algorithms vs OpenCV, guided mode, …)
cd ../backend && python -m pytest -q   # 19 API tests
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
3. Open the frontend and confirm the health indicator becomes online.

**Note:** Vercel Python functions are request-based and have execution and package-size limits. Large
images or long-running processing may need a dedicated Python host such as Render.

---

## 3. What the dashboard does

| Tab | For | What happens |
|---|---|---|
| **Tools** | Applying one technique | Pick a tool (Light & contrast · Remove noise · Sharpen & edges · Cut out object) and an effect. You get a before/after slider, histograms and the numbers. *Advanced settings* shows every parameter, plus a switch to the from-scratch NumPy engine. *Keep editing this result* chains operations, and *Undo* steps back. |
| **Compare** | The project's main idea | Choose what you want to do (segment, find edges, sharpen, equalize). The image is diagnosed, the operation runs **directly** and **after the prescribed fix**, and a verdict shows the measured difference. |
| **Auto Fix** | Casual users | One click: only the fixes the image needs, in the right order, each with the reason it was applied. |

The **Image health** card (left side) is the pipeline's diagnosis in plain words. Each problem has a one-click fix button.

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

Errors come back as HTTP 422 with `{"error": {"code", "message"}}`.

## 5. Project docs

* `pipeline/README.md`: pipeline API, layout, calibrated thresholds, known limitations.
* `pipeline/experiments/results/`: calibration and benchmark tables, runtime, and the figures for the report.
