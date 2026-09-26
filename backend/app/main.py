"""FastAPI backend for IVP Studio. A thin HTTP layer over the ivp_pipeline package.

Run locally:   uvicorn app.main:app --reload --port 8000     (from the backend/ folder)
Env vars:      ALLOWED_ORIGINS  comma-separated list for CORS (default "*")
               MAX_UPLOAD_MB    upload size limit (default 15)
"""
from __future__ import annotations

import json
import logging
import os
import time

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import ivp_pipeline as ivp
from ivp_pipeline import PipelineError

from . import samples
from .codec import decode_upload, encode, jsonable, module_result

log = logging.getLogger("ivp")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ENGINES = ("opencv", "numpy")

app = FastAPI(title="IVP Studio API", version=ivp.__version__,
              description="Image & Video Processing mini project — enhancement, diagnostics, guided comparison.")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"], max_age=3600)


# ------------------------------------------------------------------ errors
def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(PipelineError)
async def pipeline_error(_: Request, e: PipelineError):
    return _error(422, e.code, e.message)


@app.exception_handler(Exception)
async def unexpected_error(_: Request, e: Exception):  # pragma: no cover - safety net
    log.exception("unhandled error")
    return _error(500, "INTERNAL", "Something went wrong while processing the image. Please try again.")


# ------------------------------------------------------------------ helpers
def _read_image(file: UploadFile):
    data = file.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise PipelineError("TOO_LARGE", f"Image is larger than {MAX_UPLOAD_MB:g} MB.")
    return decode_upload(data)


def _params(raw: str) -> dict:
    try:
        p = json.loads(raw or "{}")
    except json.JSONDecodeError:
        raise PipelineError("BAD_PARAM", "params must be a JSON object") from None
    if not isinstance(p, dict):
        raise PipelineError("BAD_PARAM", "params must be a JSON object")
    return p


def _engine(engine: str) -> str:
    if engine not in ENGINES:
        raise PipelineError("BAD_ENGINE", f"engine must be one of {list(ENGINES)}")
    return engine


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return {"status": "ok", "version": ivp.__version__}


@app.get("/api/modules")
def modules():
    return jsonable(ivp.list_modules())


@app.get("/api/samples")
def sample_list():
    return samples.listing()


@app.get("/api/samples/{sample_id}.png")
def sample_image(sample_id: str):
    s = samples.catalogue().get(sample_id)
    if s is None:
        return _error(404, "NOT_FOUND", f"No sample called {sample_id!r}")
    return Response(content=s["png"], media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/diagnose")
def diagnose(file: UploadFile = File(...)):
    img = _read_image(file)
    return jsonable(ivp.diagnose(img))


@app.post("/api/process")
def process(file: UploadFile = File(...), module: str = Form(...), params: str = Form("{}"),
            engine: str = Form("opencv")):
    img = _read_image(file)
    t0 = time.perf_counter()
    res = ivp.run_module(img, module, _params(params), _engine(engine))
    out = module_result(res, "png", include_input=res["meta"]["scale"] != 1.0)
    out["meta"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.post("/api/guided")
def guided(file: UploadFile = File(...), intent: str = Form(...), params: str = Form("{}"),
           engine: str = Form("opencv")):
    img = _read_image(file)
    t0 = time.perf_counter()
    res = ivp.run_guided(img, intent, _params(params), _engine(engine))
    out = {
        "gate_passed": res["gate_passed"],
        "intent": res["intent"],
        "diagnosis": jsonable(res["diagnosis"]),
        "verdict": jsonable(res["verdict"]),
        "branch_a": module_result(res["branch_a"], "jpeg"),
        "branch_b": None,
        "meta": jsonable(res["meta"]),
    }
    if res["branch_b"] is not None:
        b = module_result(res["branch_b"], "jpeg")
        b["stages"] = [{"name": s["name"], "rule": s["rule"], "reason": s["reason"],
                        "params": jsonable(s["params"]), "image": encode(s["image"], "jpeg")}
                       for s in res["branch_b"]["stages"]]
        out["branch_b"] = b
    if res["meta"]["scale"] != 1.0:
        out["input"] = encode(res["input"], "png")
    out["meta"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.post("/api/auto")
def auto(file: UploadFile = File(...), engine: str = Form("opencv")):
    img = _read_image(file)
    t0 = time.perf_counter()
    res = ivp.auto_enhance(img, _engine(engine))
    out = {
        "image": encode(res["image"], "png"),
        "steps": jsonable(res["steps"]),
        "diagnosis_before": jsonable(res["diagnosis_before"]),
        "diagnosis_after": jsonable(res["diagnosis_after"]),
        "histograms": res["histograms"],
        "meta": jsonable(res["meta"]),
    }
    if res["meta"]["scale"] != 1.0:
        out["input"] = encode(res["input"], "png")
    out["meta"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
