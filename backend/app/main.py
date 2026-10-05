"""FastAPI backend for IVP Studio. A thin HTTP layer over the ivp_pipeline package.

Run locally:   uvicorn app.main:app --reload --port 8000     (from the backend/ folder)
Env vars:      ALLOWED_ORIGINS  comma-separated list for CORS (default "*")
               MAX_UPLOAD_MB    upload size limit (default 15)
               MAX_PIXELS       largest decoded image, width x height (default 25000000)
"""
from __future__ import annotations

import json
import logging
import os
import time

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import base64

import cv2
import numpy as np

import ivp_pipeline as ivp
from ivp_pipeline import PipelineError
from ivp_pipeline import metrics as Q
from ivp_pipeline.modules import compress as Z

from . import samples
from .codec import decode_upload, encode, jsonable, module_result
from .files import MIME, EXTENSIONS, inspect_upload, output_plan, save_like

log = logging.getLogger("ivp")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ENGINES = ("opencv", "numpy")

app = FastAPI(title="IVP Studio API", version=ivp.__version__,
              description="Image & Video Processing mini project — enhancement, diagnostics, guided comparison.")


# ------------------------------------------------------------------ errors
def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(PipelineError)
async def pipeline_error(_: Request, e: PipelineError):
    return _error(422, e.code, e.message)


@app.exception_handler(RequestValidationError)
async def request_error(_: Request, e: RequestValidationError):
    """Missing/invalid form fields get the same {"error": {...}} shape as every other error."""
    first = e.errors()[0] if e.errors() else {}
    field = ".".join(str(x) for x in first.get("loc", [])[1:]) or "request"
    return _error(422, "BAD_REQUEST", f"{field}: {first.get('msg', 'invalid request')}")


# Unexpected crashes are turned into JSON here, in a middleware that sits INSIDE CORSMiddleware,
# so the 500 response still carries the CORS headers. (A plain @app.exception_handler(Exception)
# runs in Starlette's ServerErrorMiddleware, outside CORS: the browser would then report a CORS
# error instead of this message.) In Starlette the LAST added middleware is the OUTERMOST, so this
# one must be registered before app.add_middleware(CORSMiddleware, ...).
@app.middleware("http")
async def unexpected_error(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        log.exception("unhandled error")
        return _error(500, "INTERNAL", "Something went wrong while processing the image. Please try again.")


app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"], max_age=3600)


# ------------------------------------------------------------------ helpers
def _read_bytes(file: UploadFile) -> bytes:
    data = file.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise PipelineError("TOO_LARGE", f"Image is larger than {MAX_UPLOAD_MB:g} MB.")
    return data


def _read_image(file: UploadFile):
    return decode_upload(_read_bytes(file))


def _inspect(data: bytes) -> dict:
    try:
        return inspect_upload(data)
    except Exception:
        raise PipelineError("BAD_IMAGE", "Could not read the file as an image. Use PNG, JPEG, BMP, TIFF or WebP.") from None


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


def _float(name: str, raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise PipelineError("BAD_PARAM", f"{name} must be a number, got {raw!r}") from None


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
    if "difference_map" in res:
        out["difference_map"] = encode(res["difference_map"], "png")
        out["difference_legend"] = res["difference_legend"]
        out["comparison"] = jsonable(res["comparison"])
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


@app.post("/api/sweep")
def sweep(file: UploadFile = File(...), intent: str = Form(...), params: str = Form("{}"),
          engine: str = Form("opencv")):
    """Accuracy vs added degradation, Direct vs Guided. Numbers only (no images)."""
    img = _read_image(file)
    t0 = time.perf_counter()
    out = jsonable(ivp.degradation_sweep(img, intent, _params(params), _engine(engine)))
    out["meta"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.post("/api/compare-compression")
def compare_compression(file: UploadFile = File(...), keep_percent: str = Form("10"),
                        jpeg_quality: str = Form("auto"), engine: str = Form("opencv")):
    """DCT vs Walsh–Hadamard at the same % of coefficients kept, plus JPEG-style coding.
    jpeg_quality: "auto" (match the DCT result's PSNR) or 1–100. Images are PNG (artefacts must be exact)."""
    img = _read_image(file)
    q = None if jpeg_quality.strip().lower() in ("", "auto") else _float("jpeg_quality", jpeg_quality)
    t0 = time.perf_counter()
    res = ivp.compare_compression(img, _float("keep_percent", keep_percent), _engine(engine), q)
    out = {
        "dct": module_result(res["dct"], "png"),
        "hadamard": module_result(res["hadamard"], "png"),
        "jpeg": module_result(res["jpeg"], "png"),
        "rd_curve": jsonable(res["rd_curve"]),
        "energy_curve": jsonable(res["energy_curve"]),
        "size_curve": jsonable(res["size_curve"]),
        "size_curve_note": res["size_curve_note"],
        "curve_scale": res["curve_scale"],
        "summary": jsonable(res["summary"]),
        "keep_percent": res["keep_percent"],
        "meta": jsonable(res["meta"]),
    }
    if res["meta"]["scale"] != 1.0:
        out["input"] = encode(res["input"], "png")
    out["meta"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.post("/api/inspect")
def inspect(file: UploadFile = File(...)):
    """What the uploaded file is: format, pixel size, file size, JPEG quality / subsampling, metadata."""
    return jsonable(_inspect(_read_bytes(file)))


@app.post("/api/compress")
def compress(file: UploadFile = File(...), params: str = Form("{}"), engine: str = Form("opencv")):
    """Compress the upload and return a real, downloadable FILE (by default in the upload's format).

    JPEG-style method: the input is saved as a JPEG at the chosen quality and colour subsampling.
    Whole-image DCT / Hadamard: the method's result is saved in the upload's format (a JPEG upload at
    its own quality). The image shown, PSNR, SSIM and the error map all describe the downloaded file.
    """
    data = _read_bytes(file)
    info = _inspect(data)
    img = decode_upload(data)
    t0 = time.perf_counter()
    res = ivp.run_module(img, "compress", _params(params), _engine(engine))
    p = res["meta"]["params"]
    plan = output_plan(info, p)
    source = res["input"] if p["method"] == "jpeg" else res["image"]
    blob = save_like(source, data, plan["format"], plan["quality"], plan["subsampling"])

    # decode the file exactly as a viewer would, and measure THAT against the input
    buf = np.frombuffer(blob, np.uint8)
    if source.ndim == 2:
        shown = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    else:
        shown = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    work = res["input"]
    err = Z.abs_error(work, shown)
    amp = Z.error_amplification(err)
    res["extras"]["error_map"] = Z.error_map(err, amp)
    method_psnr = res["metrics"].get("psnr_db")
    res["metrics"].update({"psnr_db": round(Q.psnr(work, shown), 2), "ssim": round(Q.ssim(work, shown), 4),
                           "error_amplification": amp})
    if p["method"] != "jpeg":
        res["metrics"]["method_psnr_db"] = method_psnr          # before saving the file

    out = module_result(res, "png")
    out["image"] = f"data:{MIME[plan['format']]};base64,{base64.b64encode(blob).decode('ascii')}"
    out["file"] = {
        "format": plan["format"], "mime": MIME[plan["format"]], "extension": EXTENSIONS[plan["format"]],
        "bytes": len(blob), "quality": plan["quality"], "chroma_subsampling": plan["subsampling"],
        "source": plan["source"],
        "original_format": info["format"], "original_bytes": info["file_bytes"],
        "size_percent": round(100.0 * len(blob) / max(1, info["file_bytes"]), 1),
    }
    if len(blob) >= info["file_bytes"]:
        why = (" PNG is lossless, so it cannot get much smaller; use the JPEG-style method for a small file."
               if plan["format"] == "PNG" else
               " Zeroing transform coefficients alone does not shrink a file much; the JPEG-style method does.")
        out["warnings"] = out["warnings"] + [
            f"The compressed file ({len(blob) / 1e6:.2f} MB) is not smaller than your file "
            f"({info['file_bytes'] / 1e6:.2f} MB).{why}"]
    out["upload"] = jsonable(info)
    out["meta"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
