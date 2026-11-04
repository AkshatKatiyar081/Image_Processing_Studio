"""FastAPI backend for Image Processing Studio. A thin HTTP layer over the ivp_pipeline package.

Run locally:   uvicorn app.main:app --reload --port 8000     (from the backend/ folder)
Env vars:      ALLOWED_ORIGINS  comma-separated list for CORS (default "*")
               MAX_UPLOAD_MB    upload size limit (default 15; 4.2 on Vercel, see limits.py)
               MAX_DOWNLOAD_MB  largest file returned by the /file endpoints (default = MAX_UPLOAD_MB)
               MAX_PIXELS       largest decoded image, width x height (default 25000000)
               PREVIEW_SIDE     longest side of the preview images in JSON replies (default 1280)

JSON replies carry small PREVIEW images (codec.preview), never full-resolution ones: a screen cannot show
more, and hosted functions limit the reply size. The full-quality result of any operation is returned by
the matching "/file" endpoint, which runs the same operation again and answers with the file itself
(PNG, or JPEG if the PNG would be over MAX_DOWNLOAD_MB; the X-IVP-Note header then says so).
"""
from __future__ import annotations

import json
import logging
import os
import time
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import ivp_pipeline as ivp
from ivp_pipeline import PipelineError
from ivp_pipeline import metrics as Q
from ivp_pipeline.compare_compression import match_jpeg_quality
from ivp_pipeline.modules import compress as Z

from . import samples
from .codec import MAX_PIXELS, decode_upload, jsonable, module_result, preview
from .files import EXTENSIONS, MIME, fit_file, inspect_upload, output_plan, save_like
from .limits import MAX_DOWNLOAD_BYTES, MAX_UPLOAD_BYTES, ON_VERCEL, mb

log = logging.getLogger("ivp")
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ENGINES = ("opencv", "numpy")
FILE_HEADERS = ["X-IVP-Format", "X-IVP-Quality", "X-IVP-Note"]
# Compare → Compression sends three results and three error maps in one reply: previews of at most 1024 px at
# JPEG quality 92 keep it under ~3 MB for a 12 MP photo, while their own artefacts stay far below the
# compression artefacts being compared.
COMPARE_PREVIEW_SIDE, COMPARE_PREVIEW_QUALITY = 1024, 92

app = FastAPI(title="Image Processing Studio API", version=ivp.__version__,
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
                   allow_headers=["*"], expose_headers=FILE_HEADERS, max_age=3600)


# ------------------------------------------------------------------ helpers
def _read_bytes(file: UploadFile) -> bytes:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise PipelineError("TOO_LARGE", f"The file is larger than the {mb(MAX_UPLOAD_BYTES)} upload limit.")
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


def _jpeg_quality(raw: str) -> float | None:
    return None if raw.strip().lower() in ("", "auto") else _float("jpeg_quality", raw)


def _file_response(fitted: dict) -> Response:
    """A full-quality result file. The format/quality/note travel in headers (exposed through CORS)."""
    headers = {"X-IVP-Format": fitted["format"], "X-IVP-Quality": str(fitted["quality"] or ""),
               "X-IVP-Note": quote(fitted["note"] or "")}
    return Response(content=fitted["bytes"], media_type=MIME[fitted["format"]], headers=headers)


def _result_file(img: np.ndarray, original: bytes) -> Response:
    """Full-resolution result as lossless PNG, or JPEG 95…80 if the PNG is over the download limit."""
    return _file_response(fit_file(img, original, MAX_DOWNLOAD_BYTES, "PNG"))


def _done(out: dict, t0: float) -> dict:
    out["meta"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return {"status": "ok", "version": ivp.__version__,
            "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_download_bytes": MAX_DOWNLOAD_BYTES,
                       "max_pixels": MAX_PIXELS, "hosted_on_vercel": ON_VERCEL}}


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


# ---------------------------------------------------------- Tools: one module
@app.post("/api/process")
def process(file: UploadFile = File(...), module: str = Form(...), params: str = Form("{}"),
            engine: str = Form("opencv")):
    img = _read_image(file)
    t0 = time.perf_counter()
    res = ivp.run_module(img, module, _params(params), _engine(engine))
    return _done(module_result(res, drop=("cutout",)), t0)


@app.post("/api/process/file")
def process_file(file: UploadFile = File(...), module: str = Form(...), params: str = Form("{}"),
                 engine: str = Form("opencv")):
    data = _read_bytes(file)
    res = ivp.run_module(decode_upload(data), module, _params(params), _engine(engine))
    return _result_file(res["image"], data)


# ---------------------------------------------------------- Compare: direct vs guided
def _guided(img, intent, params, engine):
    return ivp.run_guided(img, intent, _params(params), _engine(engine))


@app.post("/api/guided")
def guided(file: UploadFile = File(...), intent: str = Form(...), params: str = Form("{}"),
           engine: str = Form("opencv")):
    img = _read_image(file)
    t0 = time.perf_counter()
    res = _guided(img, intent, params, engine)
    drop = ("cutout", "raw_mask", "background")
    out = {
        "gate_passed": res["gate_passed"],
        "intent": res["intent"],
        "diagnosis": jsonable(res["diagnosis"]),
        "verdict": jsonable(res["verdict"]),
        "branch_a": module_result(res["branch_a"], drop=drop),
        "branch_b": None,
        "meta": jsonable(res["meta"]),
    }
    if "difference_map" in res:
        out["difference_map"] = preview(res["difference_map"])
        out["difference_legend"] = res["difference_legend"]
        out["comparison"] = jsonable(res["comparison"])
    if res["branch_b"] is not None:
        b = module_result(res["branch_b"], drop=drop)
        b["stages"] = [{"name": s["name"], "rule": s["rule"], "reason": s["reason"],
                        "params": jsonable(s["params"]), "image": preview(s["image"], 512)}
                       for s in res["branch_b"]["stages"]]
        out["branch_b"] = b
    return _done(out, t0)


@app.post("/api/guided/file")
def guided_file(file: UploadFile = File(...), intent: str = Form(...), params: str = Form("{}"),
                engine: str = Form("opencv"), branch: str = Form("a"), layer: str = Form("image")):
    """Full-quality result of one route: branch "a" (direct) or "b" (guided); layer "image" or "mask"."""
    if branch not in ("a", "b") or layer not in ("image", "mask"):
        raise PipelineError("BAD_PARAM", "branch must be 'a' or 'b' and layer 'image' or 'mask'")
    data = _read_bytes(file)
    res = _guided(decode_upload(data), intent, params, engine)
    route = res["branch_b"] if branch == "b" else res["branch_a"]
    if route is None:
        raise PipelineError("BAD_PARAM", "The guided route did not run for this image (no fix was needed).")
    img = route["extras"]["mask"] if layer == "mask" else route["image"]
    return _result_file(img, data)


# ---------------------------------------------------------- Auto Fix
@app.post("/api/auto")
def auto(file: UploadFile = File(...), engine: str = Form("opencv")):
    img = _read_image(file)
    t0 = time.perf_counter()
    res = ivp.auto_enhance(img, _engine(engine))
    out = {
        "image": preview(res["image"]),
        "steps": jsonable(res["steps"]),
        "diagnosis_before": jsonable(res["diagnosis_before"]),
        "diagnosis_after": jsonable(res["diagnosis_after"]),
        "histograms": res["histograms"],
        "meta": jsonable(res["meta"]),
    }
    return _done(out, t0)


@app.post("/api/auto/file")
def auto_file(file: UploadFile = File(...), engine: str = Form("opencv")):
    data = _read_bytes(file)
    return _result_file(ivp.auto_enhance(decode_upload(data), _engine(engine))["image"], data)


# ---------------------------------------------------------- Compare: compression
@app.post("/api/sweep")
def sweep(file: UploadFile = File(...), intent: str = Form(...), params: str = Form("{}"),
          engine: str = Form("opencv")):
    """Accuracy vs added degradation, Direct vs Guided. Numbers only (no images)."""
    img = _read_image(file)
    t0 = time.perf_counter()
    return _done(jsonable(ivp.degradation_sweep(img, intent, _params(params), _engine(engine))), t0)


@app.post("/api/compare-compression")
def compare_compression(file: UploadFile = File(...), keep_percent: str = Form("10"),
                        jpeg_quality: str = Form("auto"), engine: str = Form("opencv")):
    """DCT vs Walsh–Hadamard at the same % of coefficients kept, plus JPEG-style coding.
    jpeg_quality: "auto" (match the DCT result's PSNR) or 1–100. Previews: see COMPARE_PREVIEW_SIDE."""
    img = _read_image(file)
    t0 = time.perf_counter()
    res = ivp.compare_compression(img, _float("keep_percent", keep_percent), _engine(engine), _jpeg_quality(jpeg_quality))
    out = {m: module_result(res[m], quality=COMPARE_PREVIEW_QUALITY, side=COMPARE_PREVIEW_SIDE) for m in ("dct", "hadamard", "jpeg")}
    out.update({
        "rd_curve": jsonable(res["rd_curve"]),
        "energy_curve": jsonable(res["energy_curve"]),
        "size_curve": jsonable(res["size_curve"]),
        "size_curve_note": res["size_curve_note"],
        "curve_scale": res["curve_scale"],
        "summary": jsonable(res["summary"]),
        "keep_percent": res["keep_percent"],
        "meta": jsonable(res["meta"]),
    })
    return _done(out, t0)


@app.post("/api/compare-compression/file")
def compare_compression_file(file: UploadFile = File(...), method: str = Form(...), keep_percent: str = Form("10"),
                             jpeg_quality: str = Form("auto"), engine: str = Form("opencv")):
    """Full-quality reconstruction of ONE method of the comparison (dct, hadamard or jpeg), computed exactly
    as /api/compare-compression does, without the curves."""
    if method not in ("dct", "hadamard", "jpeg"):
        raise PipelineError("BAD_PARAM", "method must be 'dct', 'hadamard' or 'jpeg'")
    data = _read_bytes(file)
    img, eng, kp = decode_upload(data), _engine(engine), _float("keep_percent", keep_percent)
    if method != "jpeg":
        res = ivp.run_module(img, "compress", {"method": method, "mode": "keep_percent", "keep_percent": kp}, eng)
        return _result_file(res["image"], data)
    q = _jpeg_quality(jpeg_quality)
    if q is None:   # "auto": the lowest quality that matches the DCT result's PSNR, as in the comparison
        dct = ivp.run_module(img, "compress", {"method": "dct", "mode": "keep_percent", "keep_percent": kp}, eng)
        target = dct["metrics"]["psnr_db"]
        q = match_jpeg_quality(dct["input"], target if np.isfinite(target) else 99.0)
    # JPEG-style: the download IS the JPEG file the comparison measured (libjpeg, 4:2:0), not a re-encoded copy
    res = ivp.run_module(img, "compress", {"method": "jpeg", "quality": int(q), "chroma_subsampling": "4:2:0"}, eng)
    blob = Z.encode_jpeg_cv2(res["input"], int(q), "4:2:0")
    if MAX_DOWNLOAD_BYTES and len(blob) > MAX_DOWNLOAD_BYTES:
        raise PipelineError("RESULT_TOO_LARGE", f"The JPEG file is {mb(len(blob))}, over the "
                                                f"{mb(MAX_DOWNLOAD_BYTES)} download limit. Choose a lower quality.")
    return _file_response({"bytes": blob, "format": "JPEG", "quality": int(q), "note": None})


# ---------------------------------------------------------- Tools: compression as a real file
@app.post("/api/inspect")
def inspect(file: UploadFile = File(...)):
    """What the uploaded file is: format, pixel size, file size, JPEG quality / subsampling, metadata."""
    return jsonable(_inspect(_read_bytes(file)))


def _compress(data: bytes, params: str, engine: str) -> dict:
    """Compress the upload into a real file (shared by /api/compress and /api/compress/file).

    JPEG-style method: the input is saved as a JPEG at the chosen quality and colour subsampling. If that file
    is over the download limit it can still be previewed, but not downloaded (lower the quality).
    Whole-image DCT / Hadamard: the result is saved in the upload's format (a JPEG upload at its own quality);
    if that file is over the download limit, it is saved as JPEG 95…80 instead (fit_file), with a note.
    """
    info = _inspect(data)
    res = ivp.run_module(decode_upload(data), "compress", _params(params), _engine(engine))
    p = res["meta"]["params"]
    plan = output_plan(info, p)
    if p["method"] == "jpeg":
        blob = save_like(res["input"], data, plan["format"], plan["quality"], plan["subsampling"])
        fitted = {"bytes": blob, "format": "JPEG", "quality": plan["quality"], "subsampling": plan["subsampling"],
                  "note": None}
        downloadable = not MAX_DOWNLOAD_BYTES or len(blob) <= MAX_DOWNLOAD_BYTES
    else:
        fitted = fit_file(res["image"], data, MAX_DOWNLOAD_BYTES, plan["format"], plan["quality"], plan["subsampling"])
        downloadable = True
    return {"info": info, "res": res, "plan": plan, "fitted": fitted, "downloadable": downloadable}


@app.post("/api/compress")
def compress(file: UploadFile = File(...), params: str = Form("{}"), engine: str = Form("opencv")):
    """Compress the upload and describe the real, downloadable FILE (fetch it from /api/compress/file).
    The preview, PSNR, SSIM and the error map all describe that file, decoded as a viewer would."""
    data = _read_bytes(file)
    t0 = time.perf_counter()
    c = _compress(data, params, engine)
    res, plan, fitted, info = c["res"], c["plan"], c["fitted"], c["info"]
    blob, work = fitted["bytes"], res["input"]

    buf = np.frombuffer(blob, np.uint8)
    if work.ndim == 2:
        shown = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    else:
        shown = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    err = Z.abs_error(work, shown)
    amp = Z.error_amplification(err)
    res["extras"]["error_map"] = Z.error_map(err, amp)
    method_psnr = res["metrics"].get("psnr_db")
    res["metrics"].update({"psnr_db": round(Q.psnr(work, shown), 2), "ssim": round(Q.ssim(work, shown), 4),
                           "error_amplification": amp})
    if res["meta"]["params"]["method"] != "jpeg":
        res["metrics"]["method_psnr_db"] = method_psnr          # before saving the file
    res["image"] = shown

    out = module_result(res)
    fmt = fitted["format"]
    out["file"] = {
        "format": fmt, "mime": MIME[fmt], "extension": EXTENSIONS[fmt],
        "bytes": len(blob), "quality": fitted["quality"], "chroma_subsampling": fitted["subsampling"],
        "source": plan["source"], "note": fitted["note"], "downloadable": c["downloadable"],
        "original_format": info["format"], "original_bytes": info["file_bytes"],
        "size_percent": round(100.0 * len(blob) / max(1, info["file_bytes"]), 1),
    }
    warnings = list(out["warnings"])
    if fitted["note"]:
        warnings.append(fitted["note"])
    if not c["downloadable"]:
        warnings.append(f"The compressed file is {mb(len(blob))}, over the {mb(MAX_DOWNLOAD_BYTES)} download limit, "
                        "so it can be previewed but not downloaded. Choose a lower quality.")
    if len(blob) >= info["file_bytes"]:
        why = (" PNG is lossless, so it cannot get much smaller; use the JPEG-style method for a small file."
               if fmt == "PNG" else
               " Zeroing transform coefficients alone does not shrink a file much; the JPEG-style method does.")
        warnings.append(f"The compressed file ({mb(len(blob))}) is not smaller than your file "
                        f"({mb(info['file_bytes'])}).{why}")
    out["warnings"] = warnings
    out["upload"] = jsonable(info)
    return _done(out, t0)


@app.post("/api/compress/file")
def compress_file(file: UploadFile = File(...), params: str = Form("{}"), engine: str = Form("opencv")):
    """The compressed file itself (same bytes as described by /api/compress)."""
    c = _compress(_read_bytes(file), params, engine)
    if not c["downloadable"]:
        raise PipelineError("RESULT_TOO_LARGE",
                            f"The compressed file is {mb(len(c['fitted']['bytes']))}, over the "
                            f"{mb(MAX_DOWNLOAD_BYTES)} download limit. Choose a lower quality.")
    return _file_response(c["fitted"])
