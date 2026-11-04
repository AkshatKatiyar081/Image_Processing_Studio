"""Image Processing Studio — image-processing pipeline.

Public API (spec §12). All functions take a uint8 RGB (H x W x 3) or grayscale (H x W)
NumPy array and return plain dicts of NumPy arrays / numbers / lists. The caller is
responsible for decoding uploads (convert OpenCV's BGR to RGB!) and for serialising
the returned arrays (PNG for masks and RGBA cutouts, JPEG or PNG for photos).

    from ivp_pipeline import list_modules, diagnose, run_module, run_guided, auto_enhance
"""
from __future__ import annotations

import numpy as np

from . import registry
from .auto import auto_enhance_prepared
from .compare_compression import compare_compression_prepared
from .diagnostics import diagnose_prepared
from .errors import PipelineError
from .guided import INTENTS, run_guided_prepared
from .pipeline import check_engine, execute
from .sweep import degradation_sweep_prepared
from .utils import prepare_with_info

__all__ = ["list_modules", "diagnose", "run_module", "run_guided", "auto_enhance",
           "compare_compression", "degradation_sweep", "PipelineError", "INTENTS"]
__version__ = "1.0.0"


def _meta(scale: float, original_shape, img: np.ndarray, info: dict) -> dict:
    """Shape/scale bookkeeping plus how prepare() resized the image (see utils.prepare_with_info)."""
    return {"scale": scale, "original_shape": list(original_shape[:2]), "shape": list(img.shape[:2]),
            "resize_interp": info["resize_interp"], "impulse_density_full_res": info["impulse_density_full_res"]}


def _scale_seed(module: str, params: dict | None, scale: float) -> dict | None:
    """Seeds arrive in ORIGINAL image pixels; map them onto the (possibly downscaled) working image."""
    if module != "segment" or not params or params.get("seed") is None or scale == 1.0:
        return params
    p = dict(params)
    x, y = p["seed"]
    p["seed"] = [float(x) * scale, float(y) * scale]
    return p


def list_modules() -> dict:
    """Parameter schema of every module (name, type, range/options, default, description)."""
    return registry.list_modules()


def diagnose(img: np.ndarray) -> dict:
    """Quality profile: {"metrics", "findings", "meta"} (spec §9)."""
    work, scale, info = prepare_with_info(img)
    out = diagnose_prepared(work)
    out["meta"] = _meta(scale, img.shape, work, info)
    return out


def run_module(img: np.ndarray, module: str, params: dict | None = None,
               engine: str = "opencv") -> dict:
    """Run one module with explicit parameters (Experimenter view).

    module ∈ {"contrast", "denoise", "sharpen_edges", "segment", "compress"}; engine ∈ {"opencv", "numpy"}.
    Returns {"input", "image", "extras", "histograms", "metrics", "warnings", "meta"}.
    "input" is the (possibly downscaled) image the result corresponds to — use it as the
    "before" side of a comparison slider so both sides have the same size.
    """
    check_engine(engine)
    work, scale, info = prepare_with_info(img)
    p = registry.validate(module, _scale_seed(module, params, scale))
    res = execute(work, module, p, engine)
    res["input"] = work
    res["meta"].update(_meta(scale, img.shape, work, info))
    return res


def run_guided(img: np.ndarray, intent: str, params: dict | None = None,
               engine: str = "opencv") -> dict:
    """Direct (Branch A) vs prerequisite-conditioned (Branch B) comparison (spec §10).

    intent ∈ {"segment", "edges", "sharpen", "equalize"}; params are the target module's params.
    """
    check_engine(engine)
    work, scale, info = prepare_with_info(img)
    module = "segment" if intent == "segment" else None
    res = run_guided_prepared(work, intent, _scale_seed(module, params, scale), engine)
    res["input"] = work
    res["meta"] = _meta(scale, img.shape, work, info)
    return res


def auto_enhance(img: np.ndarray, engine: str = "opencv") -> dict:
    """One-click enhancement chosen from the image's diagnosis (spec §11)."""
    check_engine(engine)
    work, scale, info = prepare_with_info(img)
    res = auto_enhance_prepared(work, engine)
    res["input"] = work
    res["meta"] = _meta(scale, img.shape, work, info)
    return res


def compare_compression(img: np.ndarray, keep_percent: float = 10.0, engine: str = "opencv",
                        jpeg_quality: int | None = None) -> dict:
    """DCT vs Walsh–Hadamard at the same % of coefficients kept, plus JPEG-style coding
    (Compare tab → Compression). jpeg_quality=None matches the DCT result's PSNR; 1–100 sets it.

    Returns {"dct", "hadamard", "jpeg" (each a run_module-style result), "rd_curve", "energy_curve",
    "size_curve", "curve_scale", "summary", "keep_percent", "input", "meta"}.
    """
    check_engine(engine)
    if jpeg_quality is not None and not (isinstance(jpeg_quality, (int, float)) and 1 <= jpeg_quality <= 100):
        raise PipelineError("BAD_PARAM", f"jpeg_quality must be 1–100 or None (auto), got {jpeg_quality!r}")
    work, scale, info = prepare_with_info(img)
    res = compare_compression_prepared(work, keep_percent, engine,
                                       None if jpeg_quality is None else int(jpeg_quality))
    res["input"] = work
    res["meta"] = _meta(scale, img.shape, work, info)
    return res


def degradation_sweep(img: np.ndarray, intent: str, params: dict | None = None,
                      engine: str = "opencv") -> dict:
    """Accuracy vs added degradation, Direct vs Guided (intent ∈ {"segment", "edges"}).

    Returns {"axis", "metric", "points": [{level, x, direct, guided, gate_fired}],
    "user_point_index", "reference_note", "sweep_scale", "meta"}. Numbers only, no images.
    """
    check_engine(engine)
    work, scale, info = prepare_with_info(img)
    module = "segment" if intent == "segment" else None
    res = degradation_sweep_prepared(work, intent, _scale_seed(module, params, scale), engine)
    res["meta"] = _meta(scale, img.shape, work, info)
    return res
