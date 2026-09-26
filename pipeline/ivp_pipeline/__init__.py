"""IVP Enhancement & Diagnostic Studio — image-processing pipeline.

Public API (spec §12). All functions take a uint8 RGB (H x W x 3) or grayscale (H x W)
NumPy array and return plain dicts of NumPy arrays / numbers / lists. The caller is
responsible for decoding uploads (convert OpenCV's BGR to RGB!) and for serialising
the returned arrays (PNG for masks and RGBA cutouts, JPEG or PNG for photos).

    from ivp_pipeline import list_modules, diagnose, run_module, run_guided, auto_enhance
"""
from __future__ import annotations

import numpy as np

from . import metrics, registry
from .auto import auto_enhance_prepared
from .diagnostics import diagnose_prepared
from .errors import PipelineError
from .guided import INTENTS, run_guided_prepared
from .pipeline import check_engine, execute
from .utils import prepare, to_gray

__all__ = ["list_modules", "diagnose", "run_module", "run_guided", "auto_enhance",
           "PipelineError", "INTENTS"]
__version__ = "1.0.0"


def _meta(scale: float, original_shape, img: np.ndarray) -> dict:
    return {"scale": scale, "original_shape": list(original_shape[:2]), "shape": list(img.shape[:2])}


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
    work, scale = prepare(img)
    out = diagnose_prepared(work)
    out["meta"] = _meta(scale, img.shape, work)
    return out


def run_module(img: np.ndarray, module: str, params: dict | None = None,
               engine: str = "opencv") -> dict:
    """Run one module with explicit parameters (Experimenter view).

    module ∈ {"contrast", "denoise", "sharpen_edges", "segment"}; engine ∈ {"opencv", "numpy"}.
    Returns {"input", "image", "extras", "histograms", "metrics", "warnings", "meta"}.
    "input" is the (possibly downscaled) image the result corresponds to — use it as the
    "before" side of a comparison slider so both sides have the same size.
    """
    check_engine(engine)
    work, scale = prepare(img)
    p = registry.validate(module, _scale_seed(module, params, scale))
    res = execute(work, module, p, engine)
    res["input"] = work
    res["meta"].update(_meta(scale, img.shape, work))
    return res


def run_guided(img: np.ndarray, intent: str, params: dict | None = None,
               engine: str = "opencv") -> dict:
    """Direct (Branch A) vs prerequisite-conditioned (Branch B) comparison (spec §10).

    intent ∈ {"segment", "edges", "sharpen", "equalize"}; params are the target module's params.
    """
    check_engine(engine)
    work, scale = prepare(img)
    module = "segment" if intent == "segment" else None
    res = run_guided_prepared(work, intent, _scale_seed(module, params, scale), engine)
    res["input"] = work
    res["meta"] = _meta(scale, img.shape, work)
    return res


def auto_enhance(img: np.ndarray, engine: str = "opencv") -> dict:
    """One-click enhancement chosen from the image's diagnosis (spec §11)."""
    check_engine(engine)
    # Measure impulse density BEFORE prepare() downscales the image.
    # Both INTER_AREA (backend) and the browser's bilinear canvas drawImage
    # average isolated spikes into their neighbours, making impulse_density
    # nearly zero on the working image even when real salt-and-pepper noise
    # is present.  The max() inside auto_enhance_prepared makes this safe when
    # scale == 1.0 (no backend downscaling) — both measurements are identical.
    pre_id = metrics.impulse_density(to_gray(img))
    work, scale = prepare(img)
    res = auto_enhance_prepared(work, engine, pre_scale_impulse_density=pre_id)
    res["input"] = work
    res["meta"] = _meta(scale, img.shape, work)
    return res
