"""Internal execution helpers shared by the public API, guided mode and auto mode."""
from __future__ import annotations

import time

import numpy as np

from . import registry
from .errors import PipelineError
from .modules import contrast, denoise, segment, sharpen_edges
from .utils import hist_list

RUNNERS = {
    "contrast": contrast.run,
    "denoise": denoise.run,
    "sharpen_edges": sharpen_edges.run,
    "segment": segment.run,
}
ENGINES = ("opencv", "numpy")


def check_engine(engine: str) -> None:
    if engine not in ENGINES:
        raise PipelineError("BAD_ENGINE", f"engine must be one of {ENGINES}, got {engine!r}")


def execute(img: np.ndarray, module: str, params: dict, engine: str) -> dict:
    """Run one module on an already-prepared image with already-validated params."""
    t0 = time.perf_counter()
    out, extras, metrics, warnings = RUNNERS[module](img, params, engine)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "image": out,
        "extras": extras,
        "histograms": {"before": hist_list(img), "after": hist_list(out)},
        "metrics": metrics,
        "warnings": warnings,
        "meta": {"module": module, "params": params, "engine": engine,
                 "time_ms": round(elapsed, 2)},
    }


def step(img: np.ndarray, module: str, params: dict, engine: str) -> dict:
    """Validate params then execute (used for internally generated stages)."""
    return execute(img, module, registry.validate(module, params), engine)
