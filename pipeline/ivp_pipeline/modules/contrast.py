"""Module 1 — Contrast, exposure & illumination (spec §5). All transforms act on luminance Y only."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import config
from ..core import histogram as Hs
from ..core.background import estimate_background
from ..utils import apply_on_luma, luma


def _lut_from_pairs(src: np.ndarray, dst: np.ndarray) -> list[int]:
    """Recover a (monotonic) transfer curve from an input/output pair — for plotting only."""
    lut = np.zeros(256, np.int32)
    lut[src.ravel()] = dst.ravel()
    return np.maximum.accumulate(lut).tolist()


def auto_gamma(y: np.ndarray) -> float:
    """gamma such that mean^gamma = 0.5 (mean normalised to [0, 1]), clamped to [0.3, 3]."""
    m = min(max(float(y.mean()) / 255.0, 0.01), 0.99)
    g = math.log(0.5) / math.log(m)
    return float(min(max(g, config.GAMMA_MIN), config.GAMMA_MAX))


def gamma_lut(gamma: float) -> np.ndarray:
    r = np.arange(256, dtype=np.float64) / 255.0
    return np.clip(np.rint(255.0 * r ** gamma), 0, 255).astype(np.uint8)


def stretch_lut(y: np.ndarray, clip_percentile: float) -> np.ndarray | None:
    lo = float(np.percentile(y, clip_percentile))
    hi = float(np.percentile(y, 100.0 - clip_percentile))
    if hi - lo < 1.0:
        return None
    r = np.arange(256, dtype=np.float64)
    s = (r - lo) / (hi - lo) * 255.0
    return np.clip(np.rint(s), 0, 255).astype(np.uint8)


def _apply_lut(ch: np.ndarray, lut: np.ndarray, engine: str) -> np.ndarray:
    return cv2.LUT(ch, lut) if engine == "opencv" else lut[ch]


def run(img: np.ndarray, p: dict, engine: str):
    mode = p["mode"]
    y = luma(img)
    extras, metrics, warnings = {}, {}, []

    if mode == "hist_eq":
        if engine == "numpy":
            lut = Hs.equalize_lut(y)
            out = apply_on_luma(img, lambda ch: lut[ch])
            extras["lut"] = lut.astype(int).tolist()
        else:
            out = apply_on_luma(img, cv2.equalizeHist)
            extras["lut"] = _lut_from_pairs(y, cv2.equalizeHist(y))

    elif mode == "gamma":
        g = auto_gamma(y) if p["gamma"] == "auto" else float(p["gamma"])
        lut = gamma_lut(g)
        out = apply_on_luma(img, lambda ch: _apply_lut(ch, lut, engine))
        extras["lut"] = lut.astype(int).tolist()
        metrics["gamma_used"] = round(g, 3)

    elif mode == "stretch":
        lut = stretch_lut(y, p["clip_percentile"])
        if lut is None:
            out = img.copy()
            warnings.append("Image is (nearly) constant; contrast stretching was skipped.")
        else:
            out = apply_on_luma(img, lambda ch: _apply_lut(ch, lut, engine))
            extras["lut"] = lut.astype(int).tolist()

    elif mode == "flatten":
        bg, polarity = estimate_background(y, p["se_radius"], p["object_polarity"], engine)
        mean_bg = float(bg.mean())

        def fn(ch):
            f = ch.astype(np.float32) - bg.astype(np.float32) + mean_bg
            return np.clip(np.rint(f), 0, 255).astype(np.uint8)

        out = apply_on_luma(img, fn)
        extras["background"] = bg
        metrics["polarity_used"] = polarity
    else:  # pragma: no cover - guarded by registry
        raise ValueError(mode)

    metrics["mean_before"] = round(float(y.mean()), 2)
    metrics["mean_after"] = round(float(luma(out).mean()), 2)
    metrics["std_before"] = round(float(y.std()), 2)
    metrics["std_after"] = round(float(luma(out).std()), 2)
    return out, extras, metrics, warnings
