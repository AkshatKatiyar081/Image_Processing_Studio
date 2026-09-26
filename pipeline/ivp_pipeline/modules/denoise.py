"""Module 2 — Noise removal (spec §6). Filters run on each RGB channel independently."""
from __future__ import annotations

import cv2
import numpy as np

from .. import metrics as Q
from ..core import filtering as F
from ..utils import per_channel, to_gray


def _filter_fn(p: dict, engine: str):
    t, k = p["filter_type"], int(p["kernel_size"])
    if t == "median":
        return (lambda c: cv2.medianBlur(c, k)) if engine == "opencv" else (lambda c: F.median_filter(c, k))
    if t == "box":
        return (lambda c: cv2.blur(c, (k, k))) if engine == "opencv" else (lambda c: F.to_uint8(F.box_filter(c, k)))
    sigma = float(p["sigma"])
    ks = F.gaussian_ksize(sigma)
    if engine == "opencv":
        return lambda c: cv2.GaussianBlur(c, (ks, ks), sigma)
    return lambda c: F.to_uint8(F.gaussian_filter(c, sigma, ks))


def run(img: np.ndarray, p: dict, engine: str):
    out = per_channel(img, _filter_fn(p, engine))
    g0, g1 = to_gray(img), to_gray(out)
    metrics = {
        "noise_sigma_before": round(Q.noise_sigma(g0), 2),
        "noise_sigma_after": round(Q.noise_sigma(g1), 2),
        "impulse_density_before": round(Q.impulse_density(g0), 5),
        "impulse_density_after": round(Q.impulse_density(g1), 5),
    }
    if p["filter_type"] == "gaussian":
        metrics["kernel_size_used"] = F.gaussian_ksize(float(p["sigma"]))
    return out, {}, metrics, []
