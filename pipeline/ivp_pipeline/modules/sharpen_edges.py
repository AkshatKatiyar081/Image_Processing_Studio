"""Module 3 — Sharpening & edges (spec §7).

Sharpening acts on luminance Y (no colour fringes); gradients act on gray, in float32.
Axis convention: x = columns. Gx responds to VERTICAL edges.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import metrics as Q
from ..core import filtering as F
from ..utils import apply_on_luma, is_color, to_gray

GRADIENT_KERNELS = {
    "sobel": (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
              np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)),
    "prewitt": (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32),
                np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)),
    # Roberts cross: the two +-45 degree diagonal differences (reported as gx / gy for simplicity)
    "roberts": (np.array([[-1, 0], [0, 1]], np.float32),
                np.array([[0, -1], [1, 0]], np.float32)),
}


def high_boost_kernel(A: float) -> np.ndarray:
    """g = A*f - Laplacian(f) with the 8-neighbour Laplacian -> centre A + 8, others -1, sum = A."""
    k = -np.ones((3, 3), np.float32)
    k[1, 1] = A + 8.0
    return k


def _corr(img: np.ndarray, k: np.ndarray, engine: str) -> np.ndarray:
    if engine == "opencv":
        return cv2.filter2D(img.astype(np.float32), cv2.CV_32F, k)
    return F.correlate2d(img, k)


def _normalise(x: np.ndarray) -> np.ndarray:
    """Scale to 0..255 by the maximum (never clip, so the strongest response is always visible)."""
    m = float(x.max())
    return np.zeros(x.shape, np.uint8) if m <= 0 else F.to_uint8(x * (255.0 / m))


def gradients(gray: np.ndarray, operator: str, engine: str) -> tuple[np.ndarray, np.ndarray]:
    kx, ky = GRADIENT_KERNELS[operator]
    return _corr(gray, kx, engine), _corr(gray, ky, engine)


def edge_map_from(mag: np.ndarray, t: float) -> np.ndarray:
    m = float(mag.max())
    return mag > t * m if m > 0 else np.zeros(mag.shape, bool)


def run(img: np.ndarray, p: dict, engine: str):
    op = p["operation"]
    extras, metrics, warnings = {}, {}, []
    gray_in = to_gray(img)

    if op == "laplacian_boost":
        A = float(p["A"])
        k = high_boost_kernel(A)
        if A == 0.0:
            # Pure high-pass (Exp 03 Task 3): edges on black, shown as |g| rescaled to 0..255.
            out = _normalise(np.abs(_corr(gray_in, k, engine)))
            if out.max() == 0:
                warnings.append("Laplacian response is zero (constant-value image); output is all-black.")
            if is_color(img):
                warnings.append(
                    "A = 0 produces a grayscale edge image. "
                    "If you keep editing this result it will be treated as a gray image."
                )
        else:
            out = apply_on_luma(img, lambda y: F.to_uint8(_corr(y, k, engine)))
            if A > 1.0:
                warnings.append(f"High-boost with A={A:g}: kernel sum is {A:g}, so brightness is "
                                f"also scaled by ~{A:g} (intended — brightens dark images).")
        metrics["kernel_centre"] = A + 8.0

    elif op == "unsharp":
        kgain, s = float(p["k"]), float(p["blur_sigma"])
        ks = F.gaussian_ksize(s)

        def fn(y):
            f = y.astype(np.float32)
            blur = (cv2.GaussianBlur(f, (ks, ks), s) if engine == "opencv"
                    else F.gaussian_filter(y, s, ks))
            return F.to_uint8(f + kgain * (f - blur))

        out = apply_on_luma(img, fn)

    elif op == "gradient":
        gx, gy = gradients(gray_in, p["operator"], engine)
        mag = np.hypot(gx, gy) if p["magnitude"] == "l2" else np.abs(gx) + np.abs(gy)
        comp = p["component"]
        view = mag if comp == "magnitude" else np.abs(gx if comp == "gx" else gy)
        out = _normalise(view)
        if out.max() == 0:
            warnings.append("Gradient magnitude is zero (constant-value image); output is all-black.")
        extras["gx"], extras["gy"] = gx, gy
        metrics["mean_magnitude"] = round(float(mag.mean()), 3)
        if p["edge_threshold"] > 0:
            edges = edge_map_from(mag, float(p["edge_threshold"]))
            extras["edge_map"] = (edges * 255).astype(np.uint8)
            metrics["edge_density"] = round(Q.edge_density(edges), 5)
            out = extras["edge_map"]
    else:  # pragma: no cover
        raise ValueError(op)

    if op != "gradient" and out.ndim == img.ndim:
        metrics["noise_sigma_before"] = round(Q.noise_sigma(gray_in), 2)
        metrics["noise_sigma_after"] = round(Q.noise_sigma(to_gray(out)), 2)
    return out, extras, metrics, warnings
