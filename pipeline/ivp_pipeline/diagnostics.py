"""Phase 2 — image quality profile (spec §9)."""
from __future__ import annotations

import cv2
import numpy as np

from . import config as C
from . import metrics as Q
from .core.histogram import otsu
from .utils import to_gray


def measure(img: np.ndarray) -> dict:
    """All §9 metrics for a prepared image (uint8, gray or RGB)."""
    g = to_gray(img)
    p1, p99 = np.percentile(g, [1, 99])
    u_bg, _ = Q.illumination_unevenness(g)
    return {
        "mean": round(float(g.mean()) / 255.0, 4),
        "clipped_low": round(float((g <= C.EXTREME_LOW).mean()), 4),
        "clipped_high": round(float((g >= C.EXTREME_HIGH).mean()), 4),
        "std": round(float(g.std()), 2),
        "range_p1_p99": round(float(p99 - p1), 1),
        "noise_sigma": round(Q.noise_sigma(g), 2),
        "impulse_density": round(Q.impulse_density(g), 5),
        "illumination_u": round(u_bg, 4),
        "eta": round(otsu(g)["eta"], 4),
        "sharpness_lap_var": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
    }


def _finding(code, severity, value, threshold, message):
    return {"code": code, "severity": severity, "value": value, "threshold": threshold,
            "message": message}


def findings(m: dict) -> list[dict]:
    out = []
    if m["impulse_density"] > C.IMPULSE_DENSITY:
        out.append(_finding(
            "IMPULSE_NOISE", "warning", m["impulse_density"], C.IMPULSE_DENSITY,
            f"Salt-and-pepper noise on {m['impulse_density']:.1%} of pixels. Thresholding will be "
            "riddled with pin-holes and gradients will fire on every spike; a median filter removes it "
            "(Unit 2 order-statistic filters, Exp 03)."))
    if m["noise_sigma"] > C.NOISE_MODERATE:
        sev = "warning" if m["noise_sigma"] > C.NOISE_STRONG else "info"
        out.append(_finding(
            "GAUSSIAN_NOISE", sev, m["noise_sigma"], C.NOISE_MODERATE,
            f"Sensor-type noise with estimated σ ≈ {m['noise_sigma']:.1f} gray levels (Immerkær). "
            "Sharpening and derivative operators amplify it; smooth first (Unit 2, Exp 03)."))
    if m["illumination_u"] > C.ILLUMINATION_U:
        # Severity is only "info": large scene regions (sky vs ground) look like uneven lighting
        # to this measure. Guided mode settles it by checking whether flattening raises η.
        out.append(_finding(
            "UNEVEN_ILLUMINATION", "info", m["illumination_u"], C.ILLUMINATION_U,
            f"Background brightness varies by {m['illumination_u']:.0%} across the image. If this is "
            "uneven lighting, a single global Otsu threshold will split the scene by lighting, not by "
            "object; flattening the background fixes it (Unit 6 thresholding; Unit 4 morphology, "
            "Exp 07). Large scene regions can also cause this — guided mode checks whether flattening helps."))
    if m["eta"] < C.LOW_ETA:
        out.append(_finding(
            "LOW_SEPARABILITY", "info", m["eta"], C.LOW_ETA,
            f"Otsu separability η = {m['eta']:.2f}: the histogram is not clearly bimodal, so a "
            "global threshold may be unreliable (Unit 6)."))
    if m["mean"] < C.DARK_MEAN:
        out.append(_finding("UNDEREXPOSED", "info", m["mean"], C.DARK_MEAN,
                            "Image is dark overall; gamma < 1 brightens shadows (Unit 2 point processing)."))
    elif m["mean"] > C.BRIGHT_MEAN:
        out.append(_finding("OVEREXPOSED", "info", m["mean"], C.BRIGHT_MEAN,
                            "Image is bright overall; gamma > 1 recovers mid-tones (Unit 2 point processing)."))
    if max(m["clipped_low"], m["clipped_high"]) > C.CLIPPED_FRACTION:
        out.append(_finding("CLIPPING", "info", max(m["clipped_low"], m["clipped_high"]), C.CLIPPED_FRACTION,
                            "Many pixels are at pure black/white; detail there cannot be recovered."))
    if m["range_p1_p99"] < C.LOW_RANGE:
        out.append(_finding(
            "LOW_CONTRAST", "info", m["range_p1_p99"], C.LOW_RANGE,
            "Intensities use a narrow range; contrast stretching or HE will improve visibility "
            "(Unit 2). Note: this does not change an Otsu segmentation."))
    return out


def diagnose_prepared(img: np.ndarray) -> dict:
    m = measure(img)
    return {"metrics": m, "findings": findings(m)}
