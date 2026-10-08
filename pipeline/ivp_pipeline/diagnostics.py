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
    coarse = Q.coarse_noise(g)
    return {
        "mean": round(float(g.mean()) / 255.0, 4),
        "clipped_low": round(float((g <= C.EXTREME_LOW).mean()), 4),
        "clipped_high": round(float((g >= C.EXTREME_HIGH).mean()), 4),
        "std": round(float(g.std()), 2),
        "range_p1_p99": round(float(p99 - p1), 1),
        "noise_sigma": round(Q.noise_sigma(g), 2),
        "impulse_density": round(Q.impulse_density(g), 5),
        "noise_sigma_coarse": round(coarse["noise_sigma_coarse"], 2),
        "speck_density_coarse": round(coarse["speck_density_coarse"], 5),
        "illumination_u": round(u_bg, 4),
        "eta": round(otsu(g)["eta"], 4),
        "sharpness_lap_var": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
    }


def _finding(code, severity, value, threshold, message, scale=None):
    """scale: for noise findings, "pixel" or "enlarged" (seen at half scale, e.g. in a screenshot)."""
    out = {"code": code, "severity": severity, "value": value, "threshold": threshold, "message": message}
    if scale:
        out["scale"] = scale
    return out


def impulse_noise(m: dict) -> dict | None:
    """Salt & pepper, if present: {"scale": "pixel" | "enlarged", "value", "threshold", "median_ksize"}.

    pixel:    single-pixel 0/255 specks (impulse_density, 3x3 median test).
    enlarged: specks smeared over 2–3 px by resizing (typical of screenshots), seen at half scale; a 3x3
              median cannot remove them, so the fix uses 5x5 (7x7 when dense).
    """
    if m["impulse_density"] > C.IMPULSE_DENSITY:
        k = 5 if m["impulse_density"] > C.IMPULSE_HEAVY else 3
        return {"scale": "pixel", "value": m["impulse_density"], "threshold": C.IMPULSE_DENSITY, "median_ksize": k}
    if m["speck_density_coarse"] > C.SPECK_COARSE_DENSITY and m["noise_sigma"] < C.SPECK_COARSE_MAX_SIGMA:
        heavy = m["speck_density_coarse"] > C.SPECK_COARSE_HEAVY
        return {"scale": "enlarged", "value": m["speck_density_coarse"], "threshold": C.SPECK_COARSE_DENSITY,
                "median_ksize": C.COARSE_MEDIAN_KSIZE_HEAVY if heavy else C.COARSE_MEDIAN_KSIZE}
    return None


def gaussian_noise(m: dict) -> dict | None:
    """Gaussian (sensor) grain, if present: {"scale", "value" (σ), "threshold", "strong", "sigma" (filter σ)}.

    pixel:    Immerkær σ above NOISE_MODERATE; σ 1.0 smoothing above NOISE_STRONG, 0.6 between.
    enlarged: grain smeared over 2–3 px (screenshots): the half-scale σ is high AND much higher than the
              pixel-scale σ (NOISE_COARSE_RATIO); smoothing needs σ 1.5.
    """
    s = m["noise_sigma"]
    if s > C.NOISE_MODERATE:
        strong = s > C.NOISE_STRONG
        return {"scale": "pixel", "value": s, "threshold": C.NOISE_MODERATE, "strong": strong,
                "sigma": 1.0 if strong else 0.6}
    sc = m["noise_sigma_coarse"]
    if sc > C.NOISE_COARSE_MIN and sc > C.NOISE_COARSE_RATIO * s:
        return {"scale": "enlarged", "value": sc, "threshold": C.NOISE_COARSE_MIN, "strong": sc > C.NOISE_STRONG,
                "sigma": C.COARSE_GAUSS_SIGMA}
    return None


def findings(m: dict) -> list[dict]:
    out = []
    imp = impulse_noise(m)
    if imp:
        k = imp["median_ksize"]
        if imp["scale"] == "pixel":
            msg = (f"Salt-and-pepper noise on {m['impulse_density']:.1%} of pixels. Thresholding will be "
                   "riddled with pin-holes and gradients will fire on every spike; a median filter removes it "
                   "(Unit 2 order-statistic filters, Exp 03).")
        else:
            msg = (f"Enlarged salt-and-pepper specks ({imp['value']:.1%} at half scale). The image was probably "
                   "resized (e.g. a screenshot), which smeared each speck over 2–3 pixels; a "
                   f"{k}×{k} median filter removes them (a 3×3 one is too small).")
        f = _finding("IMPULSE_NOISE", "warning", imp["value"], imp["threshold"], msg, imp["scale"])
        f["median_ksize"] = k                    # the fix the dashboard offers (3, 5 or 7)
        out.append(f)
    gn = gaussian_noise(m)
    if gn and gn["scale"] == "pixel":
        out.append(_finding(
            "GAUSSIAN_NOISE", "warning" if gn["strong"] else "info", gn["value"], gn["threshold"],
            f"Sensor-type noise with estimated σ ≈ {m['noise_sigma']:.1f} gray levels (Immerkær). "
            "Sharpening and derivative operators amplify it; smooth first (Unit 2, Exp 03).", "pixel"))
    elif gn:
        out.append(_finding(
            "GAUSSIAN_NOISE", "warning", gn["value"], gn["threshold"],
            f"Coarse grain: noise σ ≈ {gn['value']:.1f} at half scale but only {m['noise_sigma']:.1f} per pixel, "
            "so the noise was enlarged (e.g. a screenshot of a noisy image). Smooth with a Gaussian of "
            f"σ ≈ {gn['sigma']:g} (Unit 2, Exp 03).", "enlarged"))
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
