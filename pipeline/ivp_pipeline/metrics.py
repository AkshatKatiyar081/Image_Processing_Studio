"""Quality measures used by diagnostics (§9), guided verification (§10) and experiments (§13)."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import config
from .core.background import estimate_background

IMMERKAER_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


# --------------------------------------------------------------------- noise
def noise_sigma(gray: np.ndarray, edge_percentile: float = config.NOISE_EDGE_PERCENTILE) -> float:
    """Immerkær's fast noise estimate, edge-excluded.

    sigma_n = sqrt(pi / 2) * mean(|N * I|) / 6, averaged over pixels whose Sobel magnitude is
    below the given percentile (so texture and edges are not mistaken for noise).
    The mask N responds to noise but is zero for any locally planar (linear ramp) signal.
    """
    f = gray.astype(np.float32)
    resp = np.abs(cv2.filter2D(f, cv2.CV_32F, IMMERKAER_MASK))
    mag = np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))
    keep = mag <= np.percentile(mag, edge_percentile)
    keep[[0, -1], :] = False
    keep[:, [0, -1]] = False
    if not keep.any():
        return 0.0
    return float(np.sqrt(np.pi / 2) * resp[keep].mean() / 6.0)


def impulse_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are extreme (<=5 or >=250) AND differ from their 3x3 median by > 50.

    The second condition stops legitimately black/white regions (shadows, sky) from counting.
    """
    med = cv2.medianBlur(gray, 3)
    extreme = (gray <= config.EXTREME_LOW) | (gray >= config.EXTREME_HIGH)
    outlier = np.abs(gray.astype(np.int16) - med.astype(np.int16)) > config.IMPULSE_MEDIAN_DIFF
    return float((extreme & outlier).mean())


# --------------------------------------------------------------- illumination
def illumination_unevenness(gray: np.ndarray) -> tuple[float, np.ndarray]:
    """U_bg = (P95(B) - P5(B)) / 255, B = morphological background with R = max(H, W) / 10."""
    radius = max(3, max(gray.shape) // config.BG_RADIUS_FRACTION)
    bg, _ = estimate_background(gray, radius, "auto")
    return float((np.percentile(bg, 95) - np.percentile(bg, 5)) / 255.0), bg


# --------------------------------------------------------------------- edges
def flat_region_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels whose gradient on a sigma=2 smoothed copy is below the 50th percentile."""
    s = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), config.FLAT_REGION_SIGMA)
    mag = np.hypot(cv2.Sobel(s, cv2.CV_32F, 1, 0), cv2.Sobel(s, cv2.CV_32F, 0, 1))
    return mag < np.percentile(mag, config.FLAT_REGION_PERCENTILE)


def edge_density(edge_map: np.ndarray, region: np.ndarray | None = None) -> float:
    e = edge_map.astype(bool)
    return float(e[region].mean()) if region is not None else float(e.mean())


def edge_f1(pred: np.ndarray, truth: np.ndarray, tolerance: int = 1) -> float:
    """F1 between binary edge maps, counting a match within `tolerance` pixels (BSDS-style).

    precision = share of predicted edge pixels within tolerance of a true edge,
    recall    = share of true edge pixels within tolerance of a predicted edge.
    tolerance=0 gives the pixel-exact score (harsh: a 1-px shift counts as a miss).
    """
    p, t = pred.astype(bool), truth.astype(bool)
    if not p.any() or not t.any():
        return 0.0
    if tolerance > 0:
        k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
        t_near = cv2.dilate(t.astype(np.uint8), k) > 0
        p_near = cv2.dilate(p.astype(np.uint8), k) > 0
    else:
        t_near, p_near = t, p
    precision = (p & t_near).sum() / p.sum()
    recall = (t & p_near).sum() / t.sum()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# -------------------------------------------------------------- segmentation
def n_components(mask: np.ndarray) -> int:
    """Number of 8-connected foreground components."""
    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)


def fragmentation(mask: np.ndarray) -> int:
    """Islands + holes: foreground components plus background components (8-connected).

    Works for Otsu and region growing alike (a grown region is 1 component, but noise punches holes).
    """
    m = mask.astype(bool)
    return n_components(m) + n_components(~m)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else 1.0


# ----------------------------------------------------------- reference-based
def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(reference, test, data_range=255))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    kw = {"channel_axis": 2} if reference.ndim == 3 else {}
    return float(structural_similarity(reference, test, data_range=255, **kw))
