"""Quality measures used by diagnostics (§9), guided verification (§10) and experiments (§13)."""
from __future__ import annotations

import cv2
import numpy as np

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
    Works on uint8 throughout (cv2.absdiff gives |a - b| without wrap-around), so no int16 or
    float copies of a large image are made.
    """
    med = cv2.medianBlur(gray, 3)
    outlier = cv2.absdiff(gray, med) > config.IMPULSE_MEDIAN_DIFF
    del med
    outlier &= (gray <= config.EXTREME_LOW) | (gray >= config.EXTREME_HIGH)
    return float(np.count_nonzero(outlier) / outlier.size)


# ---------------------------------------------------------- enlarged noise (screenshots)
def half_scale(gray: np.ndarray) -> np.ndarray:
    """Every 2nd pixel in both directions (NO averaging, which would hide the noise being measured).
    Noise that was enlarged 1.5–2x becomes pixel-sized again at this scale."""
    return np.ascontiguousarray(gray[::2, ::2])


def speck_density(gray: np.ndarray, diff: int = config.IMPULSE_MEDIAN_DIFF,
                  max_area: int = config.SPECK_MAX_AREA) -> float:
    """Share of pixels that differ from their 3x3 median by > diff AND form tiny isolated groups (≤ max_area px,
    8-connected). Unlike impulse_density it does not require pure black/white, so specks softened by resizing
    still count; the size test keeps edges and text strokes (long connected outliers) out."""
    out = (cv2.absdiff(gray, cv2.medianBlur(gray, 3)) > diff).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    return float(np.count_nonzero(small[labels]) / labels.size)


def coarse_noise(gray: np.ndarray) -> dict:
    """Noise σ and speck density measured on the half-scale copy (see config: enlarged noise)."""
    small = half_scale(gray)
    return {"noise_sigma_coarse": noise_sigma(small), "speck_density_coarse": speck_density(small)}


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
    """PSNR = 10 · log10(255² / MSE) in dB; +inf for identical images (MSE = 0)."""
    mse = float(np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


SSIM_WIN = 7


def _ssim_channel(x: np.ndarray, y: np.ndarray, data_range: float = 255.0) -> float:
    """Mean SSIM of one channel (Wang et al. 2004), with scikit-image's default settings.

    Local statistics over a 7x7 uniform window (cv2.boxFilter):
        μx, μy, σx² = N/(N−1)·(E[x²] − μx²), σy², σxy = N/(N−1)·(E[xy] − μxμy),  N = 49
        SSIM = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
        C1 = (0.01·L)², C2 = (0.03·L)², L = data range (255)
    The mean is taken after cropping 3 px on each side (where the window is not complete),
    exactly like skimage.metrics.structural_similarity, but much faster. The local maps are
    computed in float32 (agrees with skimage to ~1e-6); the final mean is float64.
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    k = (SSIM_WIN, SSIM_WIN)

    def box(a):
        return cv2.boxFilter(a, cv2.CV_32F, k, normalize=True, borderType=cv2.BORDER_REFLECT)

    ux, uy = box(x), box(y)
    n = SSIM_WIN * SSIM_WIN
    cov_norm = n / (n - 1.0)
    ux_uy = cv2.multiply(ux, uy)
    ux2, uy2 = cv2.multiply(ux, ux), cv2.multiply(uy, uy)
    vx = cv2.subtract(box(cv2.multiply(x, x)), ux2)
    vy = cv2.subtract(box(cv2.multiply(y, y)), uy2)
    vxy = cv2.subtract(box(cv2.multiply(x, y)), ux_uy)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    num = cv2.multiply(2 * ux_uy + c1, 2 * cov_norm * vxy + c2)
    den = cv2.multiply(ux2 + uy2 + c1, cov_norm * (vx + vy) + c2)
    s = cv2.divide(num, den)
    pad = (SSIM_WIN - 1) // 2
    return float(s[pad:-pad, pad:-pad].mean(dtype=np.float64))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean SSIM; for RGB the average of the per-channel values (as skimage with channel_axis=2)."""
    if reference.ndim == 3:
        return float(np.mean([_ssim_channel(reference[:, :, c], test[:, :, c])
                              for c in range(reference.shape[2])]))
    return _ssim_channel(reference, test)
