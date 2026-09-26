"""Histogram processing from scratch: equalization (Exp 04) and Otsu's method (Unit 6)."""
from __future__ import annotations

import numpy as np

L = 256


def histogram(gray: np.ndarray) -> np.ndarray:
    """256-bin histogram n_k of a uint8 image."""
    return np.bincount(gray.ravel(), minlength=L)[:L]


def equalize_lut(gray: np.ndarray) -> np.ndarray:
    """Textbook HE transfer function s_k = round((L - 1) * sum_{j<=k} p_r(r_j)) as a 256-entry LUT."""
    p = histogram(gray) / gray.size
    cdf = np.cumsum(p)
    return np.clip(np.rint((L - 1) * cdf), 0, 255).astype(np.uint8)


def equalize(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply histogram equalization. Returns (output, lut)."""
    lut = equalize_lut(gray)
    return lut[gray], lut


def otsu(gray: np.ndarray) -> dict:
    """Otsu's optimum global threshold (Gonzalez & Woods formulation).

    omega(k) = sum_{i<=k} p_i                (class-0 probability)
    mu(k)    = sum_{i<=k} i * p_i            (cumulative first moment -- NOT a class mean)
    mu_T     = mu(L - 1)                     (global mean)
    sigma_B^2(k) = (mu_T * omega(k) - mu(k))^2 / (omega(k) * (1 - omega(k)))
    k* = argmax sigma_B^2 (ties averaged)
    eta = sigma_B^2(k*) / sigma_G^2          (separability, in [0, 1])

    Pixels with value > k* belong to class 1 (same as cv2.THRESH_BINARY).
    """
    p = histogram(gray).astype(np.float64) / gray.size
    i = np.arange(L, dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(i * p)
    mu_t = mu[-1]
    sigma_g2 = float(np.sum((i - mu_t) ** 2 * p))

    denom = omega * (1.0 - omega)
    sigma_b2 = np.zeros(L, dtype=np.float64)
    valid = denom > 1e-12
    sigma_b2[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]

    if sigma_g2 <= 0 or not valid.any():  # constant image: nothing to separate
        k = int(gray.flat[0])
        return {"threshold": k, "eta": 0.0, "sigma_b2": sigma_b2}

    best = sigma_b2.max()
    ties = np.flatnonzero(sigma_b2 >= best * (1.0 - 1e-12))
    k = int(np.floor(ties.mean()))
    return {"threshold": k, "eta": float(best / sigma_g2), "sigma_b2": sigma_b2}
