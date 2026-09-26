"""From-scratch spatial filtering (NumPy engine).

Conventions (spec §4):
  * Kernels are CORRELATION masks, applied without flipping (same as cv2.filter2D).
  * Anchor = (kh // 2, kw // 2), which matches OpenCV's default anchor, also for
    even-sized kernels such as the 2x2 Roberts masks.
  * Linear filters use reflect-101 borders (OpenCV default, NumPy mode="reflect").
  * The median filter uses replicate borders (NumPy mode="edge"), because that is
    what cv2.medianBlur does.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Round to nearest (half-to-even, like OpenCV's saturate_cast) and clip to [0, 255]."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def correlate2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D correlation of a single-channel image with an arbitrary kernel.

    out[y, x] = sum_{i, j} kernel[i, j] * img[y + i - ay, x + j - ax]
    with anchor (ay, ax) = (kh // 2, kw // 2). Returns float32.
    """
    if img.ndim != 2:
        raise ValueError("correlate2d expects a single-channel (2-D) image")
    f = img.astype(np.float32, copy=False)
    k = np.asarray(kernel, dtype=np.float32)
    kh, kw = k.shape
    ay, ax = kh // 2, kw // 2
    padded = np.pad(f, ((ay, kh - 1 - ay), (ax, kw - 1 - ax)), mode="reflect")
    h, w = f.shape
    out = np.zeros((h, w), dtype=np.float32)
    # Loop over kernel taps (at most 49 for a 7x7 mask): each step is a
    # whole-image multiply-add, which keeps this fast in pure NumPy.
    for i in range(kh):
        for j in range(kw):
            c = k[i, j]
            if c != 0.0:
                out += c * padded[i:i + h, j:j + w]
    return out


def box_kernel(k: int) -> np.ndarray:
    """Normalised K x K averaging mask: (1 / K^2) * ones."""
    return np.full((k, k), 1.0 / (k * k), dtype=np.float32)


def gaussian_ksize(sigma: float) -> int:
    """Kernel size covering +-3 sigma: K = 2 * ceil(3 * sigma) + 1."""
    return int(2 * np.ceil(3.0 * sigma) + 1)


def gaussian_kernel_1d(sigma: float, ksize: int | None = None) -> np.ndarray:
    """Sampled, normalised 1-D Gaussian (same formula as cv2.getGaussianKernel for sigma > 0)."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    k = ksize or gaussian_ksize(sigma)
    x = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


def gaussian_filter(img: np.ndarray, sigma: float, ksize: int | None = None) -> np.ndarray:
    """Separable Gaussian low-pass: one 1-D pass along rows, then along columns. Returns float32."""
    g = gaussian_kernel_1d(sigma, ksize)
    tmp = correlate2d(img, g[np.newaxis, :])
    return correlate2d(tmp, g[:, np.newaxis])


def box_filter(img: np.ndarray, k: int) -> np.ndarray:
    """K x K arithmetic-mean filter. Returns float32."""
    return correlate2d(img, box_kernel(k))


def median_filter(img: np.ndarray, k: int, strip_rows: int = 128) -> np.ndarray:
    """K x K median (order-statistic) filter on a uint8 single-channel image.

    Uses np.partition (O(K^2) per pixel) and processes the image in horizontal
    strips so memory stays bounded (H * W * K^2 bytes would be ~50 MB for
    1024 x 1024 with K = 7).
    """
    if img.ndim != 2:
        raise ValueError("median_filter expects a single-channel (2-D) image")
    if k % 2 == 0 or k < 1:
        raise ValueError("kernel size must be odd")
    r = k // 2
    padded = np.pad(img, r, mode="edge")
    h, w = img.shape
    out = np.empty((h, w), dtype=img.dtype)
    mid = (k * k) // 2
    for y0 in range(0, h, strip_rows):
        y1 = min(h, y0 + strip_rows)
        win = sliding_window_view(padded[y0:y1 + 2 * r], (k, k))  # (rows, w, k, k)
        flat = win.reshape(win.shape[0], win.shape[1], k * k)
        out[y0:y1] = np.partition(flat, mid, axis=-1)[..., mid]
    return out
