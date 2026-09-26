"""Background estimation by morphology (spec §5, 1d) — shared by Module 1 flattening and diagnostics."""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from . import morphology as M
from .histogram import otsu


def remove_plane(gray: np.ndarray) -> np.ndarray:
    """Subtract a least-squares plane a*x + b*y + c (fitted on a 64x64 thumbnail) and rescale to uint8.

    A linear lighting gradient is exactly a plane, so this removes it well enough to decide
    polarity, while compact objects barely influence the fit.
    """
    h, w = gray.shape
    n = 64
    thumb = cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA).astype(np.float64)
    yy, xx = np.mgrid[0:n, 0:n]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(n * n)]
    (a, b, c), *_ = np.linalg.lstsq(A, thumb.ravel(), rcond=None)
    Y, X = np.mgrid[0:h, 0:w]
    r = gray - (a * X * n / w + b * Y * n / h + c)
    span = r.max() - r.min()
    if span <= 0:
        return np.zeros_like(gray)
    return np.clip(np.rint((r - r.min()) * 255.0 / span), 0, 255).astype(np.uint8)


def auto_polarity(gray: np.ndarray) -> str:
    """Objects = the SMALLER Otsu class of the plane-detrended image.

    Detrending first matters: on an image with a strong ramp, raw Otsu splits the frame into
    a bright half and a dark half and the "smaller class" can be the wrong one
    (2 of 10 synthetic scenes failed at ±70 without it; 100/100 correct with it).
    """
    r = remove_plane(gray)
    k = otsu(r)["threshold"]
    return "bright" if (r > k).mean() < 0.5 else "dark"


def estimate_background(gray: np.ndarray, radius: int, polarity: str = "auto",
                        engine: str = "opencv") -> tuple[np.ndarray, str]:
    """Remove objects smaller than a disk of `radius` to leave the illumination background B.

    bright objects on dark background -> B = opening(f)   (white top-hat = f - B)
    dark objects on bright background -> B = closing(f)   (black top-hat = B - f)
    For large radii the estimate is computed at 1/4 scale (radius / 4) and upsampled,
    which is ~16x cheaper and fine because B is smooth by construction.
    Returns (B as uint8, polarity used).
    """
    if polarity == "auto":
        polarity = auto_polarity(gray)
    h, w = gray.shape
    small = radius > config.FLATTEN_DOWNSCALE_ABOVE_R
    src = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA) if small else gray
    r = max(1, round(radius / 4)) if small else int(radius)

    se = M.disk(r)
    if engine == "numpy":
        bg = M.opening(src, se) if polarity == "bright" else M.closing(src, se)
    else:
        op = cv2.MORPH_OPEN if polarity == "bright" else cv2.MORPH_CLOSE
        bg = cv2.morphologyEx(src, op, se)
    if small:
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg, polarity
