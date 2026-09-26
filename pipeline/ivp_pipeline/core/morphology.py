"""Morphological operations from scratch (Unit 4 / Exp 07).

Works on uint8 images: grayscale morphology in general, and binary masks stored as 0/255.
Border handling matches OpenCV's default: pixels outside the image never win
(erosion pads with 255, dilation pads with 0).
"""
from __future__ import annotations

import numpy as np

from .region import bfs_region


def disk(radius: int) -> np.ndarray:
    """Disk structuring element of size (2r+1) x (2r+1).

    Reproduces the rasterisation used by cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ...),
    so both engines use exactly the same SE: for each row offset dy, the half-width is
    dx = round(r * sqrt((r^2 - dy^2) / r^2)).
    """
    r = int(radius)
    size = 2 * r + 1
    se = np.zeros((size, size), dtype=np.uint8)
    if r == 0:
        se[0, 0] = 1
        return se
    inv_r2 = 1.0 / (r * r)
    for i in range(size):
        dy = i - r
        dx = int(np.rint(r * np.sqrt((r * r - dy * dy) * inv_r2)))
        j1, j2 = max(r - dx, 0), min(r + dx + 1, size)
        se[i, j1:j2] = 1
    return se


def _rank(img: np.ndarray, se: np.ndarray, op) -> np.ndarray:
    kh, kw = se.shape
    ay, ax = kh // 2, kw // 2
    pad_value = 255 if op is np.minimum else 0
    padded = np.pad(img, ((ay, kh - 1 - ay), (ax, kw - 1 - ax)), mode="constant",
                    constant_values=pad_value)
    h, w = img.shape
    out = np.full((h, w), pad_value, dtype=img.dtype)
    for i, j in zip(*np.nonzero(se)):
        op(out, padded[i:i + h, j:j + w], out=out)
    return out


def erode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """(A ⊖ B)(x) = min over b in B of A(x + b)."""
    return _rank(img, se, np.minimum)


def dilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """(A ⊕ B)(x) = max over b in B of A(x + b)  (B symmetric, so no reflection needed)."""
    return _rank(img, se, np.maximum)


def opening(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """A ∘ B = (A ⊖ B) ⊕ B — removes bright details smaller than B."""
    return dilate(erode(img, se), se)


def closing(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """A • B = (A ⊕ B) ⊖ B — removes dark details (holes, gaps) smaller than B."""
    return erode(dilate(img, se), se)


def fill_holes(mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Region filling: background pixels NOT reachable from the image border are holes.

    mask: bool array. Returns bool array with holes filled.
    connectivity: 4 or 8 — controls which background pixels are considered connected.
    """
    background = ~mask
    h, w = mask.shape
    border = np.zeros_like(mask)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    seeds = np.flatnonzero((border & background).ravel())
    if seeds.size == 0:
        return mask.copy()
    reachable = bfs_region(background, seeds, connectivity=connectivity)
    return mask | (background & ~reachable)
