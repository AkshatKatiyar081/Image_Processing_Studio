"""Module 4 — Segmentation & morphological cleanup (spec §8)."""
from __future__ import annotations

import cv2
import numpy as np

from .. import metrics as Q
from ..core import histogram as Hs
from ..core import morphology as M
from ..core import region as R
from ..utils import cutout, overlay, to_gray


def _otsu_mask(gray: np.ndarray, foreground: str, engine: str):
    res = Hs.otsu(gray)  # eta is always computed from the from-scratch implementation
    if engine == "opencv":
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = int(t)
    else:
        k = res["threshold"]
    mask = gray > k
    if foreground == "dark" or (foreground == "auto" and mask.mean() > 0.5):
        mask = ~mask
    return mask, k, res["eta"]


def _grow(gray: np.ndarray, seed, p: dict, engine: str) -> np.ndarray:
    h, w = gray.shape
    if seed is None:
        x, y = w // 2, h // 2
    else:
        x = int(min(max(round(seed[0]), 0), w - 1))
        y = int(min(max(round(seed[1]), 0), h - 1))
    if engine == "opencv" and not p["adaptive"]:
        fs = R.seed_value(gray, x, y, "median3")
        accept = (np.abs(gray.astype(np.float32) - fs) <= p["delta_t"]).astype(np.uint8)
        accept[y, x] = 1  # the seed always belongs to its region
        _, labels = cv2.connectedComponents(accept, connectivity=int(p["connectivity"]))
        return labels == labels[y, x]
    return R.region_grow(gray, (x, y), p["delta_t"], int(p["connectivity"]), p["adaptive"])


def _fill_holes(mask: np.ndarray, engine: str, connectivity: int = 4) -> np.ndarray:
    if engine == "numpy":
        return M.fill_holes(mask, connectivity)
    h, w = mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)      # 1-px frame so (0, 0) is always background
    canvas[1:-1, 1:-1] = mask.astype(np.uint8) * 255
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(canvas, ff_mask, (0, 0), 128, flags=connectivity)
    return mask | (canvas[1:-1, 1:-1] == 0)


def _largest(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def cleanup(mask: np.ndarray, p: dict, engine: str) -> np.ndarray:
    """Opening then closing (removing specks first stops closing from merging them into objects)."""
    m = mask.astype(np.uint8) * 255
    se = M.disk(int(p["disk_radius"]))
    if p["morph_open"]:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se) if engine == "opencv" else M.opening(m, se)
    if p["morph_close"]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se) if engine == "opencv" else M.closing(m, se)
    out = m > 0
    if p["fill_holes"]:
        out = _fill_holes(out, engine, int(p["connectivity"]))
    if p["keep_largest"]:
        out = _largest(out)
    return out


def run(img: np.ndarray, p: dict, engine: str):
    gray = to_gray(img)
    metrics, warnings = {}, []
    if p["method"] == "otsu":
        raw, k, eta = _otsu_mask(gray, p["foreground"], engine)
        metrics["threshold"] = k
    else:
        raw = _grow(gray, p["seed"], p, engine)
        eta = Hs.otsu(gray)["eta"]
        if raw.sum() <= 1:
            warnings.append("Region did not grow beyond the seed; try a larger ΔT or another seed.")
    metrics["eta"] = round(eta, 4)
    mask = cleanup(raw, p, engine)
    metrics["foreground_fraction"] = round(float(mask.mean()), 4)
    metrics["n_components_raw"] = Q.n_components(raw)
    metrics["n_components"] = Q.n_components(mask)

    mask_u8 = (mask * 255).astype(np.uint8)
    extras = {"mask": mask_u8, "raw_mask": (raw * 255).astype(np.uint8), "cutout": cutout(img, mask)}
    return overlay(img, mask), extras, metrics, warnings
