"""Synthetic test scenes with exact ground truth, and controlled degradations (spec §13.1–13.2).

Also handy for the dashboard: "load a demo image" buttons can use these.
"""
from __future__ import annotations

import cv2
import numpy as np


def scene(seed: int = 0, shape=(384, 512), n_objects: int = 14, polarity: str = "bright",
          fg: int = 170, bg: int = 90, texture: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Random disks and rectangles on a flat background.

    Returns (gray uint8 image, bool ground-truth mask). polarity='dark' swaps fg/bg levels
    so objects are darker than the background. `texture` adds mild smooth variation
    (sigma in gray levels) so the scene is not perfectly piecewise-constant.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for i in range(n_objects):
        cx, cy = int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40))
        if i % 3 == 2:
            hw, hh = int(rng.integers(10, 26)), int(rng.integers(10, 26))
            cv2.rectangle(mask, (cx - hw, cy - hh), (cx + hw, cy + hh), 1, -1)
        else:
            cv2.circle(mask, (cx, cy), int(rng.integers(12, 28)), 1, -1)
    gt = mask.astype(bool)
    lo, hi = (bg, fg) if polarity == "bright" else (fg, bg)
    img = np.where(gt, hi, lo).astype(np.float32)
    if texture > 0:
        t = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6)
        img += texture * t / (t.std() + 1e-6)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), gt


# ------------------------------------------------------------------ degradations
def salt_pepper(img: np.ndarray, density: float, seed: int = 0) -> np.ndarray:
    """Set `density` of pixels to 0 or 255 (half each). Colour images get the same spike in all channels."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < density / 2] = 0
    out[(r >= density / 2) & (r < density)] = 255
    return out


def gaussian_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(np.rint(img + rng.normal(0, sigma, img.shape)), 0, 255).astype(np.uint8)


def illumination_ramp(img: np.ndarray, strength: float, direction: str = "horizontal") -> np.ndarray:
    """Add a linear brightness ramp from -strength to +strength gray levels."""
    h, w = img.shape[:2]
    ramp = np.linspace(-strength, strength, w if direction == "horizontal" else h, dtype=np.float32)
    ramp = np.tile(ramp, (h, 1)) if direction == "horizontal" else np.tile(ramp[:, None], (1, w))
    if img.ndim == 3:
        ramp = ramp[:, :, None]
    return np.clip(np.rint(img + ramp), 0, 255).astype(np.uint8)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    r = np.arange(256) / 255.0
    lut = np.clip(np.rint(255 * r ** gamma), 0, 255).astype(np.uint8)
    return lut[img]


def low_contrast(img: np.ndarray, lo: int = 80, hi: int = 160) -> np.ndarray:
    """Linearly compress the full range [0, 255] into [lo, hi]."""
    return np.clip(np.rint(lo + img.astype(np.float32) * (hi - lo) / 255.0), 0, 255).astype(np.uint8)
