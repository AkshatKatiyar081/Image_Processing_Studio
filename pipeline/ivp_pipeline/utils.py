"""Phase 1 preparation and shared helpers (colour conversion, luminance-only processing, histograms)."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from . import config
from .core.histogram import histogram
from .errors import PipelineError


def prepare(img: np.ndarray, max_side: int = config.MAX_SIDE) -> tuple[np.ndarray, float]:
    """Validate an input image and downscale it so max(H, W) <= max_side.

    Accepts uint8 RGB (H x W x 3), RGBA (alpha dropped) or grayscale (H x W).
    Returns (image, scale) where scale = new_size / original_size (<= 1).
    """
    if not isinstance(img, np.ndarray):
        raise PipelineError("BAD_IMAGE", "image must be a NumPy array")
    if img.dtype != np.uint8:
        raise PipelineError("BAD_IMAGE", f"image dtype must be uint8, got {img.dtype}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if not (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3)):
        raise PipelineError("BAD_IMAGE", f"unsupported image shape {img.shape}")
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        raise PipelineError("BAD_IMAGE", "image must be at least 8x8 pixels")
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img), scale


def is_color(img: np.ndarray) -> bool:
    return img.ndim == 3


def to_gray(img: np.ndarray) -> np.ndarray:
    """Luminance Y = 0.299 R + 0.587 G + 0.114 B."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if is_color(img) else img


def luma(img: np.ndarray) -> np.ndarray:
    """The channel luminance-only operations work on: Y of YCrCb (colour) or the image itself (gray)."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0] if is_color(img) else img


def apply_on_luma(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Run fn on Y only (colour) and merge back with the untouched Cr, Cb — avoids hue shifts."""
    if not is_color(img):
        return fn(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycc[:, :, 0] = fn(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def per_channel(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply fn to R, G, B independently (or to the single gray channel)."""
    if not is_color(img):
        return fn(img)
    return np.dstack([fn(np.ascontiguousarray(img[:, :, c])) for c in range(3)])


def hist_list(img: np.ndarray) -> list[int]:
    """256-bin histogram of the luminance channel as a plain list (JSON-friendly)."""
    return histogram(luma(img)).astype(int).tolist()


def as_rgb(img: np.ndarray) -> np.ndarray:
    return img if is_color(img) else np.dstack([img] * 3)


def overlay(img: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha: float = 0.4) -> np.ndarray:
    """RGB image with a translucent colour tint over the mask."""
    base = as_rgb(img).astype(np.float32)
    tint = np.array(color, np.float32)
    m = mask.astype(bool)
    base[m] = (1 - alpha) * base[m] + alpha * tint
    return np.clip(np.rint(base), 0, 255).astype(np.uint8)


def cutout(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGBA image whose alpha channel is the mask (encode as PNG, not JPEG)."""
    return np.dstack([as_rgb(img), np.where(mask.astype(bool), 255, 0).astype(np.uint8)])
